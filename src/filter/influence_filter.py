from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import os
import torch.multiprocessing as mp

import torch
from torch import nn
import numpy as np
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from kronfluence.task import Task
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs
from torch.utils.data import Dataset

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SentimentTask(Task):
    """Task for sentiment classification using transformers."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
    def compute_train_loss(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        """Compute the training loss for a batch."""
        input_ids, attention_mask, labels = batch
        
        # Forward pass through the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Scale by batch size for sum reduction
        return outputs.loss * input_ids.size(0)

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the measurement (using loss as measurement)."""
        return self.compute_train_loss(batch, model, sample=False)

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Get list of modules to track for influence computation."""
        # Track all modules by default
        return None
    
    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Get attention mask from the batch."""
        # Return None if no attention mask to let the model handle it internally
        if "attention_mask" not in batch:
            return None
        # Return the attention mask as a single tensor
        return batch["attention_mask"]


class HFDatasetWrapper(Dataset):
    """Wrapper to make Hugging Face Dataset compatible with PyTorch DataLoader."""
    
    def __init__(self, hf_dataset: Dataset, device: str = "cpu"):
        self.hf_dataset = hf_dataset
        self.device = device
        
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.hf_dataset[idx]
        # Always create tensors on CPU first
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        
        # Let DataLoader handle device placement if using pin_memory
        return input_ids, attention_mask, labels


class InfluenceFilter:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cache_dir: str = "influence_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.task = SentimentTask(tokenizer)
        
    def _prepare_datasets(
        self,
        train_dataset: Dataset,
        query_dataset: Dataset,
        text_column: str = "text",
        label_column: str = "label",
    ) -> tuple[Dataset, Dataset]:
        """Prepare datasets by tokenizing and formatting."""
        # Store original examples before tokenization
        self.original_train_examples = [
            {"text": ex[text_column], "label": ex[label_column]}
            for ex in train_dataset
        ]
        self.original_query_examples = [
            {"text": ex[text_column], "label": ex[label_column]}
            for ex in query_dataset
        ]
        
        def tokenize_and_format(examples):
            # Tokenize the text inputs
            tokenized = self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                return_tensors=None  # Let kronfluence handle tensor conversion
            )
            
            # Add labels to the tokenized output
            tokenized["labels"] = examples[label_column]
            
            return tokenized
        
        # Tokenize and format datasets
        train_dataset = train_dataset.map(
            tokenize_and_format,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        query_dataset = query_dataset.map(
            tokenize_and_format,
            batched=True,
            remove_columns=query_dataset.column_names
        )
        
        return train_dataset, query_dataset

    def compute_and_save_influences(
        self,
        train_dataset: Dataset,
        query_dataset: Dataset,
        output_file: str,
        text_column: str = "text",
        label_column: str = "label",
        strategy: str = "ekfac",
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """Compute influence scores and save them to a JSON file."""
        import time
        start_time = time.time()
        
        # Adjust num_workers based on device
        if self.device == "cuda":
            num_workers = min(2, num_workers)
        else:
            num_workers = num_workers

        # Prepare datasets
        dataset_start = time.time()
        train_dataset, query_dataset = self._prepare_datasets(
            train_dataset, query_dataset, text_column, label_column
        )
        dataset_time = time.time() - dataset_start
        
        # Wrap datasets for proper device handling
        train_dataset_wrapped = HFDatasetWrapper(train_dataset, device=self.device)
        query_dataset_wrapped = HFDatasetWrapper(query_dataset, device=self.device)
        
        # Prepare model for influence computation
        model = prepare_model(model=self.model, task=self.task)
        
        # Move model to correct device
        if self.device == "cuda":
            model = model.cuda()
            for param in model.parameters():
                if not param.is_cuda:
                    param.data = param.data.cuda()
        else:
            model = model.cpu()
            
        # Initialize analyzer with explicit device setting
        analyzer = Analyzer(
            analysis_name="sentiment_influence",
            model=model,
            task=self.task,
            cpu=(self.device == "cpu"),
            disable_tqdm=False
        )
        
        # Set up dataloader kwargs with appropriate settings
        dataloader_kwargs = DataLoaderKwargs(
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        analyzer.set_dataloader_kwargs(dataloader_kwargs)
        
        # Configure factor arguments
        factor_args = FactorArguments(
            strategy=strategy,
            use_empirical_fisher=True,
            amp_dtype=torch.float16,
            eigendecomposition_dtype=torch.float64,
            activation_covariance_dtype=torch.float32,
            gradient_covariance_dtype=torch.float32,
            has_shared_parameters=False
        )
        
        try:
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                print(f"CUDA memory before factor computation: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            # Debug: Print model architecture
            print("\nModel Architecture:")
            for name, module in model.named_modules():
                if len(name) > 0:  # Skip the root module
                    print(f"  {name}: {type(module).__name__}")
            
            print("\nInfluence tracking settings:")
            print(f"  Task tracked modules: {self.task.get_influence_tracked_modules()}")
            
            # Compute factors
            print("\nComputing influence factors...")
            factor_start = time.time()
            analyzer.fit_all_factors(
                factors_name="sentiment_factors",
                dataset=train_dataset_wrapped,
                factor_args=factor_args,
                per_device_batch_size=1 if self.device == "cuda" else batch_size,
                overwrite_output_dir=True
            )
            factor_time = time.time() - factor_start
            
            # Clear CUDA cache again before score computation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                print(f"CUDA memory before score computation: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            # Compute pairwise scores
            print("Computing influence scores...")
            score_start = time.time()
            analyzer.compute_pairwise_scores(
                scores_name="sentiment_scores",
                factors_name="sentiment_factors",
                query_dataset=query_dataset_wrapped,
                train_dataset=train_dataset_wrapped,
                per_device_query_batch_size=1,
                per_device_train_batch_size=1 if self.device == "cuda" else batch_size,
                overwrite_output_dir=True
            )
            score_time = time.time() - score_start
            
            # Load the computed scores
            print("Loading computed scores...")
            scores_dict = analyzer.load_pairwise_scores("sentiment_scores")
            scores = scores_dict["all_modules"]
            
            if scores is None:
                raise RuntimeError("Failed to compute influence scores - scores are None")
            
            # Average influence scores over query examples
            print("Computing final influence scores...")
            influence_scores = scores.mean(dim=0).cpu().numpy()
            
            # Extract query and training examples for reference
            query_examples = []
            for i in range(len(query_dataset)):
                query_examples.append({
                    "idx": i,
                    "text": self.original_query_examples[i]["text"],
                    "label": self.original_query_examples[i]["label"]
                })
            
            # Get top influenced training examples
            top_k = min(10, len(train_dataset))  # Get top 10 examples or all if less
            top_indices = np.argsort(influence_scores)[-top_k:][::-1]
            top_examples = []
            for idx in top_indices:
                top_examples.append({
                    "idx": int(idx),
                    "text": self.original_train_examples[idx]["text"],
                    "label": self.original_train_examples[idx]["label"],
                    "influence_score": float(influence_scores[idx])
                })
            
            # Save scores and metadata
            total_time = time.time() - start_time
            output_data = {
                "influence_scores": influence_scores.tolist(),
                "metadata": {
                    "num_train_examples": len(train_dataset),
                    "num_query_examples": len(query_dataset),
                    "strategy": strategy,
                    "device": self.device,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "timing": {
                        "total_time": total_time,
                        "dataset_preparation": dataset_time,
                        "factor_computation": factor_time,
                        "score_computation": score_time
                    }
                },
                "query_examples": query_examples,
                "top_influenced_examples": top_examples,
                "model_info": {
                    "name": self.model.name_or_path if hasattr(self.model, "name_or_path") else str(type(self.model)),
                    "num_parameters": sum(p.numel() for p in self.model.parameters())
                }
            }
            
            # Create output directory if needed
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            print(f"Influence scores saved to: {output_file}")
            print(f"\nTiming Summary:")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Dataset preparation: {dataset_time:.2f} seconds")
            print(f"  Factor computation: {factor_time:.2f} seconds")
            print(f"  Score computation: {score_time:.2f} seconds")
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                print(f"CUDA error during computation: {e}")
                print("Try reducing batch size or using CPU device")
                raise
            else:
                print(f"Error during computation: {e}")
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @staticmethod
    def get_top_k_from_saved(
        influence_file: str,
        train_dataset: Dataset,
        k: int,
    ) -> Dataset:
        """Get top-k examples from saved influence scores.
        
        Args:
            influence_file: Path to saved influence scores JSON
            train_dataset: Original training dataset to filter
            k: Number of most influential examples to keep
            
        Returns:
            Dataset: Filtered dataset containing k most influential examples
        """
        # Load influence scores
        with open(influence_file, 'r') as f:
            data = json.load(f)
        
        influence_scores = np.array(data["influence_scores"])
        
        # Get indices of top k influential examples
        top_k_indices = np.argsort(influence_scores)[-k:]
        
        # Filter the dataset
        filtered_dataset = train_dataset.select(top_k_indices.tolist())
        
        print(f"Filtered dataset from {len(train_dataset)} to {len(filtered_dataset)} examples")
        return filtered_dataset


def main():
    """Example usage"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    
    # Load model and tokenizer
    model_name = "distilbert/distilgpt2"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets
    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")
    
    # Initialize filter
    influence_filter = InfluenceFilter(model, tokenizer)
    
    # Compute and save all influence scores
    influence_file = "src/data/distilgpt2/influence_scores.json"
    influence_filter.compute_and_save_influences(
        train_dataset=train_dataset,
        query_dataset=test_dataset,
        output_file=influence_file,
        batch_size=32
    )
    
    # Later, you can get different sized filtered datasets without recomputing:
    '''for k in [1000, 2000, 5000]:
        filtered_dataset = InfluenceFilter.get_top_k_from_saved(
            influence_file=influence_file,
            train_dataset=train_dataset,
            k=k
        )
        print(f"Got filtered dataset with {k} examples")'''


if __name__ == "__main__":
    main()
