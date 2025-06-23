from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from kronfluence.task import Task
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs


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
        outputs = model(**batch)
        # Sum over batch dimension as required by kronfluence
        return outputs.loss.sum()

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the measurement (using loss as measurement)."""
        outputs = model(**batch)
        return outputs.loss

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Get list of modules to track for influence computation."""
        # Track all Linear layers except embeddings
        return None  # Will be set in filter_dataset based on model inspection
    
    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Get attention mask from the batch."""
        return batch.get("attention_mask", None)


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
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        query_dataset = query_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=query_dataset.column_names
        )
        
        return train_dataset, query_dataset
    
    def filter_dataset(
        self,
        train_dataset: Dataset,
        query_dataset: Dataset,
        text_column: str = "text",
        label_column: str = "label",
        strategy: str = "ekfac",
        k: int = 1000,  # Number of examples to keep
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> Dataset:
        """Filter dataset using influence scores.
        
        Args:
            train_dataset: Training dataset to filter
            query_dataset: Query dataset to compute influence against
            text_column: Name of text column in datasets
            label_column: Name of label column in datasets
            strategy: Influence computation strategy ('identity', 'diagonal', 'kfac', or 'ekfac')
            k: Number of most influential examples to keep
            batch_size: Batch size for influence computation
            num_workers: Number of workers for data loading
            
        Returns:
            Dataset: Filtered dataset containing k most influential examples
        """
        # Prepare datasets
        train_dataset, query_dataset = self._prepare_datasets(
            train_dataset, query_dataset, text_column, label_column
        )
        
        # Prepare model for influence computation
        model = prepare_model(model=self.model, task=self.task)
        
        # Initialize analyzer
        analyzer = Analyzer(
            analysis_name="sentiment_influence",
            model=model,
            task=self.task,
            cache_dir=self.cache_dir
        )
        
        # Set up dataloader kwargs
        dataloader_kwargs = DataLoaderKwargs(
            num_workers=num_workers,
            pin_memory=True,
            batch_size=batch_size
        )
        analyzer.set_dataloader_kwargs(dataloader_kwargs)
        
        # Configure factor arguments
        factor_args = FactorArguments(
            strategy=strategy,
            use_empirical_fisher=False,
            amp_dtype=torch.float16,  # Use mixed precision for efficiency
            has_shared_parameters=False
        )
        
        # Compute factors
        print("Computing influence factors...")
        analyzer.fit_all_factors(
            factors_name="sentiment_factors",
            dataset=train_dataset,
            factor_args=factor_args
        )
        
        # Compute pairwise influence scores
        print("Computing influence scores...")
        scores = analyzer.compute_pairwise_scores(
            scores_name="sentiment_scores",
            factors_name="sentiment_factors",
            query_dataset=query_dataset,
            train_dataset=train_dataset,
            per_device_query_batch_size=batch_size
        )
        
        # Get indices of top k influential examples
        influence_scores = scores.mean(dim=0)  # Average over query examples
        top_k_indices = torch.topk(influence_scores, k=k, largest=True).indices.tolist()
        
        # Filter the original dataset
        filtered_dataset = train_dataset.select(top_k_indices)
        
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
    
    # Filter dataset
    filtered_dataset = influence_filter.filter_dataset(
        train_dataset=train_dataset,
        query_dataset=test_dataset,
        k=1000  # Keep top 1000 influential examples
    )
    
    print(f"Original dataset size: {len(train_dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")


if __name__ == "__main__":
    main()
