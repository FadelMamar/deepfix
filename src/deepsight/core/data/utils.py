from torch.utils.data import Dataset
import torch
from typing import Optional, Union,List,Dict

class DataStatistics:
    def __init__(self,
                 train_data:Dataset,
                 test_data:Optional[Dataset]=None,
                 ):
        self.train_data = train_data
        self.test_data = test_data
    
    def get_statistics(self,)->Dict[str,Union[int,List[float]]]:
        stats = dict(self.get_train_statistics(),**self.get_test_statistics())
        return stats
    
    def get_train_statistics(self,)->Dict[str,Union[int,List[float]]]:
        num_samples = len(self.train_data)
        mean_and_std = self.get_dataset_images_statistics(self.train_data)
        stats = dict(num_train_samples=num_samples,train_mean=mean_and_std['mean'],train_std=mean_and_std['std'])
        return stats
    
    def get_test_statistics(self,)->Dict[str,Union[int,List[float]]]:
        num_samples = len(self.test_data)
        mean_and_std = self.get_dataset_images_statistics(self.test_data)
        stats = dict(num_test_samples=num_samples,test_mean=mean_and_std['mean'],test_std=mean_and_std['std'])
        return stats

    def get_dataset_images_statistics(self, dataset: Dataset)->Dict[str,List[float]]:
        """
        Compute mean and standard deviation statistics for an image dataset.
        
        Args:
            dataset: PyTorch Dataset containing images in (C, H, W) format
            
        Returns:
            dict: Dictionary containing 'mean' and 'std' for each channel
        """
        assert isinstance(dataset, Dataset), "dataset must be a PyTorch Dataset"

        # Initialize accumulators
        num_samples = len(dataset)
        if num_samples == 0:
            return {"mean": None, "std": None}
        
        # Get first sample to determine number of channels
        first_sample = next(iter(dataset))
        if isinstance(first_sample, tuple):
            # Handle case where dataset returns (image, label) tuples
            first_image = first_sample[0]
        else:
            # Handle case where dataset returns just the image
            first_image = first_sample
        
        num_channels = first_image.shape[0]  # C dimension
        
        # Initialize accumulators for mean and variance computation
        sum_pixels = torch.zeros(num_channels)
        sum_squared_pixels = torch.zeros(num_channels)
        total_pixels = 0
        
        # Iterate through all samples in the dataset
        for sample in dataset:
            if isinstance(sample, tuple):
                image = sample[0]
            else:
                image = sample
                
            # Ensure image is a tensor
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image)
            
            # Flatten spatial dimensions (H, W) and keep channel dimension
            image_flat = image.view(num_channels, -1)  # Shape: (C, H*W)
            
            # Accumulate sums
            sum_pixels += image_flat.sum(dim=1)  # Sum over spatial dimensions
            sum_squared_pixels += (image_flat ** 2).sum(dim=1)
            total_pixels += image_flat.shape[1]  # H*W
        
        # Compute mean and standard deviation
        mean = sum_pixels / total_pixels
        
        # Compute variance using the formula: Var(X) = E[X^2] - E[X]^2
        mean_squared = (sum_squared_pixels / total_pixels)
        variance = mean_squared - (mean ** 2)
        std = torch.sqrt(variance)
        
        return {
            "mean": mean.tolist(),
            "std": std.tolist()
        }
