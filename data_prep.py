# Custom data preparation functions and transformations

import torch

def compute_min_max(dataset):
    """
    Finds the min and max values in a dataset (of tensors). Used to find the
    min and max of an unscaled dataset.
    
    :param dataset: The dataset to find the min and max of
    """
    data_min = float("inf")
    data_max = float("-inf")
    for i in range(len(dataset)):
        x, _ = dataset[i]
        data_min = min(data_min, x.min().item())
        data_max = max(data_max, x.max().item())
    return data_min, data_max

def compute_mean_std(dataset):
    """
    Computes the mean and standard deviation of a dataset. Used in z-score
    normalization.
    
    :param dataset: Dataset to compute the mean and std dev for
    """
    x0, _ = dataset[0] # get first image from dataset to extract dimensions
    c = x0.shape[0]

    sum_n = torch.zeros(c)
    sum_sq = torch.zeros(c)
    num_pixels = 0

    for i in range(len(dataset)):
        x, _ = dataset[i]
        sum_n += x.sum(dim=(1,2))
        sum_sq += (x ** 2).sum(dim=(1,2))
        num_pixels += x.shape[1] * x.shape[2]
    
    mean = sum_n / num_pixels
    std = torch.sqrt(sum_sq / num_pixels - mean ** 2)

    return mean, std

class MinMaxScaling:
    """
    Custom transformation for min-max scaling.

    Returns scaled tensor on call
    """
    def __init__(self, min_unscaled, max_unscaled, min_scaled, max_scaled):
        self.min_unscaled = min_unscaled
        self.max_unscaled = max_unscaled
        self.min_scaled = min_scaled
        self.max_scaled = max_scaled
    
    def __call__(self, x):
        return ((x-self.min_unscaled)/(self.max_unscaled-self.min_unscaled))*(self.max_scaled-self.min_scaled)+self.min_scaled