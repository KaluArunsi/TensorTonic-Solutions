import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns: a DataLoader that oversamples underrepresented classes
    """
    dataset = TensorDataset(features, labels)

    classes, count = torch.unique(labels, return_counts=True)
    class_weights = 1 / count.float()

    sample_weights = class_weights[labels.long().view(-1)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return dataloader