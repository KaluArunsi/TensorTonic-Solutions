import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    Returns: (features, label) from __getitem__ where features is float32 (D,) and label is float32 (1,)
    """

    def __init__(self, data, label_col):
        self.data = torch.tensor(data)
        self.label_col = label_col
        
        all_cols = torch.arange(self.data.shape[1])
        feature_cols = all_cols[all_cols != self.label_col]
        
        self.feature = self.data[:, feature_cols].float()
        self.label = self.data[:, [self.label_col]].float()

    def __len__(self):
        self.len = len(self.data)
        return self.len

    def __getitem__(self, idx):
        self.idx = idx
        feature_tensor = torch.tensor(self.feature[idx])
        label_tensor = torch.tensor(self.label[idx])

        return (feature_tensor, label_tensor)