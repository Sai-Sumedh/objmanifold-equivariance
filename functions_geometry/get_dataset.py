
import torch
import numpy as np
from torch.utils.data import Dataset

class MNISTRotated(Dataset):

    def __init__(self, path_to_data):
        """
        path_to_data: path specifying location of dataset
        once loaded, keys 'data' and 'labels' have arrays of shape
        (#datapoints, 1, image-size0, image-size1), and (#datapoints,) respectively
        """
        data_file = np.load(path_to_data)
        self.data = torch.from_numpy(data_file['data']).type(torch.float32)
        self.labels = torch.from_numpy(data_file['labels']).type(torch.float32)

    def __len__(self):
        return self.labels.shape[0]
        
    def __getitem__(self, idx):
        return [self.data[idx,:,:,:], self.labels[idx]]