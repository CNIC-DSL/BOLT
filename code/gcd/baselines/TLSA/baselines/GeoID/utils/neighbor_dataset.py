import torch
import numpy as np
from torch.utils.data import Dataset

class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = list(self.dataset.__getitem__(index))
        neighbor_index = np.random.choice(self.indices[index], 10)[0]
        neighbor = list(self.dataset.__getitem__(neighbor_index))

        output['anchor'] = anchor[:3]
        output['neighbor'] = neighbor[:3]
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor[-2]
        output['true'] = anchor[-1]
        output['index'] = index
        return output
