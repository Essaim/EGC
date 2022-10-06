import torch, os
import torch.nn as nn
import numpy as np
import Script.normalization as normal

from torch.utils.data import Dataset, DataLoader


class multi_dataset(Dataset):
    def __init__(self, data_path, in_dim, out_dim, scaler):
        super(multi_dataset, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaler = scaler
        data = np.load(data_path)

        self.inputs = scaler.transform(data['x'])
        self.outputs = scaler.transform(data['y'])
        assert self.inputs.shape[0] == self.outputs.shape[0]
        assert self.inputs.shape[2] == self.outputs.shape[2]



    def __getitem__(self, item):
        x, y = self.inputs[item][..., :self.in_dim], self.outputs[item][..., :self.out_dim]
        return torch.tensor(x).float(), torch.tensor(y).float()

    def __len__(self):
        return self.inputs.shape[0]

class single_dataset(Dataset):
    def __init__(self, data_path, in_dim, out_dim, scaler):
        super(single_dataset, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaler = scaler
        data = np.load(data_path)

        self.inputs = scaler.transform(data['x'])
        self.outputs = scaler.transform(data['y'][:,out_dim,:,:])


    def __getitem__(self, item):
        x, y = self.inputs[item][..., :self.in_dim], self.outputs[item][..., :self.out_dim]
        return torch.tensor(x).float(), torch.tensor(y).float()

    def __len__(self):
        return self.inputs.shape[0]


def get_dataloader(dataset_id: str,
                   batch_size: int,
                   scaler_id: str,
                   input_dim: int,
                   output_dim: int,
                   out_catagory: str,
                   num_workers: int):
    scaler_data = np.load(os.path.join('Data', dataset_id, 'train.npz'))
    scaler = getattr(normal, scaler_id)(scaler_data['x'][..., 0])


    if out_catagory == 'multi':
        my_datasets = {
            key: multi_dataset(data_path=os.path.join('Data', dataset_id, key + '.npz'), in_dim=input_dim, out_dim=output_dim,
                            scaler=scaler) for key in ['train', 'val', 'test']}

        my_dataloader = {
            key: DataLoader(dataset=ds, batch_size=batch_size, shuffle=(key == 'train'),
                            num_workers=num_workers) for key,ds in my_datasets.items()
        }
        my_dataloader['scaler'] = scaler
    else:
        my_datasets = {
            key: single_dataset(data_path=os.path.join('Data', dataset_id, key + '.npz'), in_dim=input_dim,
                               out_dim=output_dim,
                               scaler=scaler) for key in ['train', 'val', 'test']}
        my_dataloader = {
            key: DataLoader(dataset=ds, batch_size=batch_size, shuffle=(key == 'train'),
                            num_workers=num_workers) for key, ds in my_datasets.items()
        }
        my_dataloader['scaler'] = scaler

    return my_dataloader


