from torch.utils.data import Dataset, ConcatDataset
import xarray as xr
import numpy as np
import torch



class ERA5dataset(Dataset):
    def __init__(
            self,
            input_data:list[str],
            input_vars:list[str],
            target_data:list[str],
            target_vars:list[str],
            static_data:str=None,
            static_vars:list[str]=None
            ):
        super().__init__()

        datasets = []
        for X, Y in zip(input_data,target_data):
             datasets.append(DataUnit(X,input_vars,Y,target_vars))
        self.dataset = ConcatDataset(datasets)

        if static_data:
            self.static_data = xr.open_dataset(static_data)
            if isinstance(static_vars,str): static_vars = [static_vars]
            missing = [v for v in static_vars if v not in self.static_data.data_vars]
            if missing:
                raise ValueError(f"Variable not in dataset: {missing}")

        self.static_vars = static_vars

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x,y = self.dataset[index]

        if self.static_vars:
            static = torch.tensor(np.array(
                [self.static_data[v].values for v in self.static_vars]
                ), dtype=torch.float32)        
            x = torch.cat([x,static],dim=0)

        return x,y

class DataUnit(Dataset):
    def __init__(
            self,
            input_data:str,
            input_vars:list[str],
            target_data:str,
            target_vars:list[str],
            ):
        super().__init__()

        self.X = xr.open_dataset(input_data)
        self.Y = xr.open_dataset(target_data)

        self.input_vars = input_vars
        self.target_vars = target_vars

        self.N = self.X[input_vars[0]].shape[0]

        if not self.X.coords.equals(self.Y.coords):
            raise ValueError(f"Non-matching coordinates between {input_data} and {target_data}.")

        missing = [v for v in input_vars if v not in self.X.data_vars]
        if missing:
             raise ValueError(f"Variable not in dataset: {missing}")

        missing = [v for v in target_vars if v not in self.Y.data_vars]
        if missing:
             raise ValueError(f"Variable not in dataset: {missing}")

    def __len__(self):
            return self.N

    def __getitem__(self, idx):
        Xt = self.X.isel(valid_time=idx)
        Yt = self.Y.isel(valid_time=idx)

        input_arr = np.stack([Xt[v].values for v in self.input_vars], axis=0)
        input_tensor = torch.tensor(input_arr, dtype=torch.float32)

        target_arr = np.stack([Yt[v].values for v in self.target_vars], axis=0)
        target_tensor = torch.tensor(target_arr, dtype=torch.float32)

        return input_tensor, target_tensor