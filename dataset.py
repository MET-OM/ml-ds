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
            static_vars:list[str]=None,
            mean_sd:dict[str,float]=None,
            ):
        super().__init__()

        # Initialize sub-datasets (one per file pair)
        datasets = []
        for X, Y in zip(input_data,target_data):
             datasets.append(DataUnit(X,input_vars,Y,target_vars))
        self.dataset = ConcatDataset(datasets)

        # Variables
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.static_vars = static_vars

        # Files
        self.input_data = input_data
        self.target_data = target_data

        # Static data
        if self.static_vars:
            self.static_data = xr.open_dataset(static_data)

        # Mean/sd for each variable
        if mean_sd is None:
            self.mean_sd = self._compute_mean_sd()
        else:
            self.mean_sd = mean_sd
        
        # Get mean and sd for input and target, and reshape to match dimensions
        inp_vars = input_vars + static_vars if static_vars else input_vars
        self.input_means = torch.tensor([self.mean_sd[var][0] for var in inp_vars],
                                        dtype=torch.float32).view(-1, 1, 1)
        self.input_sds = torch.tensor([self.mean_sd[var][1] for var in inp_vars],
                                      dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x,y = self.dataset[index]

        if self.static_vars:
            static = torch.tensor(np.array(
                [self.static_data[v].values for v in self.static_vars]
                ), dtype=torch.float32)        
            x = torch.cat([x,static],dim=0)

        x = (x - self.input_means) / self.input_sds

        return x,y

    def _compute_mean_sd(self):
        sums = {}
        sums_sq = {}
        counts = {}

        # Initialize dicts for all variables to normalize
        vars_to_compute = self.input_vars + (self.static_vars if self.static_vars else [])
        for var in vars_to_compute:
            sums[var] = 0.0
            sums_sq[var] = 0.0
            counts[var] = 0

        # Compute mean/std for dynamic input variables
        for file_path in self.input_data:
            print(f"Computing mean/std from {file_path}...")
            with xr.open_dataset(file_path) as ds:
                for var in self.input_vars:
                    arr = ds[var].values
                    sums[var]     += arr.sum()
                    sums_sq[var]  += (arr**2).sum()
                    counts[var]   += arr.size

        # Compute for static variables
        if self.static_vars:
            for var in self.static_vars:
                arr = self.static_data[var].values
                sums[var]    += arr.sum()
                sums_sq[var] += (arr**2).sum()
                counts[var]  += arr.size

        # Finalize means/stds
        mean_sd = {}
        for var in vars_to_compute:
            mean = sums[var] / counts[var]
            sd = np.sqrt((sums_sq[var] / counts[var]) - mean**2)
            mean_sd[var] = (mean, sd)

        return mean_sd
    

    
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