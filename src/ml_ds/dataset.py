from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Subset

EPS = 1e-6


@dataclass(frozen=True)
class VariableGroups:
    dynamic_inputs: list[str]
    static_inputs: list[str]
    targets: list[str]


class ZarrVariableDiscovery:
    def __init__(self, static_candidates: Sequence[str] = ("x_lsm", "x_orog")):
        self.static_candidates = tuple(static_candidates)

    def discover(self, data: xr.Dataset) -> VariableGroups:
        dynamic_inputs = sorted(
            [
                name
                for name, array in data.data_vars.items()
                if name.startswith("x_") and array.ndim == 3
            ]
        )
        static_inputs = [
            name for name in self.static_candidates if name in data and data[name].ndim == 2
        ]
        targets = sorted(
            [
                name
                for name, array in data.data_vars.items()
                if name.startswith("y_") and array.ndim == 3
            ]
        )

        if not dynamic_inputs:
            raise ValueError(
                "No dynamic predictor variables found. Expected 3D variables starting with 'x_'."
            )
        if len(static_inputs) != len(self.static_candidates):
            raise ValueError("Missing static predictors. Expected 2D variables: x_lsm and x_orog.")
        if len(targets) != 3:
            raise ValueError("Expected exactly 3 target variables with prefix 'y_'.")

        return VariableGroups(
            dynamic_inputs=dynamic_inputs, static_inputs=static_inputs, targets=targets
        )


class ZarrNormalizationStats:
    def __init__(self, stats_path: str | Path):
        self.stats_path = Path(stats_path)
        self.stats = xr.open_zarr(self.stats_path)

    def tensor_stats(self, variables: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        means: list[float] = []
        stds: list[float] = []
        for variable in variables:
            mean, std = self._extract_mean_std(variable)
            means.append(mean)
            stds.append(max(std, EPS))

        mean_tensor = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)
        return mean_tensor, std_tensor

    def _extract_mean_std(self, variable: str) -> tuple[float, float]:
        if variable not in self.stats:
            raise ValueError(f"Variable '{variable}' was not found in normalization stats zarr.")

        array = self.stats[variable]
        dim_candidates = ["stat", "stats", "statistic", "statistics"]
        stat_dim = next((dim for dim in dim_candidates if dim in array.dims), None)

        if stat_dim is not None:
            coord_values = [str(v).lower() for v in array.coords[stat_dim].values]
            if "mean" in coord_values and "std" in coord_values:
                mean = float(array.sel({stat_dim: "mean"}).values)
                std = float(array.sel({stat_dim: "std"}).values)
                return mean, std
            if array.sizes[stat_dim] >= 2:
                mean = float(array.isel({stat_dim: 0}).values)
                std = float(array.isel({stat_dim: 1}).values)
                return mean, std

        mean_key = f"{variable}_mean"
        std_key = f"{variable}_std"
        if mean_key in self.stats and std_key in self.stats:
            return float(self.stats[mean_key].values), float(self.stats[std_key].values)

        if "mean" in array.attrs and "std" in array.attrs:
            return float(array.attrs["mean"]), float(array.attrs["std"])

        raise ValueError(
            f"Could not infer mean/std format for variable '{variable}' in {self.stats_path}."
        )


class ERA5ZarrDataset(Dataset):
    def __init__(
        self,
        input_file: str | Path,
        stats_file: str | Path,
        indices: Sequence[int] | None = None,
        variable_groups: VariableGroups | None = None,
    ):
        super().__init__()

        self.input_file = Path(input_file)
        self.data = xr.open_zarr(self.input_file)

        discovery = ZarrVariableDiscovery()
        self.variable_groups = variable_groups or discovery.discover(self.data)

        self.input_vars = self.variable_groups.dynamic_inputs + self.variable_groups.static_inputs
        self.target_vars = self.variable_groups.targets

        self._time_dims = {
            var: self.data[var].dims[0]
            for var in (self.variable_groups.dynamic_inputs + self.variable_groups.targets)
        }
        total_samples = int(self.data[self.variable_groups.dynamic_inputs[0]].shape[0])
        self.indices = np.asarray(
            indices if indices is not None else np.arange(total_samples), dtype=np.int64
        )

        self.static_tensor = self._load_static_tensor()

        stats = ZarrNormalizationStats(stats_file)
        self.input_means, self.input_stds = stats.tensor_stats(self.input_vars)
        self.target_means, self.target_stds = stats.tensor_stats(self.target_vars)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_index = int(self.indices[index])

        dynamic = np.stack(
            [
                self.data[var].isel({self._time_dims[var]: time_index}).values
                for var in self.variable_groups.dynamic_inputs
            ],
            axis=0,
        )
        dynamic_tensor = torch.tensor(dynamic, dtype=torch.float32)
        x = torch.cat([dynamic_tensor, self.static_tensor], dim=0)

        targets = np.stack(
            [
                self.data[var].isel({self._time_dims[var]: time_index}).values
                for var in self.variable_groups.targets
            ],
            axis=0,
        )
        y = torch.tensor(targets, dtype=torch.float32)

        x = (x - self.input_means) / self.input_stds
        y = (y - self.target_means) / self.target_stds
        return x, y

    def _load_static_tensor(self) -> torch.Tensor:
        static = np.stack(
            [self.data[var].values for var in self.variable_groups.static_inputs], axis=0
        )
        return torch.tensor(static, dtype=torch.float32)


def build_split_subsets(
    dataset: Dataset,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[Subset, Subset]:
    n_samples = len(dataset)
    val_size = int(n_samples * val_fraction)
    test_size = int(n_samples * test_fraction)
    val_start = max(n_samples - (val_size + test_size), 0)
    test_start = max(n_samples - test_size, 0)

    val_subset = Subset(dataset, list(range(val_start, test_start)))
    test_subset = Subset(dataset, list(range(test_start, n_samples)))
    return val_subset, test_subset
