from __future__ import annotations  
  
from pathlib import Path  
  
from .schema import DatasetMetadata, TrajectoryDataset  
  
  
class HDF5Reader:  
  
    def __init__(self, path: Path) -> None:  
        self._path = path  
  
    def read(self) -> TrajectoryDataset:  
        import h5py  
        import torch  
        import numpy as np  
  
        with h5py.File(self._path, "r") as f:  
            states = torch.from_numpy(np.array(f["states"], dtype=np.float32))  
            observations = torch.from_numpy(np.array(f["observations"], dtype=np.float32))  
            timestamps = torch.from_numpy(np.array(f["timestamps"], dtype=np.float64))  
            metadata = self._read_metadata_from_file(f)  
  
        return TrajectoryDataset(  
            states=states,  
            observations=observations,  
            timestamps=timestamps,  
            metadata=metadata,  
        )  
  
    def read_metadata(self) -> DatasetMetadata:  
        import h5py  
  
        with h5py.File(self._path, "r") as f:  
            return self._read_metadata_from_file(f)  
  
    @staticmethod  
    def _read_metadata_from_file(f) -> DatasetMetadata:  
        attrs = f["metadata"].attrs  
        return DatasetMetadata(  
            benchmark_name=str(attrs["benchmark_name"]),  
            state_dimension=int(attrs["state_dimension"]),  
            observation_dimension=int(attrs["observation_dimension"]),  
            trajectory_length=int(attrs["trajectory_length"]),  
            num_trajectories=int(attrs["num_trajectories"]),  
            random_seed=int(attrs["random_seed"]),  
            generation_time=str(attrs["generation_time"]),  
        )
