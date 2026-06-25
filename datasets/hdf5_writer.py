from __future__ import annotations  
  
from pathlib import Path  
  
import numpy as np  
  
from .schema import DatasetMetadata  
  
  
class HDF5Writer:  
  
    def __init__(self, path: Path) -> None:  
        self._path = path  
  
    def write(  
        self,  
        states: np.ndarray,  
        observations: np.ndarray,  
        timestamps: np.ndarray,  
        metadata: DatasetMetadata,  
    ) -> None:  
        import h5py  
  
        self._path.parent.mkdir(parents=True, exist_ok=True)  
  
        with h5py.File(self._path, "w") as f:  
            f.create_dataset("states", data=states.astype("float32"))  
            f.create_dataset("observations", data=observations.astype("float32"))  
            f.create_dataset("timestamps", data=timestamps.astype("float64"))  
  
            grp = f.create_group("metadata")  
            grp.attrs["benchmark_name"] = metadata.benchmark_name  
            grp.attrs["state_dimension"] = metadata.state_dimension  
            grp.attrs["observation_dimension"] = metadata.observation_dimension  
            grp.attrs["trajectory_length"] = metadata.trajectory_length  
            grp.attrs["num_trajectories"] = metadata.num_trajectories  
            grp.attrs["random_seed"] = metadata.random_seed  
            grp.attrs["generation_time"] = metadata.generation_time
