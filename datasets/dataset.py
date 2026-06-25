from __future__ import annotations  
  
from pathlib import Path  
  
from .hdf5_reader import HDF5Reader  
from .schema import DatasetMetadata, TrajectoryDataset  
  
_VALID_SPLITS = {"train", "val", "test"}  
  
  
def load_dataset(path: Path) -> TrajectoryDataset:  
    return HDF5Reader(path).read()  
  
  
def load_metadata(path: Path) -> DatasetMetadata:  
    return HDF5Reader(path).read_metadata()  
  
  
def load_split(dataset_dir: Path, split: str) -> TrajectoryDataset:  
    if split not in _VALID_SPLITS:  
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")  
    return load_dataset(dataset_dir / f"{split}.h5")
