"""Writer MUST:

Open file with libver="latest"

Enable swmr_mode = True after creating datasets

Use chunked datasets

Call flush() after each write

Reader MUST:

Open file in "r" mode

Set swmr=True

Periodically refresh datasets"""

import h5py
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional

class ExperimentHDF5WriterSWMR:
    """
    HDF5 writer with live acquisition + SWMR support.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "x",
        compression: Optional[str] = "gzip",
        compression_level: int = 4,
        swmr: bool = True,
        version: float = 1.1
    ):
        self.filename = filename
        self.compression = compression
        self.compression_level = compression_level
        self.swmr = swmr


        # libver='latest' is REQUIRED for SWMR
        self.file = h5py.File(
            filename,
            mode,
            libver="latest"
        )

        # Core groups
        self.data_group = self.file.create_group("data")
        self.metadata_group = self.file.create_group("metadata")
        self.hardware_group = self.file.create_group("hardware")
        self.analysis_group = self.file.create_group("analysis")

        # Metadata
        self.metadata_group.attrs["created"] = datetime.utcnow().isoformat()
        self.metadata_group.attrs["file_version"] = version
        self.metadata_group.attrs["swmr"] = int(swmr)

        self._datasets = {}

        # NOTE:
        # swmr_mode MUST be enabled only AFTER datasets are created
        self._swmr_enabled = False

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        for key, value in metadata.items():
            self.metadata_group.attrs[key] = self._sanitize(value)
        self.flush()

    def add_hardware(self, device_name: str, config: Dict[str, Any]) -> None:
        device_group = self.hardware_group.create_group(device_name)
        for key, value in config.items():
            device_group.attrs[key] = self._sanitize(value)
        self.flush()

    # --------------------------------------------------
    # Dataset creation (MUST be chunked for SWMR)
    # --------------------------------------------------

    def create_dataset(
        self,
        name: str,
        shape: tuple,
        dtype=np.float32,
        maxshape: Optional[tuple] = None,
        chunks: Optional[tuple] = None,
    ) -> None:
        if name in self._datasets:
            raise ValueError(f"Dataset '{name}' already exists.")

        if chunks is None:
            # Default: chunk along first axis
            chunks = (1,) + shape[1:]

        dset = self.data_group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression=self.compression,
            compression_opts=self.compression_level,
        )

        self._datasets[name] = dset

    # --------------------------------------------------
    # Enable SWMR (call AFTER all datasets created)
    # --------------------------------------------------

    def enable_swmr(self) -> None:
        if self.swmr and not self._swmr_enabled:
            self.file.swmr_mode = True
            self._swmr_enabled = True
            self.flush()

    # --------------------------------------------------
    # Live append
    # --------------------------------------------------

    def append(self, name: str, data: np.ndarray, axis: int = 0) -> None:
        if not self._swmr_enabled and self.swmr:
            raise RuntimeError(
                "SWMR not enabled. Call enable_swmr() after dataset creation."
            )

        dset = self._datasets[name]
        data = np.asarray(data)

        new_shape = list(dset.shape)
        new_shape[axis] += data.shape[axis]
        dset.resize(tuple(new_shape))

        selection = [slice(None)] * dset.ndim
        start = new_shape[axis] - data.shape[axis]
        selection[axis] = slice(start, new_shape[axis])

        dset[tuple(selection)] = data

        # CRITICAL for SWMR readers
        self.flush()

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.flush()
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _sanitize(value: Any) -> Any:
        if isinstance(value, (int, float, str, np.number)):
            return value
        if isinstance(value, bool):
            return int(value)
        return str(value)

"""
#how to use: # writer side: live acquisition
writer = ExperimentHDF5WriterSWMR(
    "live_experiment.h5",
    mode="w",
    swmr=True
)

writer.add_metadata({
    "user": "Alice",
    "experiment": "live_camera_test"
})

writer.add_hardware("camera", {
    "exposure_ms": 10,
    "gain": 2
})

writer.create_dataset(
    name="images",
    shape=(0, 512, 512),
    maxshape=(None, 512, 512),
    dtype=np.uint16,
    chunks=(1, 512, 512)
)

# IMPORTANT: enable SWMR AFTER datasets are created
writer.enable_swmr()

for i in range(1000):
    img = acquire_image()
    writer.append("images", img[np.newaxis, :, :])

# Reader-side (live monitoring / plotting)
import h5py
import time
import matplotlib.pyplot as plt

with h5py.File("live_experiment.h5", "r", swmr=True) as f:
    dset = f["data/images"]

    plt.ion()
    fig, ax = plt.subplots()

    last_frame = -1

    while True:
        dset.refresh()  # REQUIRED
        n_frames = dset.shape[0]

        if n_frames > last_frame:
            img = dset[n_frames - 1]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Frame {n_frames}")
            plt.pause(0.01)
            last_frame = n_frames

        time.sleep(0.1)"""
