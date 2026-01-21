"""Writer MUST:

Open file with libver="latest"

Enable swmr_mode = True after creating datasets

Use chunked datasets

Call flush() after each write

Reader MUST:

Open file in "r" mode

Set swmr=True

Periodically refresh datasets"""

from datetime import datetime
from typing import Any, Dict, Optional
import time
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np

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
        self.data_group = self.file.require_group("data")
        self.metadata_group = self.file.require_group("metadata")
        self.hardware_group = self.file.require_group("hardware")
        self.analysis_group = self.file.require_group("analysis")

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

"""


class ExperimentHDF5ReaderSWMR:
    """
    HDF5 SWMR reader with live monitoring and inspection utilities.
    Compatible with ExperimentHDF5WriterSWMR.
    """

    def __init__(
        self,
        filename: str,
        swmr: bool = True,
    ):
        self.filename = filename
        self.swmr = swmr

        # SWMR readers MUST open file with swmr=True
        self.file = h5py.File(
            filename,
            "r",
            libver="latest",
            swmr=swmr,
        )

    # --------------------------------------------------
    # Dataset access helpers
    # --------------------------------------------------

    def get_dataset(self, path: str) -> h5py.Dataset:
        if path not in self.file:
            raise KeyError(f"Dataset '{path}' not found in file.")
        obj = self.file[path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"'{path}' is not a dataset.")
        return obj

    # --------------------------------------------------
    # Live monitoring / plotting
    # --------------------------------------------------

    def live_monitoring_plotting(
        self,
        dataset: str = "data/images",
        refresh_interval: float = 0.1,
        cmap: str = "gray",
    ) -> None:
        """
        Live plot the last frame of a growing dataset.
        Ctrl+C cleanly exits.
        """

        dset = self.get_dataset(dataset)

        plt.ion()
        fig, ax = plt.subplots()
        im = None

        last_frame = 0

        try:
            while True:
                # REQUIRED for SWMR readers
                dset.refresh()

                n_frames = dset.shape[0]
                if n_frames == 0:
                    time.sleep(refresh_interval)
                    continue

                if n_frames > last_frame:
                    frame = dset[n_frames - 1]

                    if im is None:
                        im = ax.imshow(frame, cmap=cmap)
                        ax.set_title(f"Frame {n_frames}")
                        plt.colorbar(im, ax=ax)
                    else:
                        im.set_data(frame)
                        ax.set_title(f"Frame {n_frames}")
                        im.autoscale()

                    plt.pause(0.001)
                    last_frame = n_frames

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\nLive monitoring stopped by user.")

        finally:
            plt.ioff()
            plt.close(fig)

    # --------------------------------------------------
    # File inspection / debugging
    # --------------------------------------------------

    def read_structure(self) -> None:
        """
        Print full file tree, dataset shapes, dtypes, and attributes.
        """

        def walk(name, obj):
            print("\n" + "=" * 60)
            print(name)
            print("-" * 60)

            if isinstance(obj, h5py.Dataset):
                print("DATASET")
                print("  shape:", obj.shape)
                print("  dtype:", obj.dtype)
                print("  chunks:", obj.chunks)
            else:
                print("GROUP")

            if obj.attrs:
                print("ATTRIBUTES:")
                for k, v in obj.attrs.items():
                    print(f"  {k}: {v}")
            else:
                print("ATTRIBUTES: <none>")

        self.file.visititems(walk)

    def get_structure(self):
        """
        Extract positioning data into a structured dict.

        Returns:
            {
              "INITIAL": {
                  "top_left": {...},
                  "top_right": {...},
                  "bottom_left": {...},
                  "bottom_right": {...},
                  "nv": {...} or None
              },
              "FINAL": {
                  "top_left": {...},
                  "top_right": {...},
                  "bottom_left": {...},
                  "bottom_right": {...}
              }
            }
        """
        structure = {
            "INITIAL": {},
            "FINAL": {}
        }

        def read_attrs(group):
            return {k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in group.attrs.items()}

        pos_group = self.file.get("positioning")
        if pos_group is None:
            return structure

        for stage in ("INITIAL", "FINAL"):
            stage_group = pos_group.get(stage)
            if stage_group is None:
                continue

            for name, obj in stage_group.items():
                if not isinstance(obj, h5py.Group):
                    continue
                structure[stage][name] = read_attrs(obj)

        return structure

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
