"""
MATLAB-style struct / struct-array save & load using HDF5 (+ SWMR-ready).
"""

import numpy as np
import h5py

# ============================================================
# Core data containers
# ============================================================

class MyStruct:
    """
    MATLAB-like struct:
    - arbitrary attributes
    - attribute access (obj.field)
    - dict-like access (obj['field'])
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"MyStruct({items})"


class StructArray:
    """
    MATLAB-like struct array:
    obj[0].field = value
    obj[10].other = value   # auto-expands
    """

    def __init__(self):
        self._items = []

    def __getitem__(self, index):
        while len(self._items) <= index:
            self._items.append(MyStruct())
        return self._items[index]

    def __repr__(self):
        return f"StructArray({self._items})"


# ============================================================
# Public API
# ============================================================
def save_data(filename, obj, mode="w", swmr=True):
    with h5py.File(filename, mode, libver="latest") as f:
        if isinstance(obj, StructArray):
            _write_structarray(f, obj)
        else:
            _write_mystruct(f, obj)

        if swmr and mode in ("w", "r+"):
            f.swmr_mode = True
            f.flush()


def load_data(filename):
    with h5py.File(filename, "r", libver="latest", swmr=True) as f:
        obj = MyStruct()
        _read_mystruct(f, obj)
        return obj

# ============================================================
# Internal: write helpers
# ============================================================

def _write_structarray(h5group, struct_array):
    for i, item in enumerate(struct_array._items):
        grp = h5group.require_group(str(i))
        _write_mystruct(grp, item)


def _write_mystruct(h5group, mystruct):
    for key, value in mystruct.__dict__.items():
        _write_value(h5group, key, value)


def _write_value(h5group, name, value):
    # Nested struct
    if isinstance(value, MyStruct):
        grp = h5group.require_group(name)
        _write_mystruct(grp, value)

    # Nested struct array
    elif isinstance(value, StructArray):
        grp = h5group.require_group(name)
        _write_structarray(grp, value)

    # Scalars & strings → attributes
    elif np.isscalar(value) or isinstance(value, str):
        h5group.attrs[name] = value

    # Everything else → dataset
    else:
        arr = np.asarray(value)
        if name in h5group:
            del h5group[name]
        h5group.create_dataset(
            name,
            data=arr,
            chunks=True
        )


# ============================================================
# Internal: read helpers
# ============================================================

def _read_structarray(h5group):
    sa = StructArray()

    for key in sorted(h5group.keys(), key=int):
        item = MyStruct()
        _read_mystruct(h5group[key], item)
        sa._items.append(item)

    return sa


def _read_mystruct(h5group, mystruct):
    # Attributes → scalars
    for k, v in h5group.attrs.items():
        mystruct.__dict__[k] = v

    # Datasets / subgroups
    for name, obj in h5group.items():
        if isinstance(obj, h5py.Dataset):
            mystruct.__dict__[name] = obj[()]

        elif isinstance(obj, h5py.Group):
            # Numeric group names → StructArray
            if obj.keys() and all(k.isdigit() for k in obj.keys()):
                mystruct.__dict__[name] = _read_structarray(obj)
            else:
                sub = MyStruct()
                _read_mystruct(obj, sub)
                mystruct.__dict__[name] = sub


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    obj = StructArray()
    obj[0].name = "first"
    obj[0].color = "blue"
    obj[0].matrix = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    obj[1].vector = [1.0, 2.0, 3.0]
    obj[1].matrix = np.eye(2)

    save_data("example.h5", obj)
    loaded = load_data("example.h5")

    print(loaded)
    print(loaded[1].matrix)





