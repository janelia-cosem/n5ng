import zarr
from pathlib import Path
from os.path import sep


def is_multiscale(group):
    keys = group.attrs.keys()
    return 'downsamplingFactors' in keys


def get_datasets(container_path):
    zr = zarr.open(container_path)
    return zr.visitvalues(lambda v: [container_path + sep + v[1].path for v in v.arrays()])


def parse(container_path):
    pl = Path(container_path)
    parent = str(pl.parent.parts[-1])
    result = {}
    result[parent] = {}
    inner_result = result[parent]
    inner_result['name'] = parent
    inner_result['container_path'] = container_path
    inner_result['dataset_paths'] = get_datasets(container_path)
    inner_result['thumbnail'] = f'{parent}/thumbnail.png'
    inner_result['readme'] = f'{parent}/README.md'
    return result
