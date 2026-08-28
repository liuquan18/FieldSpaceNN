import copy
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import xarray as xr
from omegaconf import DictConfig, ListConfig, OmegaConf

from .datasets_base import BaseDataset
from ..modules.grids.grid_utils import icon_get_parent_index, identity_grid_mapping


class ICONLoader(BaseDataset):
    def __init__(
        self,
        data_dict: Mapping[str, Any],
        sampling_zooms: Mapping[int, Mapping[str, Any]],
        sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = None,
        sampling_times_emb: Optional[Mapping[str, Any]] = None,
        sampling_zooms_target: Mapping[int, Mapping[str, Any]] = None,
        grid_files: Optional[Mapping[int, str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ICON native-grid dataset loader and build per-zoom patch indices.

        Unlike Healpix, ICON refinement levels (``R*B*``) do not follow an
        analytic pixel-count formula, so each zoom level in ``sampling_zooms``
        is expected to have its own source file(s) under
        ``data_dict['source'][zoom]['files']`` from which the number of cells
        is read directly.

        :param data_dict: Dataset configuration including source file paths,
            keyed per zoom level (``data_dict['source'][zoom]['files']``).
        :param sampling_zooms: Sampling configuration keyed by zoom level.
        :param sampling_zooms_collate: Optional collate configuration keyed by zoom level.
        :param sampling_times_emb: Optional shared embedding-only time window with
            ``n_past_ts`` and ``n_future_ts``.
        :param sampling_zooms_target: Optional target sampling configuration keyed
            by zoom level (defaults to ``sampling_zooms``).
        :param grid_files: Optional mapping from zoom level to the path of the
            corresponding ICON *grid* file (containing ``clon``/``clat`` and,
            ideally, ``child_cell_index``/``parent_cell_index``), covering
            every integer zoom level between the coarsest sampled patch zoom
            and the finest sampled zoom. When provided, it is used to derive
            the exact quad-tree parent-child hierarchy for patch sampling
            instead of assuming a Healpix-like contiguous cell ordering.
        :param kwargs: Additional arguments forwarded to the base dataset initializer.
        :return: None.
        """
        self.data_dict: Mapping[str, Any] = data_dict

        if isinstance(sampling_zooms, (DictConfig, ListConfig)):
            sampling_zooms = OmegaConf.to_container(sampling_zooms, resolve=True)
        if isinstance(sampling_zooms_collate, (DictConfig, ListConfig)):
            sampling_zooms_collate = OmegaConf.to_container(sampling_zooms_collate, resolve=True)
        if isinstance(sampling_times_emb, (DictConfig, ListConfig)):
            sampling_times_emb = OmegaConf.to_container(sampling_times_emb, resolve=True)
        if isinstance(sampling_zooms_target, (DictConfig, ListConfig)):
            sampling_zooms_target = OmegaConf.to_container(sampling_zooms_target, resolve=True)
        if isinstance(grid_files, (DictConfig, ListConfig)):
            grid_files = OmegaConf.to_container(grid_files, resolve=True)

        self.sampling_zooms: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms)
        self.sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = copy.deepcopy(
            sampling_zooms_collate
        )
        self.sampling_times_emb: Optional[Mapping[str, Any]] = copy.deepcopy(sampling_times_emb)

        if sampling_zooms_target is None:
            self.sampling_zooms_target: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms)
        else:
            self.sampling_zooms_target: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms_target)

        if grid_files is not None:
            grid_files = {int(z): path for z, path in grid_files.items()}

        # Build patch indices per zoom from each zoom's own ICON native grid
        # file. The number of cells (and thus the zoom/refinement level) is
        # read directly from the data instead of derived analytically.
        self.indices: Dict[int, np.ndarray] = {}
        for zoom, sampling in sampling_zooms.items():
            zoom = int(zoom)
            source_files = data_dict["source"][zoom]["files"]

            if isinstance(source_files, list):
                data_source = source_files
            else:
                data_source = np.loadtxt(source_files, dtype='str', ndmin=1)

            with xr.open_dataset(data_source[0]) as ds:
                cell_dim = "cell" if "cell" in ds.sizes else "ncells"
                npix = int(ds.sizes[cell_dim])

            # ICON global grids follow ncells = 5 * 4**zoom (e.g. R2B04 has
            # zoom = 5, ncells = 20480). Validate that the configured zoom
            # key matches what the file actually provides.
            zoom_from_npix = int(math.log(npix // 5, 4))
            if zoom_from_npix != zoom:
                raise ValueError(
                    f"Configured zoom {zoom} does not match the zoom inferred "
                    f"from the ICON grid file ({zoom_from_npix}, ncells={npix})."
                )

            if sampling['zoom_patch_sample'] == -1:
                self.indices[zoom] = np.arange(npix).reshape(1, -1)
            elif grid_files is not None:
                self.indices[zoom] = _build_patch_indices_from_hierarchy(
                    zoom, sampling['zoom_patch_sample'], npix, grid_files
                )
            else:
                # Fallback: assumes the on-disk cell ordering already follows
                # the recursive quad-tree bisection order (true for some, but
                # not all, ICON grid generation pipelines). Prefer passing
                # `grid_files` so the exact hierarchy from the grid file(s)
                # is used instead of this assumption.
                n_pts_patch = 4 ** (zoom - sampling['zoom_patch_sample'])
                self.indices[zoom] = np.arange(npix).reshape(-1, n_pts_patch)

        super().__init__(mapping_fcn=identity_grid_mapping, **kwargs)

    def get_indices_from_patch_idx(self, zoom: int, patch_idx: int) -> np.ndarray:
        """
        Get the pixel indices corresponding to a patch index at a given zoom.

        :param zoom: Zoom level whose patch grid is being queried.
        :param patch_idx: Patch index within the sampled patch grid.
        :return: Column vector of pixel indices for the selected patch with shape ``(n,)``.
        """
        return self.indices[int(zoom)][patch_idx].reshape(-1)


def _build_patch_indices_from_hierarchy(
    zoom: int,
    zoom_patch_sample: int,
    npix: int,
    grid_files: Mapping[int, str],
) -> np.ndarray:
    """
    Group a zoom level's cells into patches using the real ICON parent-child hierarchy.

    Chains ``icon_get_parent_index`` through every intermediate integer zoom
    level between ``zoom`` and ``zoom_patch_sample`` (all of which must be
    present in ``grid_files``) to find each cell's ancestor at
    ``zoom_patch_sample``, then groups cells sharing the same ancestor into a
    patch.

    :param zoom: Zoom level of the cells being grouped into patches.
    :param zoom_patch_sample: Target (coarser) zoom level defining the patches.
    :param npix: Number of cells at ``zoom``.
    :param grid_files: Mapping from zoom level to ICON grid file path,
        covering every integer zoom from ``zoom_patch_sample`` to ``zoom``.
    :return: Index array of shape ``(n_patches, patch_size)``.
    """
    missing = [z for z in range(zoom_patch_sample, zoom + 1) if z not in grid_files]
    assert not missing, (
        f"`grid_files` must include every zoom level from {zoom_patch_sample} to "
        f"{zoom} to resolve the ICON parent-child hierarchy; missing {missing}."
    )

    ancestor = np.arange(npix)
    datasets = {z: xr.open_dataset(grid_files[z]) for z in range(zoom_patch_sample, zoom + 1)}
    try:
        for z in range(zoom, zoom_patch_sample, -1):
            parent_index = icon_get_parent_index(datasets[z], datasets[z - 1]).numpy()
            ancestor = parent_index[ancestor]
    finally:
        for ds in datasets.values():
            ds.close()

    patch_size = 4 ** (zoom - zoom_patch_sample)
    order = np.argsort(ancestor, kind="stable")
    ancestor_sorted = ancestor[order]

    _, counts = np.unique(ancestor_sorted, return_counts=True)
    assert (counts == patch_size).all(), (
        "The ICON grid hierarchy does not produce uniformly sized patches "
        f"(expected {patch_size} cells per patch); the grid may include "
        "local refinement/nesting that this uniform patch sampler does not support."
    )

    return order.reshape(-1, patch_size)
