import copy
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import xarray as xr
from omegaconf import DictConfig, ListConfig, OmegaConf

from .datasets_base import BaseDataset
from ..modules.grids.grid_utils import hierarchical_zoom_distance_map


class ICONLoader(BaseDataset):
    def __init__(
        self,
        data_dict: Mapping[str, Any],
        sampling_zooms: Mapping[int, Mapping[str, Any]],
        sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = None,
        sampling_times_emb: Optional[Mapping[str, Any]] = None,
        sampling_zooms_target: Mapping[int, Mapping[str, Any]] = None,
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

        self.sampling_zooms: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms)
        self.sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = copy.deepcopy(
            sampling_zooms_collate
        )
        self.sampling_times_emb: Optional[Mapping[str, Any]] = copy.deepcopy(sampling_times_emb)

        if sampling_zooms_target is None:
            self.sampling_zooms_target: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms)
        else:
            self.sampling_zooms_target: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms_target)

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
            else:
                n_pts_patch = 4 ** (zoom - sampling['zoom_patch_sample'])
                self.indices[zoom] = np.arange(npix).reshape(-1, n_pts_patch)

        super().__init__(mapping_fcn=hierarchical_zoom_distance_map, **kwargs)

    def get_indices_from_patch_idx(self, zoom: int, patch_idx: int) -> np.ndarray:
        """
        Get the pixel indices corresponding to a patch index at a given zoom.

        :param zoom: Zoom level whose patch grid is being queried.
        :param patch_idx: Patch index within the sampled patch grid.
        :return: Column vector of pixel indices for the selected patch with shape ``(n,)``.
        """
        return self.indices[int(zoom)][patch_idx].reshape(-1)
