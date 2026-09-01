import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import healpy as hp
import numpy as np
import torch
import xarray as xr

from scipy.interpolate import griddata
from scipy.spatial import cKDTree

radius_earth = 6371

def get_zoom_from_npix(npix: int):
    """
    Get the zoom level for a Healpix grid from its pixel count.

    :param npix: Total number of pixels in the Healpix grid.
    :return: Zoom level (log2 of nside) or None if the input is invalid.
    """
    try:
        nside = hp.npix2nside(npix)
        zoom_level = int(np.log2(nside))
        return zoom_level
    except ValueError:
        return None

def get_lon_lat_names(grid_type: Optional[str]):
    """
    Resolve longitude/latitude variable names for a grid type.

    :param grid_type: Grid type identifier (cell, vertex, edge, lonlat, longitudelatitude).
    :return: Tuple of (longitude_name, latitude_name).
    """
    if grid_type == 'cell':
        lon, lat = 'clon', 'clat'

    elif grid_type == 'vertex':
        lon, lat = 'vlon', 'vlat'

    elif grid_type == 'edge':
        lon, lat = 'elon', 'elat'

    elif grid_type == 'lonlat':
        lon, lat = 'lon', 'lat'

    elif grid_type == 'longitudelatitude':
        lon, lat = 'longitude', 'latitude'

    else:
        lon, lat = 'vlon', 'vlat'
    return lon, lat


def get_coords_as_tensor(
    ds: xr.Dataset,
    lon: str = 'vlon',
    lat: str = 'vlat',
    grid_type: Optional[str] = None,
    target: str = 'torch'
):
    """
    Load coordinates from a dataset and return a stacked (lon, lat) array.

    :param ds: Input Xarray Dataset to read data from.
    :param lon: Input longitude variable to read (overridden by grid_type).
    :param lat: Input latitude variable to read (overridden by grid_type).
    :param grid_type: Either cell, edge, vertex, lonlat, or longitudelatitude.
    :param target: Output backend ("torch" or "numpy").
    :return: Coordinates of shape ``(n, 2)`` or ``(n_lon * n_lat, 2)``.
    """
    lon, lat = get_lon_lat_names(grid_type)

    lons_values = None
    lats_values = None
    if lon in ds.variables and lat in ds.variables:
        lons_values = ds[lon].values
        lats_values = ds[lat].values
    elif grid_type == 'cell' and 'lon_cell_centre' in ds.variables and 'lat_cell_centre' in ds.variables:
        # ICON grid-generator files (e.g. Earth_IcosS_*.nc) name cell-center
        # coordinates 'lon_cell_centre'/'lat_cell_centre' instead of 'clon'/'clat'.
        lons_values = ds['lon_cell_centre'].values.astype(np.float64)
        lats_values = ds['lat_cell_centre'].values.astype(np.float64)

        # These may be stored in degrees or radians depending on the grid
        # generator; detect degrees either via a 'units' attribute or,
        # failing that, via value range, and normalize to radians.
        lon_units = str(ds['lon_cell_centre'].attrs.get('units', '')).lower()
        lat_units = str(ds['lat_cell_centre'].attrs.get('units', '')).lower()
        is_degree_input = 'degree' in lon_units or 'degree' in lat_units
        if not lon_units and not lat_units:
            max_abs_lon = float(np.nanmax(np.abs(lons_values)))
            max_abs_lat = float(np.nanmax(np.abs(lats_values)))
            is_degree_input = max_abs_lon > (2 * np.pi + 1e-6) or max_abs_lat > (0.5 * np.pi + 1e-6)
        if is_degree_input:
            lons_values = np.deg2rad(lons_values)
            lats_values = np.deg2rad(lats_values)
    else:
        # Fallback to Healpix coordinates when explicit cell coordinates are absent.
        if grid_type == 'cell':
            npix = None
            if 'cell' in ds.sizes:
                npix = int(ds.sizes['cell'])
            elif 'ncells' in ds.sizes:
                npix = int(ds.sizes['ncells'])

            if npix is not None:
                zoom = get_zoom_from_npix(npix)
                if zoom is not None:
                    return healpix_pixel_lonlat_torch(zoom, return_numpy=target == 'numpy')

                # `npix` does not correspond to any valid Healpix pixel count
                # (e.g. ICON's ncells = 5 * 4**zoom). Rather than silently
                # returning None (which surfaces as a confusing crash much
                # later), fail loudly here.
                raise ValueError(
                    f"Could not resolve cell coordinates: no '{lon}'/'{lat}' "
                    f"(or 'lon_cell_centre'/'lat_cell_centre') variables found, "
                    f"and npix={npix} is not a valid Healpix pixel count. "
                    "For ICON native grids, pass the grid file (containing "
                    "cell coordinates) rather than a data file that lacks them."
                )
            return None

        # For regular lon/lat-style grids, synthesize coordinates from dimensions
        # when explicit coordinate variables are missing.
        if grid_type in ('lonlat', 'longitudelatitude'):
            lon_dim = lon if lon in ds.sizes else None
            lat_dim = lat if lat in ds.sizes else None

            if lon_dim is None or lat_dim is None:
                lon_candidates = [d for d in ds.sizes if 'lon' in d.lower()]
                lat_candidates = [d for d in ds.sizes if 'lat' in d.lower()]
                if lon_dim is None and lon_candidates:
                    lon_dim = lon_candidates[0]
                if lat_dim is None and lat_candidates:
                    lat_dim = lat_candidates[0]

            if lon_dim is None or lat_dim is None:
                return None

            n_lon = int(ds.sizes[lon_dim])
            n_lat = int(ds.sizes[lat_dim])
            if n_lon <= 0 or n_lat <= 0:
                return None

            if n_lon == 1:
                lons_values = np.array([0.0], dtype=np.float32)
            else:
                lons_values = np.linspace(0.0, 360.0, n_lon, endpoint=False, dtype=np.float32)

            if n_lat == 1:
                lats_values = np.array([0.0], dtype=np.float32)
            else:
                # Use regular latitude-cell centers.
                lat_step = 180.0 / float(n_lat)
                lats_values = np.linspace(
                    -90.0 + 0.5 * lat_step,
                    90.0 - 0.5 * lat_step,
                    n_lat,
                    dtype=np.float32,
                )
        else:
            return None

    if target == 'torch':
        lons = torch.tensor(lons_values)
        lats = torch.tensor(lats_values)

        if grid_type == 'lonlat' or grid_type == 'longitudelatitude':
            # Expand lon/lat grids into flattened coordinate pairs.
            lons, lats = torch.meshgrid((lons.deg2rad(),lats.deg2rad()), indexing='xy')
            lons = lons.reshape(-1)
            lats = lats.reshape(-1)


        coords = torch.stack((lons, lats), dim=-1).float()
    else:
        lons = np.array(lons_values)
        lats = np.array(lats_values)

        if grid_type == 'lonlat' or grid_type == 'longitudelatitude':
            # Expand lon/lat grids into flattened coordinate pairs.
            lons, lats = np.meshgrid(np.deg2rad(lons),np.deg2rad(lats),indexing='xy')
            lons = lons.reshape(-1)
            lats = lats.reshape(-1) 
        coords = np.stack((lons, lats), axis=-1).astype(np.float32)

    return coords


def get_grid_type_from_var(ds: xr.Dataset, variable: str):
    """
    Infer the grid type from a variable's spatial dimension.

    :param ds: Input Xarray Dataset.
    :param variable: Variable name to inspect.
    :return: Grid type string or None if it cannot be inferred.
    """
    dims = ds[variable].dims
    spatial_dim = dims[-1]

    if 'longitude' in spatial_dim or 'latitude' in spatial_dim:
        return 'longitudelatitude'

    if 'lon' in spatial_dim or 'lat' in spatial_dim:
        return 'lonlat'
    
    elif 'cell' in spatial_dim:
        return 'cell'

    elif 'vertex' in spatial_dim:
        return 'vertex'
    
    elif 'edge' in spatial_dim:
        return 'edge'
    


def get_coord_dict_from_var(ds: xr.Dataset, variable: str):
    """
    Get longitude/latitude coordinate variable names for a dataset variable.

    :param ds: Input Xarray Dataset to read data from.
    :param variable: Input variable to read.
    :return: Dictionary with keys "lon" and "lat".
    """
    dims = ds[variable].dims
    spatial_dim = dims[-1]
    
    if not 'lon' in spatial_dim and not 'lat' in spatial_dim:
        coords = list(ds[spatial_dim].coords.keys())
    else:
        coords = dims

    if len(coords)==0:
        coords = list(ds[variable].coords.keys())
        
    lon_c = [var for var in coords if 'lon' in var]
    lat_c = [var for var in coords if 'lat' in var]
    
    if len(lon_c)==0:
        len_points = len(ds.coords[spatial_dim].values)
        dims_match = [dim for dim in ds.coords if len(ds[dim].values)==len_points]

        lon_c = [var for var in dims_match if 'lon' in var]
        lat_c = [var for var in dims_match if 'lat' in var]

    assert len(lon_c)==1, "no longitude variable was found"
    assert len(lat_c)==1, "no latitude variable was found"

    return {'lon':lon_c[0],'lat':lat_c[0]}


def scale_coordinates(coords: torch.Tensor, scale_factor: float):
    """
    Scale coordinates around their centroid by a factor.

    :param coords: Coordinate tensor of shape ``(d, n)``.
    :param scale_factor: Factor by which to scale coordinates around their mean.
    :return: Scaled coordinates of shape ``(d, n)``.
    """
    m = coords.mean(dim=1, keepdim=True)
    return (coords - m) * scale_factor + m


def distance_on_sphere(
    lon1: torch.Tensor,
    lat1: torch.Tensor,
    lon2: torch.Tensor,
    lat2: torch.Tensor
):
    """
    Compute great-circle distances between target and source coordinates.

    :param lon1: Target longitude tensor (radians) of shape ``(...,)``.
    :param lat1: Target latitude tensor (radians) of shape ``(...,)``.
    :param lon2: Source longitude tensor (radians) of shape ``(...,)``.
    :param lat2: Source latitude tensor (radians) of shape ``(...,)``.
    :return: Distance tensor (radians) with broadcasted shape ``(...,)``.
    """
    d_lat = torch.abs(lat1-lat2)
    d_lon = torch.abs(lon1-lon2)
    asin = torch.sin(d_lat/2)**2

    d_rad = 2*torch.arcsin(torch.sqrt(asin + (1-asin-torch.sin((lat1+lat2)/2)**2)*torch.sin(d_lon/2)**2))
    return d_rad


def rotate_coord_system(
    lons: torch.Tensor,
    lats: torch.Tensor,
    rotation_lon: torch.Tensor,
    rotation_lat: torch.Tensor
):
    """
    Rotate spherical coordinates by a given rotation origin.

    :param lons: Input longitudes (radians) of shape ``(n,)``.
    :param lats: Input latitudes (radians) of shape ``(n,)``.
    :param rotation_lon: Rotation longitudes (radians) of shape ``(b,)``.
    :param rotation_lat: Rotation latitudes (radians) of shape ``(b,)``.
    :return: Tuple of rotated (lon, lat) tensors of shape ``(b, n)``.
    """

    theta = rotation_lat
    phi = rotation_lon

    theta = theta.view(-1,1)
    phi = phi.view(-1,1)

    x = (torch.cos(lons) * torch.cos(lats)).view(1,-1)
    y = (torch.sin(lons) * torch.cos(lats)).view(1,-1)
    z = (torch.sin(lats)).view(1,-1)

    rotated_x =  torch.cos(theta)*torch.cos(phi) * x + torch.cos(theta)*torch.sin(phi)*y + torch.sin(theta)*z
    rotated_y = -torch.sin(phi)*x + torch.cos(phi)*y
    rotated_z = -torch.sin(theta)*torch.cos(phi)*x - torch.sin(theta)*torch.sin(phi)*y + torch.cos(theta)*z

    rot_lon = torch.atan2(rotated_y, rotated_x)
    rot_lat = torch.arcsin(rotated_z)

    return rot_lon, rot_lat


def get_distance_angle(
    lon1: torch.Tensor,
    lat1: torch.Tensor,
    lon2: torch.Tensor,
    lat2: torch.Tensor,
    base: str = 'polar',
    periodic_fov: Optional[Tuple[float, float]] = None,
    rotate_coords: bool = True
):
    """
    Compute relative distances/angles between two sets of coordinates.

    :param lon1: Target longitude tensor (radians) of shape ``(b, m)``.
    :param lat1: Target latitude tensor (radians) of shape ``(b, m)``.
    :param lon2: Source longitude tensor (radians) of shape ``(b, n)``.
    :param lat2: Source latitude tensor (radians) of shape ``(b, n)``.
    :param base: Output basis ("polar" for distance/angle or "cartesian" for lon/lat deltas).
    :param periodic_fov: Optional (min, max) longitude bounds for periodic wrapping.
    :param rotate_coords: Whether to rotate coordinates to the target frame.
    :return: Tuple of tensors with shape ``(b, m, n)`` for (distance, angle) or (d_lon, d_lat).
    """
    if rotate_coords:
        # Rotate sources into the target reference frame to stabilize distances.
        theta = lat1
        phi = lon1

        x = (torch.cos(lon2) * torch.cos(lat2))
        y = (torch.sin(lon2) * torch.cos(lat2))
        z = (torch.sin(lat2))

        rotated_x =  torch.cos(theta)*torch.cos(phi) * x + torch.cos(theta)*torch.sin(phi)*y + torch.sin(theta)*z
        rotated_y = -torch.sin(phi)*x + torch.cos(phi)*y
        rotated_z = -torch.sin(theta)*torch.cos(phi)*x - torch.sin(theta)*torch.sin(phi)*y + torch.cos(theta)*z

        lon2 = torch.atan2(rotated_y, rotated_x)
        lat2 = torch.arcsin(rotated_z)

        lat1=lon1=0

        d_lons = (lon2).abs()
        d_lats = (lat2).abs()

        sgn_lat = torch.sign(lat2)
        sgn_lon = torch.sign(lon2)

    else:
        # Use direct angular distances without rotation.
        d_lons =  2*torch.arcsin(torch.cos(lat1)*torch.sin(torch.abs(lon2-lon1)/2))
        d_lats = (lat2-lat1).abs() 
        sgn_lat = torch.sign(lat1-lat2)
        sgn_lon = torch.sign(lon1-lon2)
     
    # Correct sign ambiguities across the pi boundary.
    sgn_lat[(d_lats).abs()/torch.pi>1] = sgn_lat[(d_lats).abs()/torch.pi>1]*-1
    d_lats = d_lats*sgn_lat

    sgn_lon[(d_lons).abs()/torch.pi>1] = sgn_lon[(d_lons).abs()/torch.pi>1]*-1
    d_lons = d_lons*sgn_lon

    if periodic_fov is not None:
        # Wrap longitude distances for periodic patches.
        rng_lon = (periodic_fov[1] - periodic_fov[0])
        d_lons[d_lons > rng_lon] = d_lons[d_lons > rng_lon] - rng_lon
        d_lons[d_lons < -rng_lon] = d_lons[d_lons < -rng_lon] + rng_lon

    if base == "polar":
        distance = torch.sqrt(d_lats**2 + d_lons**2)
        phi = torch.atan2(d_lats, d_lons)

        return distance.float(), phi.float()

    else:
        return d_lons.float(), d_lats.float()

def global_indices_to_paths_dict(
    global_indices: torch.Tensor,
    zoom: Optional[int] = None,
    sizes: Optional[Union[torch.Tensor, Sequence[int]]] = None
):
    """
    Convert global indices into per-zoom paths.

    :param global_indices: Global index tensor of shape ``(n,)``.
    :param zoom: Optional max zoom level (used when sizes is None).
    :param sizes: Optional list or tensor with number of children per zoom level.
    :return: Dict mapping zoom level to index tensor of shape ``(n,)``.
    """
    if sizes is not None:
        num_levels = len(sizes)
    elif zoom is not None:
        num_levels = zoom + 1
        sizes = torch.full((num_levels,), 4, dtype=torch.long, device=global_indices.device)
    else:
        raise ValueError("At least one of 'sizes' or 'zoom' must be provided.")

    # Compute strides as the product of later sizes.
    rev_sizes = sizes.flip(0)
    rev_cumprod = torch.cumprod(rev_sizes, 0)
    strides = torch.ones_like(sizes)
    strides[:-1] = rev_cumprod.flip(0)[1:]

    # Compute indices for all levels at once.
    indices_matrix = (global_indices.unsqueeze(1) // strides) % sizes

    # Return as dict {zoom_level: 1D tensor}.
    out = {z: indices_matrix[:, z] for z in range(num_levels)}
    return out


def get_zoom_x(x: torch.Tensor, zoom_patch_sample: Optional[int] = None, **kwargs: Any):
    """
    Infer zoom level from an input tensor's spatial dimension.

    :param x: Input tensor with spatial dimension at index 2, shape ``(..., s, ...)``.
    :param zoom_patch_sample: Optional patch zoom adjustment.
    :param kwargs: Additional keyword arguments (unused).
    :return: Inferred zoom level.
    """
    s = x.shape[2]
    zoom_x = int(math.log(s) / math.log(4) + zoom_patch_sample) if zoom_patch_sample else int(math.log(s/5) / math.log(4))
    return zoom_x


def healpix_get_adjacent_cell_indices(zoom: int):
    """
    Get neighbor indices for a Healpix grid.

    :param zoom: Healpix zoom level.
    :return: Tuple of (adjacency, duplicates_mask), both shape ``(npix, 9)``.
    """

    nside = 2**zoom
    npix = hp.nside2npix(nside)

    adjc = torch.tensor(hp.get_all_neighbours(nside, np.arange(npix),nest=True)).transpose(0,1)

    adjc = torch.concat((torch.arange(npix).view(-1,1),adjc),dim=-1)
    duplicates = adjc == -1

    c,n = torch.where(duplicates)
    adjc[c,n] = adjc[c,0]

    return adjc, duplicates


def healpix_pixel_lonlat_torch(zoom: int, return_numpy: bool = False):
    """
    Get Healpix pixel coordinates (lon, lat) for a zoom level.

    :param zoom: Healpix zoom level.
    :param return_numpy: Whether to return a NumPy array.
    :return: Coordinate array of shape ``(npix, 2)``.
    """
    nside = 2**zoom

    npix = hp.nside2npix(nside)  # Total number of pixels

    # Get pixel indices as a PyTorch tensor
    pixel_indices = torch.arange(npix, dtype=torch.long)

    # Get theta (colatitude) and phi (longitude) for each pixel using healpy
    theta, phi = hp.pix2ang(nside, pixel_indices.numpy(), nest=True)

    # Convert theta and phi to PyTorch tensors
    theta_tensor = torch.tensor(theta, dtype=torch.float32) - 0.5 * torch.pi
    phi_tensor = torch.tensor(phi, dtype=torch.float32) - torch.pi

    coords = torch.stack([phi_tensor, theta_tensor], dim=-1).float()

    if return_numpy:
        return coords.numpy()
    else:
        return coords


def estimate_healpix_cell_radius_rad(n_cells: int):
    """
    Estimate the average angular radius for Healpix cells.

    :param n_cells: Number of Healpix cells.
    :return: Cell radius in radians.
    """
    return math.sqrt(4*math.pi / n_cells)


def regular_to_healpix_nearest_mapping(
    longitudes: Union[np.ndarray, torch.Tensor, Sequence[float]],
    latitudes: Union[np.ndarray, torch.Tensor, Sequence[float]],
    healpix_level: int,
):
    """
    Compute nearest-neighbor mappings between regular lon/lat points and Healpix cells.

    :param longitudes: Regular-grid longitudes in degrees or radians. Can be a lon axis
        ``(n_lon,)`` or flattened values ``(n_points,)``.
    :param latitudes: Regular-grid latitudes in degrees or radians. Can be a lat axis
        ``(n_lat,)`` or flattened values ``(n_points,)``.
    :param healpix_level: Healpix level where ``nside = 2**healpix_level``.
    :return: Mapping dictionary with keys:
        ``regular_to_healpix_indices`` (``(n_points,)``),
        ``indices`` (``(n_healpix, 1)`` nearest regular index per Healpix cell),
        ``distances`` (``(n_healpix, 1)`` in radians), and
        ``resolution`` (scalar radius proxy in radians).
    """
    lon = np.asarray(longitudes, dtype=np.float64).squeeze()
    lat = np.asarray(latitudes, dtype=np.float64).squeeze()

    if lon.ndim == 1 and lat.ndim == 1 and lon.size != lat.size:
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
        lon = lon_grid.reshape(-1)
        lat = lat_grid.reshape(-1)
    else:
        if lon.shape != lat.shape:
            raise ValueError(
                f"Expected matching lon/lat shapes, got {lon.shape} and {lat.shape}."
            )
        lon = lon.reshape(-1)
        lat = lat.reshape(-1)

    if lon.size == 0:
        raise ValueError("Regular grid coordinates are empty.")

    # Convert degrees to radians when inputs are outside radian ranges.
    max_abs_lon = float(np.nanmax(np.abs(lon)))
    max_abs_lat = float(np.nanmax(np.abs(lat)))
    is_degree_input = max_abs_lon > (2 * np.pi + 1e-6) or max_abs_lat > (0.5 * np.pi + 1e-6)
    if is_degree_input:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)

    lon = ((lon + np.pi) % (2 * np.pi)) - np.pi
    lat = np.clip(lat, -0.5 * np.pi, 0.5 * np.pi)

    reg_xyz = np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
        axis=-1,
    )

    hp_lonlat = healpix_pixel_lonlat_torch(healpix_level, return_numpy=True).astype(np.float64)
    hp_lon = hp_lonlat[:, 0]
    hp_lat = hp_lonlat[:, 1]
    hp_xyz = np.stack(
        (
            np.cos(hp_lat) * np.cos(hp_lon),
            np.cos(hp_lat) * np.sin(hp_lon),
            np.sin(hp_lat),
        ),
        axis=-1,
    )

    # Regular -> Healpix nearest-neighbor assignment.
    hp_tree = cKDTree(hp_xyz)
    _, regular_to_healpix = hp_tree.query(reg_xyz, k=1)

    # Healpix -> Regular nearest-neighbor assignment used by the dataloader mapping.
    reg_tree = cKDTree(reg_xyz)
    _, healpix_to_regular = reg_tree.query(hp_xyz, k=1)

    matched_reg_xyz = reg_xyz[healpix_to_regular]
    cos_angle = np.einsum("ij,ij->i", hp_xyz, matched_reg_xyz)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    distances = np.arccos(cos_angle)
    distances = np.clip(distances, 1e-8, None)

    resolution = estimate_healpix_cell_radius_rad(hp_xyz.shape[0]) / 2.0

    return {
        "regular_to_healpix_indices": torch.from_numpy(regular_to_healpix.astype(np.int64)),
        "indices": torch.from_numpy(healpix_to_regular.astype(np.int64)).view(-1, 1),
        "distances": torch.from_numpy(distances.astype(np.float32)).view(-1, 1),
        "resolution": torch.tensor(resolution, dtype=torch.float32),
    }

def identity_grid_mapping(input_coords: torch.Tensor, max_zoom: int):
    """
    Build a trivial identity mapping for native (non-regridded) grid loaders.

    Some grid types (e.g. ICON's native triangular mesh) are consumed
    directly on their own cells, with no nearest-neighbor regridding step
    involved. ``hierarchical_zoom_distance_map`` is unsuitable here since it
    always measures distances to analytic Healpix pixel centers, which would
    silently snap native-grid cells onto the wrong (Healpix) geometry. This
    function instead returns each cell mapped onto itself, matching the
    ``mapping_fcn`` interface expected by :class:`BaseDataset`.

    :param input_coords: Coordinate tensor of shape ``(n, 2)`` (unused beyond
        determining ``n``, kept for interface parity with ``mapping_fcn``).
    :param max_zoom: Zoom level to key the returned mapping dict with.
    :return: Dict ``{max_zoom: {"indices": (n, 1), "distances": (n, 1), "resolution": ()}}``.
    """
    n = input_coords.shape[0]
    indices = torch.arange(n, dtype=torch.long).view(-1, 1)
    distances = torch.zeros((n, 1), dtype=torch.float32)

    return {
        max_zoom: {
            "indices": indices,
            "distances": distances,
            "resolution": torch.tensor(1.0),
        }
    }

def hierarchical_zoom_distance_map(input_coords: torch.Tensor, max_zoom: int):
    """
    Iteratively computes distances between HEALPix cell centers and input_coords across zoom levels.

    :param input_coords: Coordinate tensor of shape ``(1, n, 2)`` in radians.
    :param max_zoom: Maximum zoom level to evaluate.
    :return: Dict mapping zoom levels to tensors with shapes
        ``indices: (n_cells, k)``, ``distances: (n_cells, k)``, ``resolution: ()``.
    """
    current_input = input_coords  # (1, N, 2)
    results = {}
    global_indices = torch.arange(input_coords.shape[0]).view(1,-1)

    for zoom in range(1, max_zoom + 1):
        c = healpix_pixel_lonlat_torch(zoom)  # (num_cells, 2)
        
        n_cells = c.shape[0]
        r = estimate_healpix_cell_radius_rad(n_cells)

        if zoom == 1:
            # First level: (1, n_cells, 1, 2)
            c = c.view(1, -1, 1, 2)
        else:
            # Next levels: batch size = current_input.shape[0]
            c = c.view(current_input.shape[0], -1, 1, 2)

        # Compute distance and angle for all pairs.
        d, _ = get_distance_angle(
            c[..., 0], c[..., 1],                   # shape: (B, M, 1)
            current_input[..., 0].unsqueeze(dim=-2), current_input[..., 1].unsqueeze(dim=-2),  # shape: (B, 1, N)
            base='polar',
            rotate_coords=False
        )  # Output: (B, M, N)

        # Keep neighbors within a local search radius.
        in_search_radius = d <= r * 2
        n_radius = in_search_radius.sum(dim=-1)  # (B, M)
        n_keep = n_radius.max().item()   # scalar

        # Get top-k distances and indices
        d_sorted, idx_sorted = torch.topk(d, k=n_keep, dim=-1, largest=False)  # (B, M, k)

        global_indices = torch.gather(global_indices, dim=1,index=idx_sorted.view(idx_sorted.shape[0],-1))
        global_indices = global_indices.view(idx_sorted.shape[0]*idx_sorted.shape[1], n_keep)

        current_input = input_coords[global_indices]

        c = c.view(-1,2)
        dim_out = c.shape[0]

        # Save current zoom level.
        results[zoom] = {
            "indices": global_indices.view(dim_out,-1),
            "distances": d_sorted.view(dim_out,-1),
            "resolution": r/2
        }
        
    return results


def get_mapping_weights(
    mapping: Dict[str, torch.Tensor],
    drop_mask: Optional[torch.Tensor] = None,
    mode: str = 'binary'
):
    """
    Compute mapping weights for neighborhood aggregation.

    :param mapping: Mapping dict with "distances" of shape ``(m, k)`` and "indices" of shape ``(m, k)``.
    :param drop_mask: Optional drop mask broadcastable to ``mapping["indices"]``.
    :param mode: Weighting mode ("1/r", "normal", or "binary").
    :return: Weight tensor of shape ``(m, k)`` aligned to mapping["distances"].
    """

    weights = mapping['distances'] 

    if drop_mask is not None:
        # Penalize dropped indices to push them out of selection.
        drop_mask_zoom = drop_mask[mapping['indices']]
        weights = weights + (drop_mask_zoom * 1e5)

    if mode == '1/r':
        weights = mapping["resolution"]/weights 
        weights[weights> 1.] = 1.
    
    elif mode == 'normal':
        pass

    elif mode == 'binary':
        weights = mapping["resolution"]/weights 

        weights[weights> 1.] = 1.
        weights[weights< 1.] = 0.

        weights = weights.bool()

    return weights

def to_zoom(
    x: torch.Tensor,
    in_zoom: int,
    out_zoom: int,
    mask: Optional[torch.Tensor] = None,
    binarize_mask: bool = False
):
    """
    Rescale a tensor between zoom levels by averaging or repeating.

    :param x: Input tensor of shape ``(p, q, n, d, f)``.
    :param in_zoom: Input zoom level.
    :param out_zoom: Output zoom level.
    :param mask: Optional mask tensor of shape ``(p, q, n, d, m)``.
    :param binarize_mask: Whether to binarize the mask after downsampling.
    :return: Tuple of (x_zoom, mask_zoom) with spatial dimension rescaled by the zoom factor.
    """
    if in_zoom == out_zoom:
        mask = mask.bool() if (binarize_mask and mask is not None) else mask
        return x, mask

    scale_factor = 4 ** abs(in_zoom - out_zoom)
    
    prefix = x.shape[:-3]
    dc = x.shape[-2:]

    if in_zoom > out_zoom:
        # Downsample by averaging.
        x = x.view(*prefix, -1, scale_factor, *dc)
        if mask is not None:
            mask = mask.reshape(*prefix, -1, scale_factor, *mask.shape[-2:])
            weight = mask.sum(dim=-3, keepdim=True)
            x_zoom = (x * mask).sum(dim=-3, keepdim=True) / weight.clamp(min=1e-6)
            x_zoom[weight == 0] = 0

            mask_zoom = (weight / (1.*scale_factor))
            if binarize_mask:
                mask_zoom[mask_zoom > 0] = 1
                mask_zoom = mask_zoom.bool()

            x_zoom = x_zoom.view(*vt, -1, *dc)
            mask_zoom = mask_zoom.view(*vt, -1, *mask_zoom.shape[-2:])
            return x_zoom, mask_zoom
        else:
            x_zoom = x.mean(dim=-3)
            return x_zoom, None

    else:
        # Upsample by repeating.
        x_zoom = x.unsqueeze(-3).repeat_interleave(scale_factor, dim=-3)
        if mask is not None:
            mask_zoom = mask.unsqueeze(-3).repeat_interleave(scale_factor, dim=-3)
            mask_zoom = mask_zoom * 1. if not binarize_mask else mask.bool()
            return x_zoom, mask_zoom
        else:
            return x_zoom, None

def insert_matching_time_patch(
    x_h: torch.Tensor,
    x_s: torch.Tensor,
    zoom_h: int,
    zoom_target: int,
    sample_configs: Dict,
    base: int = 12
):
    """
    Insert a lower-zoom patch into a higher-zoom tensor at aligned timesteps.

    :param x_h: High-zoom tensor of shape ``(b, v, t, n, d, f)``.
    :param x_s: Patch tensor to insert with shape ``(b, v, t_patch, n_patch, d, f)``.
    :param zoom_h: Zoom level of x_h.
    :param zoom_target: Target zoom level for the patch.
    :param sample_configs: Sampling configuration per zoom.
    :param base: Base patch count multiplier.
    :return: Tensor with patch inserted, same shape as x_h.
    """
    if zoom_h == zoom_target:
        return x_s

    batched = True
    if x_h.dim() < 6:
        x_h = x_h.unsqueeze(0)
        x_s = x_s.unsqueeze(0)
        batched = False

    # Validate temporal coverage and patch sampling consistency.
    assert sample_configs[zoom_h]['n_past_ts'] >= sample_configs[zoom_target]['n_past_ts'], \
        f'zoom {zoom_h} has a smaller number of past timesteps than {zoom_target}'
    assert sample_configs[zoom_h]['n_future_ts'] >= sample_configs[zoom_target]['n_future_ts'], \
        f'zoom {zoom_h} has a smaller number of future timesteps than {zoom_target}'
    assert sample_configs[zoom_h]['zoom_patch_sample'] <= sample_configs[zoom_target]['zoom_patch_sample'], \
        f'zoom {zoom_h} has a higher zoom_patch_sample than {zoom_target}'

    ts_start = sample_configs[zoom_h]['n_past_ts'] - sample_configs[zoom_target]['n_past_ts']
    ts_end = sample_configs[zoom_h]['n_future_ts'] - sample_configs[zoom_target]['n_future_ts']

    b, nv, nt, n = x_h.shape[:4]
    c = x_h.shape[4:]
    t_range = slice(ts_start, nt - ts_end)

    if sample_configs[zoom_h]['zoom_patch_sample'] == -1 and sample_configs[zoom_target]['zoom_patch_sample'] == -1:
        # Global case: patch is the whole field at the aligned time range.
        x_h_ = x_h.clone()
        x_h_[:, :, t_range] = x_s
        return x_h_ if batched else x_h_[0]
    
    else:
        base_exp = 0

        if sample_configs[zoom_h]['zoom_patch_sample'] == -1:
            zoom_patch_diff = sample_configs[zoom_target]['zoom_patch_sample']
            base_exp = 1
        else:
            zoom_patch_diff = sample_configs[zoom_target]['zoom_patch_sample'] - sample_configs[zoom_h]['zoom_patch_sample']

        n_patches = base**base_exp * 4**zoom_patch_diff
        patch_index = sample_configs[zoom_target]['patch_index'] % n_patches

        # Reshape into patches and insert the patch by index.
        x_h_ = x_h.view(b, nv, nt, n_patches, -1, *c).clone()

        if isinstance(patch_index, int) or (isinstance(patch_index, torch.Tensor) and patch_index.numel() == 1):
            # Fill in the patch directly at index
            x_h_[:, :, t_range, patch_index] = x_s.unsqueeze(dim=3)
        else:
            # Broadcast and scatter each batch
            x_s = x_s.unsqueeze(dim=3)
            view_shape = (patch_index.shape[0], *([1] * (x_s.dim()-1)))
            expand_shape = list(x_s.shape)
            expand_shape[3] = 1
            patch_index_exp = patch_index.view(view_shape).expand(expand_shape)

            x_h_[:, :, t_range] = x_h_[:, :, t_range].scatter(3, patch_index_exp, x_s)


        x_h_ = x_h_.view(b, nv, nt, n, *c)

        return x_h_ if batched else x_h_[0]

def get_matching_time_patch(
    x_h: torch.Tensor,
    zoom_h: int,
    zoom_target: int,
    sample_configs: Dict,
    patch_index_zooms: Optional[Dict[int, torch.Tensor]] = None,
    base: int = 12
):
    """
    Extract a time-aligned patch from a high-zoom tensor.

    :param x_h: High-zoom tensor of shape ``(b, v, t, n, d, f)``.
    :param zoom_h: Zoom level of x_h.
    :param zoom_target: Target zoom to extract.
    :param sample_configs: Sampling configuration per zoom.
    :param patch_index_zooms: Optional per-zoom patch indices.
    :param base: Base patch count multiplier.
    :return: Extracted tensor patch aligned in time, shape ``(b, v, t_patch, n_patch, d, f)``.
    """

    if zoom_h==zoom_target:
        return x_h
    
    batched = True

    if x_h.dim()<5:
        x_h = x_h.unsqueeze(dim=0)
        batched = False
    

    if sample_configs[zoom_h]['n_past_ts'] < sample_configs[zoom_target]['n_past_ts']:
        assert f'zoom {zoom_h} has a smaller number of past timesteps than {zoom_target}'

    if sample_configs[zoom_h]['n_future_ts'] < sample_configs[zoom_target]['n_future_ts']:
        assert f'zoom {zoom_h} has a smaller number of future timesteps than {zoom_target}'

    if sample_configs[zoom_h]['zoom_patch_sample'] > sample_configs[zoom_target]['zoom_patch_sample']:
        assert f'zoom {zoom_h} has a higher zoom_patch_sample than {zoom_target}'

    ts_start = sample_configs[zoom_h]['n_past_ts'] - sample_configs[zoom_target]['n_past_ts']
    ts_end = sample_configs[zoom_h]['n_future_ts'] - sample_configs[zoom_target]['n_future_ts']
    
    b,nv,nt,n = x_h.shape[:4]
    c = x_h.shape[-2:]
    
    if sample_configs[zoom_h]['zoom_patch_sample'] == -1 and sample_configs[zoom_target]['zoom_patch_sample'] == -1:
        # Global case: just slice the aligned time range.

        x_h = x_h[:, :, ts_start:(nt-ts_end)]
        return x_h if batched else x_h[0]
    
    else:
        base_exp = 0 
        if sample_configs[zoom_h]['zoom_patch_sample'] == -1:
            zoom_patch_diff = sample_configs[zoom_target]['zoom_patch_sample']
            base_exp = 1

        else:
            zoom_patch_diff = sample_configs[zoom_target]['zoom_patch_sample'] - sample_configs[zoom_h]['zoom_patch_sample']

        n_patches = base**base_exp * 4**zoom_patch_diff
        patch_index = (patch_index_zooms[zoom_target] if patch_index_zooms is not None else sample_configs[zoom_target]['patch_index']) % n_patches

        

        # Reshape into patches and gather the target patch.
        x_h = x_h.view(b,nv,nt, n_patches, -1 ,*c)

        if isinstance(patch_index, int) or patch_index.numel()==1:
            x_h = x_h[:, :, ts_start:(nt-ts_end), patch_index]
        else:
            x_h = x_h[:, :, ts_start:(nt-ts_end)]
            view_shape = (patch_index.shape[0],*(1,)*(x_h.dim()-1))
            expand_shape = list(x_h.shape)
            expand_shape[3] = 1
            x_h = torch.gather(x_h, dim=3, index=patch_index.view(view_shape).expand(expand_shape))

        x_h = x_h.view(*x_h.shape[:3],-1,*c)

        if batched:
            return x_h
        else:
            return x_h[0]

def healpix_grid_to_mgrid(zoom_max: int = 10, nh: int = 1):
    """
    Build a list of Healpix grid levels with coordinates and adjacency.

    :param zoom_max: Maximum zoom level to include.
    :param nh: Neighborhood depth (unused).
    :return: List of grid dicts with "coords", "adjc", "adjc_mask", and "zoom".
    """
    zooms = []
    grids = []

    for zoom in range(zoom_max + 1):
        zooms.append(zoom)

        adjc, adjc_mask = healpix_get_adjacent_cell_indices(zoom)

        grid_lvl = {
            "coords": healpix_pixel_lonlat_torch(zoom),
            "adjc": adjc,
            "adjc_mask": adjc_mask,
            "zoom": zoom
        }
        grids.append(grid_lvl)

    return grids


def icon_get_adjacent_cell_indices(ds: xr.Dataset):
    """
    Get neighbor indices for an ICON native (triangular) grid.

    Uses the ``neighbor_cell_index`` variable that ships with every ICON grid
    file. It is 1-based and has shape ``(3, ncells)`` (up to 3 direct
    neighbors per triangular cell); missing neighbors (e.g. at a limited-area
    boundary) are encoded as 0 or a negative value.

    :param ds: Xarray Dataset of the ICON grid file (containing
        ``neighbor_cell_index``).
    :return: Tuple of (adjacency, duplicates_mask), both shape ``(ncells, 4)``,
        where column 0 is the cell itself followed by up to 3 neighbors.
    """
    assert "neighbor_cell_index" in ds.variables, (
        "`neighbor_cell_index` was not found in the ICON grid file; "
        "it is required to build the cell adjacency."
    )

    nb = ds["neighbor_cell_index"].values
    if nb.shape[0] == 3 and nb.shape[-1] != 3:
        nb = nb.T

    npix = nb.shape[0]
    # ICON grid files use 1-based, Fortran-style indexing; missing neighbors
    # are encoded as 0 (or occasionally -1 for LAM boundary cells).
    nb = nb.astype(np.int64) - 1

    adjc = torch.tensor(np.concatenate((np.arange(npix).reshape(-1, 1), nb), axis=-1))
    duplicates = (adjc < 0) | (adjc >= npix)

    c, n = torch.where(duplicates)
    adjc[c, n] = adjc[c, 0]

    return adjc, duplicates


def icon_get_parent_index(
    ds_fine: xr.Dataset,
    ds_coarse: Optional[xr.Dataset] = None,
    fine_coords: Optional[torch.Tensor] = None,
    coarse_coords: Optional[torch.Tensor] = None,
):
    """
    Get the child-to-parent mapping between two consecutive ICON refinement levels.

    ICON's bisection-refined ("R*B*") grids retain, in most standard grid
    files, an explicit quad-tree link between a coarser mesh and its
    once-bisected finer mesh. Three strategies are tried, in order of
    preference:

    1. ``child_cell_index`` on the coarse grid file: shape ``(4, n_coarse)``,
       1-based, listing the (up to) four fine-grid children of every coarse
       cell. This is inverted into a per-fine-cell parent index.
    2. ``parent_cell_index`` directly on the fine grid file: shape
       ``(n_fine,)``, 1-based. Note this variable is also used by ICON for
       limited-area (nested) domain-to-parent-domain bookkeeping, which is a
       different concept from same-mesh quad-tree refinement; only rely on
       it here if ``child_cell_index`` is unavailable.
    3. Geometric nearest-neighbor fallback: match every fine-cell center
       (``clon``/``clat``) to its closest coarse-cell center. This does not
       require any refinement metadata and works for any pair of nested
       "R*B*" grids, at the cost of being approximate near cell boundaries.

    :param ds_fine: Xarray Dataset of the finer-resolution ICON grid file.
    :param ds_coarse: Optional Xarray Dataset of the coarser-resolution ICON
        grid file (required for the ``child_cell_index`` strategy and the
        geometric fallback).
    :param fine_coords: Optional precomputed fine-grid coordinates of shape
        ``(n_fine, 2)`` (avoids reloading them from ``ds_fine``).
    :param coarse_coords: Optional precomputed coarse-grid coordinates of
        shape ``(n_coarse, 2)`` (avoids reloading them from ``ds_coarse``).
    :return: LongTensor ``parent_index`` of shape ``(n_fine,)`` mapping every
        fine cell to its parent cell index in the coarse grid.
    """
    if ds_coarse is not None and "child_cell_index" in ds_coarse.variables:
        child = ds_coarse["child_cell_index"].values
        if child.shape[0] != 4 and child.shape[-1] == 4:
            child = child.T

        n_fine = int(child.max())
        parent_index = torch.full((n_fine,), -1, dtype=torch.long)
        parent_ids = torch.arange(child.shape[1], dtype=torch.long)
        for k in range(child.shape[0]):
            child_ids = torch.as_tensor(child[k].astype(np.int64) - 1)
            valid = child_ids >= 0
            parent_index[child_ids[valid]] = parent_ids[valid]

        if (parent_index >= 0).all():
            return parent_index

    if "parent_cell_index" in ds_fine.variables:
        parent_index = torch.as_tensor(
            ds_fine["parent_cell_index"].values.astype(np.int64) - 1
        ).view(-1)
        if (parent_index >= 0).all():
            return parent_index

    # Fall back to a geometric nearest-neighbor match between cell centers.
    assert ds_coarse is not None, (
        "A geometric parent-child match requires the coarse grid dataset "
        "when neither `child_cell_index` nor `parent_cell_index` is available."
    )
    if fine_coords is None:
        fine_coords = get_coords_as_tensor(ds_fine, grid_type="cell", target="numpy")
    else:
        fine_coords = np.asarray(fine_coords)
    if coarse_coords is None:
        coarse_coords = get_coords_as_tensor(ds_coarse, grid_type="cell", target="numpy")
    else:
        coarse_coords = np.asarray(coarse_coords)

    tree = cKDTree(coarse_coords)
    _, parent_index = tree.query(fine_coords, k=1)

    return torch.as_tensor(parent_index, dtype=torch.long)


def icon_scale_pool(x: torch.Tensor, parent_index: torch.Tensor, n_parent: int):
    """
    Aggregate fine-cell features to their parent cells (scale pooling).

    :param x: Fine-cell feature tensor of shape ``(n_fine, ..., f)``.
    :param parent_index: LongTensor of shape ``(n_fine,)`` mapping fine cells
        to parent cell indices, as returned by ``icon_get_parent_index``.
    :param n_parent: Number of parent cells.
    :return: Parent-cell tensor of shape ``(n_parent, ..., f)`` holding the
        arithmetic mean of each parent's children.
    """
    out_shape = (n_parent,) + tuple(x.shape[1:])
    index = parent_index.view(-1, *([1] * (x.dim() - 1))).expand_as(x)

    x_parent = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    x_parent = x_parent.scatter_reduce(0, index, x, reduce="mean", include_self=False)

    return x_parent


def icon_scale_unpool(x_parent: torch.Tensor, parent_index: torch.Tensor):
    """
    Broadcast parent-cell features back to their fine-grid children (scale unpooling).

    :param x_parent: Parent-cell feature tensor of shape ``(n_parent, ..., f)``.
    :param parent_index: LongTensor of shape ``(n_fine,)`` mapping fine cells
        to parent cell indices, as returned by ``icon_get_parent_index``.
    :return: Fine-cell tensor of shape ``(n_fine, ..., f)``.
    """
    return x_parent[parent_index]


def scale_constraining_icon(
    x_child: torch.Tensor,
    x_parent: torch.Tensor,
    parent_index: torch.Tensor,
):
    """
    Enforce the zero-mean scale-conservation constraint on an unstructured grid.

    Removes each parent's local mean from its children (leaving zero-mean
    fine-scale residuals) and adds that mean back onto the parent, so the
    combined (parent + children) field quantity is conserved.

    :param x_child: Fine-cell feature tensor of shape ``(n_fine, ..., f)``.
    :param x_parent: Parent-cell feature tensor of shape ``(n_parent, ..., f)``.
    :param parent_index: LongTensor of shape ``(n_fine,)`` mapping fine cells
        to parent cell indices, as returned by ``icon_get_parent_index``.
    :return: Tuple ``(x_child_zero_mean, x_parent_conserved)`` with the same
        shapes as the inputs.
    """
    n_parent = x_parent.shape[0]
    local_mean = icon_scale_pool(x_child, parent_index, n_parent)

    x_child_zero_mean = x_child - icon_scale_unpool(local_mean, parent_index)
    x_parent_conserved = x_parent + local_mean

    return x_child_zero_mean, x_parent_conserved


def icon_grid_to_mgrid(grid_files: Mapping[int, str]):
    """
    Build a list of ICON native-grid levels with coordinates, adjacency, and
    the cross-zoom (parent-child) hierarchy needed for scale pooling.

    Unlike Healpix, ICON refinement levels (``R*B*``) do not follow an
    analytic pixel-count formula, so each zoom level requires its own grid
    file (containing at least ``clon``/``clat`` and ``neighbor_cell_index``,
    plus ``child_cell_index`` and/or ``parent_cell_index`` where available to
    resolve the exact quad-tree hierarchy between consecutive zoom levels).

    :param grid_files: Mapping from zoom level to the path of the
        corresponding ICON grid file.
    :return: List of grid dicts with "coords", "adjc", "adjc_mask", "zoom",
        and "parent_index" (``None`` for the coarsest level; otherwise a
        LongTensor mapping this level's cells to the previous, coarser
        level's cells), sorted by ascending zoom.
    """
    grids = []
    datasets: Dict[int, xr.Dataset] = {}

    zooms = sorted(int(z) for z in grid_files.keys())

    try:
        for zoom in zooms:
            datasets[zoom] = xr.open_dataset(grid_files[zoom])

        for i, zoom in enumerate(zooms):
            ds = datasets[zoom]
            adjc, adjc_mask = icon_get_adjacent_cell_indices(ds)
            coords = get_coords_as_tensor(ds, grid_type="cell")

            if i == 0:
                parent_index = None
            else:
                parent_index = icon_get_parent_index(
                    ds,
                    datasets[zooms[i - 1]],
                    fine_coords=coords,
                )

            grid_lvl = {
                "coords": coords,
                "adjc": adjc,
                "adjc_mask": adjc_mask,
                "zoom": zoom,
                "parent_index": parent_index,
            }
            grids.append(grid_lvl)
    finally:
        for ds in datasets.values():
            ds.close()

    return grids


def encode_zooms(
    x_zooms: Dict[int, torch.Tensor],
    sample_configs: Dict,
    patch_index_zooms: Dict
):
    """
    Encode zoom levels as residuals from their next-coarser zoom.

    :param x_zooms: Dict mapping zoom level to tensor ``(b, v, t, n, d, f)``.
    :param sample_configs: Sampling configuration per zoom.
    :param patch_index_zooms: Patch indices per zoom.
    :return: Updated dict with residualized tensors.
    """

    zooms = sorted(x_zooms.keys(),reverse=True)

    for k in range(len(zooms)-1):
        zoom = zooms[k]
        zoom_h = zooms[k+1]
        
        x = x_zooms[zoom]
        x_h = x_zooms[zoom_h]

        if x.ndim < 6:
            x = x.unsqueeze(dim=0)
            x_h = x_h.unsqueeze(dim=0)
            batched = False
        else:
            batched = True

        # Align higher zoom to current zoom and compute residuals.
        bvt = x.shape[:-3]
        x_h_patch = get_matching_time_patch(x_h, zoom_h, zoom, sample_configs, patch_index_zooms)
        x = x.view(*bvt, -1, 4**(zoom-zoom_h), *x.shape[-2:])
        x.sub_(x_h_patch.unsqueeze(dim=-3))

        if batched:
            x_zooms[zoom] = x.reshape(*bvt, -1, *x.shape[-2:])
        else:
            x_zooms[zoom] = x.reshape(*bvt[1:], -1, *x.shape[-2:])

    return x_zooms
    
def decode_zooms(x_zooms: Dict[int, torch.Tensor], sample_configs: Dict, out_zoom: int):
    """
    Reconstruct a target zoom by summing contributions from multiple levels.

    :param x_zooms: Dict mapping zoom level to tensor ``(b, v, t, n, d, f)``.
    :param sample_configs: Sampling configuration per zoom.
    :param out_zoom: Target zoom level to decode.
    :return: Dict containing the reconstructed tensor at out_zoom.
    """
    active_zooms = [zoom for zoom in sorted(x_zooms.keys()) if zoom <= out_zoom]
    if not active_zooms:
        raise ValueError(f"decode_zooms requires at least one zoom <= out_zoom={out_zoom}.")

    remove_batch_dim = False
    x: Optional[torch.Tensor] = None
    current_zoom: Optional[int] = None
    for zoom in active_zooms:
        if x_zooms[zoom].ndim == 4:
            x_zoom = x_zooms[zoom].unsqueeze(dim=0)
            remove_batch_dim = True
        else:
            x_zoom = x_zooms[zoom]

        x_zoom = get_matching_time_patch(x_zoom, zoom, zoom, sample_configs)

        if x is None:
            x = x_zoom
            current_zoom = zoom
            continue

        assert current_zoom is not None
        x = get_matching_time_patch(x, current_zoom, zoom, sample_configs)
        up_factor = 4 ** (zoom - current_zoom)
        x = x.unsqueeze(-3).repeat_interleave(up_factor, dim=-3).reshape(*x.shape[:3], -1, *x.shape[-2:])
        x = x + x_zoom
        current_zoom = zoom

    assert x is not None
    assert current_zoom is not None
    if current_zoom < out_zoom:
        x = get_matching_time_patch(x, current_zoom, out_zoom, sample_configs)
        up_factor = 4 ** (out_zoom - current_zoom)
        x = x.unsqueeze(-3).repeat_interleave(up_factor, dim=-3).reshape(*x.shape[:3], -1, *x.shape[-2:])

    if remove_batch_dim:
        return {out_zoom: x.squeeze(dim=0)}
    return {out_zoom: x}
