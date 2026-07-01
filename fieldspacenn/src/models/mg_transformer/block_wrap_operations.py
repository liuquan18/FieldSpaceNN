import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ...modules.field_space.field_space_base import (
    GLOBAL_EMBEDDER_CACHE_KEY,
    coarsen_zoom,
    refine_zoom,
)
from ...modules.grids.grid_utils import decode_zooms, encode_zooms, to_zoom
from ...modules.grids.grid_layer import GridLayer

ZoomGroup = Dict[int, torch.Tensor]
ZoomGroups = List[ZoomGroup]
MaskGroup = Optional[Dict[int, torch.Tensor]]
MaskGroups = Optional[Sequence[MaskGroup]]
EmbGroups = Optional[Sequence[Optional[Dict[str, Any]]]]


@dataclass
class BlockWrapContext:
    mask_groups: MaskGroups
    emb_groups: EmbGroups
    sample_configs: Mapping[int, Dict[str, Any]]


class BlockWrapOperation(nn.Module):
    operation_kind = "base"

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, Any]:
        return x_zooms_groups, None

    def post(
        self,
        x_zooms_groups: ZoomGroups,
        state: Any,
        context: BlockWrapContext,
    ) -> ZoomGroups:
        return x_zooms_groups


@dataclass
class MergeGroupsBlockWrapState:
    original_mask_groups: MaskGroups
    original_emb_groups: EmbGroups
    group_shapes: List[Dict[int, torch.Size]]
    zoom_order: List[int]


class BlockWrapConfig:
    operation_kind = "base"

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        raise NotImplementedError

    def get_block_build_overrides(
        self,
        *,
        n_groups_variables: Sequence[int],
        n_groups_depths: Sequence[int],
        base_block_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        del n_groups_variables
        del n_groups_depths
        del base_block_kwargs
        return {}

    def get_stage_input_zooms(self, current_in_zooms: Sequence[int]) -> List[int]:
        return [int(zoom) for zoom in current_in_zooms]

    def get_stage_input_features(
        self,
        *,
        current_in_zooms: Sequence[int],
        current_in_features: Sequence[int],
    ) -> List[int]:
        del current_in_zooms
        return [int(feature) for feature in current_in_features]


def create_block_wrap_operation(
    wrap_conf: Any,
    *,
    grid_layers: nn.ModuleDict,
) -> BlockWrapOperation:
    if not isinstance(wrap_conf, BlockWrapConfig):
        raise TypeError(
            "block_wrap_configs entries must instantiate BlockWrapConfig targets, "
            f"got {type(wrap_conf).__name__}."
        )

    operation = wrap_conf.build(grid_layers=grid_layers)
    if not isinstance(operation, BlockWrapOperation):
        raise TypeError(
            f"{type(wrap_conf).__name__}.build() must return a BlockWrapOperation, "
            f"got {type(operation).__name__}."
        )

    return operation


def validate_block_wrap_operation_sequence(operations: Mapping[str, BlockWrapOperation]) -> None:
    seen_kinds: Dict[str, str] = {}
    for name, operation in operations.items():
        kind = getattr(operation, "operation_kind", type(operation).__name__)
        if kind in seen_kinds:
            raise ValueError(
                f"Duplicate block wrap operation kind `{kind}` is not supported: "
                f"`{seen_kinds[kind]}` and `{name}`."
            )
        seen_kinds[kind] = name


def _normalize_unique_sorted_zooms(zooms: Optional[Sequence[int]]) -> List[int]:
    if zooms is None:
        return []
    return sorted(dict.fromkeys(int(zoom) for zoom in zooms))


def _extract_patch_index_zooms(sample_configs: Mapping[int, Dict[str, Any]]) -> Dict[int, Any]:
    patch_index_zooms: Dict[int, Any] = {}
    for zoom, cfg in sample_configs.items():
        if not isinstance(zoom, int) or not isinstance(cfg, Mapping):
            continue
        if "patch_index" in cfg:
            patch_index_zooms[int(zoom)] = cfg["patch_index"]
    return patch_index_zooms


def _validate_matching_timestep_counts(
    *,
    zooms: Sequence[int],
    sample_configs: Mapping[int, Dict[str, Any]],
) -> None:
    if len(zooms) <= 1:
        return

    timestep_counts = {
        int(zoom): int(sample_configs[int(zoom)]["n_past_ts"]) + int(sample_configs[int(zoom)]["n_future_ts"]) + 1
        for zoom in zooms
    }
    reference_count = next(iter(timestep_counts.values()))
    if any(count != reference_count for count in timestep_counts.values()):
        raise ValueError(
            "ReencodeZoomsBlockWrapOperation requires matching timestep counts across output zooms, "
            f"got {timestep_counts}."
        )


def clone_zoom_groups(x_zooms_groups: Sequence[ZoomGroup]) -> ZoomGroups:
    return [
        {zoom: tensor.clone() for zoom, tensor in x_zooms.items()}
        for x_zooms in x_zooms_groups
    ]


def clone_mask_groups(mask_zooms_groups: MaskGroups) -> Optional[List[MaskGroup]]:
    if mask_zooms_groups is None:
        return None

    return [
        None if mask_zooms is None else {zoom: mask.clone() for zoom, mask in mask_zooms.items()}
        for mask_zooms in mask_zooms_groups
    ]


def apply_saved_residuals(
    *,
    x_zooms_groups: Sequence[ZoomGroup],
    saved_residual_groups: Optional[Sequence[ZoomGroup]],
    mask_zooms_groups: MaskGroups,
    saved_mask_groups: Optional[Sequence[MaskGroup]],
    mode: str,
) -> ZoomGroups:
    if saved_residual_groups is None:
        raise ValueError("Residual block wrap operation requires a saved residual state.")

    if len(x_zooms_groups) != len(saved_residual_groups):
        raise ValueError(
            "Residual block wrap operation requires the same number of groups in the saved and current states."
        )

    if mode not in {"add", "masked"}:
        raise ValueError(f"Unsupported residual mode `{mode}`.")

    x_zooms_groups_out = list(x_zooms_groups)
    for group_idx, (x_zooms, saved_zooms) in enumerate(zip(x_zooms_groups_out, saved_residual_groups)):
        current_zooms = set(x_zooms.keys())
        saved_zoom_keys = set(saved_zooms.keys())
        if current_zooms != saved_zoom_keys:
            raise ValueError(
                f"Residual block wrap operation requires matching zoom keys for group {group_idx}: "
                f"{sorted(saved_zoom_keys)} != {sorted(current_zooms)}."
            )

        current_masks = None if mask_zooms_groups is None else mask_zooms_groups[group_idx]
        saved_masks = None if saved_mask_groups is None else saved_mask_groups[group_idx]
        if mode == "masked" and (current_masks is None or saved_masks is None):
            raise ValueError(
                f"Residual block wrap operation in masked mode requires masks for group {group_idx}."
            )

        for zoom in x_zooms.keys():
            current = x_zooms[zoom]
            saved = saved_zooms[zoom]
            if current.shape != saved.shape:
                raise ValueError(
                    f"Residual block wrap operation requires matching tensor shapes for group {group_idx}, "
                    f"zoom {zoom}: {tuple(saved.shape)} != {tuple(current.shape)}."
                )

            if mode == "add":
                x_zooms[zoom] = saved + current
                continue

            assert current_masks is not None
            assert saved_masks is not None
            if zoom not in current_masks or zoom not in saved_masks:
                raise ValueError(
                    f"Residual block wrap operation in masked mode requires masks for group {group_idx}, zoom {zoom}."
                )

            mask = current_masks[zoom]
            saved_mask = saved_masks[zoom]
            if mask.shape != current.shape or saved_mask.shape != saved.shape:
                raise ValueError(
                    "Residual block wrap operation in masked mode requires mask shapes to match tensor shapes "
                    f"for group {group_idx}, zoom {zoom}."
                )

            if mask.dtype == torch.bool:
                x_zooms[zoom] = torch.where(mask, current, saved)
            else:
                x_zooms[zoom] = saved * (1 - mask) + current * mask

    return x_zooms_groups_out


class ResidualBlockWrapConfig(BlockWrapConfig):
    operation_kind = "residual"

    def __init__(self, mode: str = "add", **kwargs: Any) -> None:
        self.mode: str

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        del grid_layers
        return ResidualBlockWrapOperation(mode=self.mode)


class ResidualBlockWrapOperation(BlockWrapOperation):
    operation_kind = "residual"

    def __init__(self, mode: str = "add") -> None:
        super().__init__()
        self.mode = mode

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, Tuple[ZoomGroups, Optional[List[MaskGroup]]]]:
        saved_residual_groups = clone_zoom_groups(x_zooms_groups)
        saved_mask_groups = clone_mask_groups(context.mask_groups)
        return x_zooms_groups, (saved_residual_groups, saved_mask_groups)

    def post(
        self,
        x_zooms_groups: ZoomGroups,
        state: Tuple[ZoomGroups, Optional[List[MaskGroup]]],
        context: BlockWrapContext,
    ) -> ZoomGroups:
        saved_residual_groups, saved_mask_groups = state
        return apply_saved_residuals(
            x_zooms_groups=x_zooms_groups,
            saved_residual_groups=saved_residual_groups,
            mask_zooms_groups=context.mask_groups,
            saved_mask_groups=saved_mask_groups,
            mode=self.mode,
        )


class ShiftGroupsBlockWrapConfig(BlockWrapConfig):
    operation_kind = "shift_groups"

    def __init__(
        self,
        token_zoom: int,
        zooms: Optional[Sequence[int]] = None,
        q_zooms: Union[Sequence[int], int] = -1,
        kv_zooms: Union[Sequence[int], int] = -1,
        multi_shift: bool = False,
        direction: str = "east",
        **kwargs: Any,
    ) -> None:
        self.token_zoom: int
        self.zooms: Optional[Sequence[int]]
        self.q_zooms: Union[Sequence[int], int]
        self.kv_zooms: Union[Sequence[int], int]
        self.multi_shift: bool
        self.direction: str

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        zooms = _resolve_shift_zooms(self.zooms, self.q_zooms, self.kv_zooms)
        return ShiftGroupsBlockWrapOperation(
            grid_layers=grid_layers,
            token_zoom=self.token_zoom,
            zooms=zooms,
            multi_shift=self.multi_shift,
            direction=self.direction,
        )


class ShiftGroupsBlockWrapOperation(BlockWrapOperation):
    operation_kind = "shift_groups"

    def __init__(
        self,
        *,
        grid_layers: Mapping[str, GridLayer],
        token_zoom: int,
        zooms: Sequence[int],
        multi_shift: bool = False,
        direction: str = "east",
    ) -> None:
        super().__init__()
        self.grid_layers = grid_layers
        self.token_zoom = token_zoom
        self.zooms = list(zooms)
        self.multi_shift = multi_shift
        self.direction = direction

        if not self.zooms:
            raise ValueError("ShiftGroupsBlockWrapOperation requires at least one zoom.")

        if not multi_shift and str(int(token_zoom + 1)) not in grid_layers:
            raise ValueError(
                f"ShiftGroupsBlockWrapOperation requires grid layer `{token_zoom + 1}` when multi_shift is False."
            )

        if multi_shift:
            missing = [zoom for zoom in self.zooms if str(int(zoom)) not in grid_layers]
            if missing:
                raise ValueError(
                    "ShiftGroupsBlockWrapOperation requires grid layers for all shifted zooms, "
                    f"missing {missing}."
                )

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, None]:
        return self._apply_shift(x_zooms_groups, sample_configs=context.sample_configs, reverse=False), None

    def post(
        self,
        x_zooms_groups: ZoomGroups,
        state: None,
        context: BlockWrapContext,
    ) -> ZoomGroups:
        del state
        return self._apply_shift(x_zooms_groups, sample_configs=context.sample_configs, reverse=True)

    def _apply_shift(
        self,
        x_zooms_groups: ZoomGroups,
        *,
        sample_configs: Mapping[int, Dict[str, Any]],
        reverse: bool,
    ) -> ZoomGroups:
        for group_idx, x_zooms in enumerate(x_zooms_groups):
            for zoom in self.zooms:
                if zoom not in x_zooms:
                    raise ValueError(
                        f"ShiftGroupsBlockWrapOperation requires zoom {zoom} in group {group_idx}."
                    )

                if zoom not in sample_configs:
                    raise ValueError(
                        f"ShiftGroupsBlockWrapOperation requires sample_configs for zoom {zoom}."
                    )

                grid_layer = (
                    self.grid_layers[str(int(zoom))]
                    if self.multi_shift
                    else self.grid_layers[str(int(self.token_zoom + 1))]
                )
                x_zooms[zoom] = grid_layer.apply_shift(
                    x_zooms[zoom],
                    self.direction,
                    **sample_configs[zoom],
                    reverse=reverse,
                )[0]
            x_zooms_groups[group_idx] = x_zooms

        return x_zooms_groups


class RefineGroupsBlockWrapConfig(BlockWrapConfig):
    operation_kind = "refine_groups"

    def __init__(self, refine_zooms: Mapping[int, int], **kwargs: Any) -> None:
        self.refine_zooms: Dict[int, int]

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            elif input_name == "refine_zooms":
                setattr(self, input_name, {int(key): int(val) for key, val in value.items()})
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        del grid_layers
        return RefineGroupsBlockWrapOperation(refine_zooms=self.refine_zooms)


class RefineGroupsBlockWrapOperation(BlockWrapOperation):
    operation_kind = "refine_groups"

    def __init__(self, *, refine_zooms: Mapping[int, int]) -> None:
        super().__init__()
        self.refine_zooms = dict(refine_zooms)
        if not self.refine_zooms:
            raise ValueError("RefineGroupsBlockWrapOperation requires a non-empty refine_zooms mapping.")

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, None]:
        del context
        for group_idx, x_zooms in enumerate(x_zooms_groups):
            for in_zoom, out_zoom in self.refine_zooms.items():
                if in_zoom not in x_zooms:
                    raise ValueError(
                        f"RefineGroupsBlockWrapOperation requires zoom {in_zoom} in group {group_idx}."
                    )
                x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
            x_zooms_groups[group_idx] = x_zooms
        return x_zooms_groups, None


class CoarsenGroupsBlockWrapConfig(BlockWrapConfig):
    operation_kind = "coarsen_groups"

    def __init__(self, coarsen_zooms: Mapping[int, int], **kwargs: Any) -> None:
        self.coarsen_zooms: Dict[int, int]

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            elif input_name == "coarsen_zooms":
                setattr(self, input_name, {int(key): int(val) for key, val in value.items()})
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        del grid_layers
        return CoarsenGroupsBlockWrapOperation(coarsen_zooms=self.coarsen_zooms)


class CoarsenGroupsBlockWrapOperation(BlockWrapOperation):
    operation_kind = "coarsen_groups"

    def __init__(self, *, coarsen_zooms: Mapping[int, int]) -> None:
        super().__init__()
        self.coarsen_zooms = dict(coarsen_zooms)
        if not self.coarsen_zooms:
            raise ValueError("CoarsenGroupsBlockWrapOperation requires a non-empty coarsen_zooms mapping.")

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, None]:
        del context
        for group_idx, x_zooms in enumerate(x_zooms_groups):
            for in_zoom, out_zoom in self.coarsen_zooms.items():
                if in_zoom not in x_zooms:
                    raise ValueError(
                        f"CoarsenGroupsBlockWrapOperation requires zoom {in_zoom} in group {group_idx}."
                    )
                x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
            x_zooms_groups[group_idx] = x_zooms
        return x_zooms_groups, None


class ReencodeZoomsBlockWrapConfig(BlockWrapConfig):
    operation_kind = "reencode_zooms"

    def __init__(
        self,
        decode_zoom: Optional[int] = None,
        out_zooms: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        self.decode_zoom: Optional[int]
        self.out_zooms: Optional[List[int]]

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            elif input_name == "out_zooms":
                setattr(self, input_name, None if value is None else [int(zoom) for zoom in value])
            elif input_name == "decode_zoom":
                setattr(self, input_name, None if value is None else int(value))
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        del grid_layers
        return ReencodeZoomsBlockWrapOperation(
            decode_zoom=self.decode_zoom,
            out_zooms=self.out_zooms,
        )

    def get_stage_input_zooms(self, current_in_zooms: Sequence[int]) -> List[int]:
        return self._resolve_stage_output_zooms(current_in_zooms)

    def get_stage_input_features(
        self,
        *,
        current_in_zooms: Sequence[int],
        current_in_features: Sequence[int],
    ) -> List[int]:
        stage_input_zooms = self._resolve_stage_output_zooms(current_in_zooms)
        feature_by_zoom = {
            int(zoom): int(feature)
            for zoom, feature in zip(current_in_zooms, current_in_features)
        }
        reencoded_highest_zoom = self._resolve_stage_reencoded_highest_zoom(current_in_zooms)
        reencode_feature = None if reencoded_highest_zoom is None else feature_by_zoom.get(reencoded_highest_zoom)
        if reencoded_highest_zoom is not None and reencode_feature is None:
            raise ValueError(
                "ReencodeZoomsBlockWrapConfig requires at least one re-encodable zoom in the current stage inputs "
                "when computing stage input features, "
                f"got decode_zoom={self.decode_zoom} and in_zooms={list(current_in_zooms)}."
            )

        stage_input_features: List[int] = []
        for zoom in stage_input_zooms:
            feature = feature_by_zoom.get(zoom)
            if feature is not None:
                stage_input_features.append(feature)
                continue
            if reencode_feature is None:
                raise ValueError(
                    "ReencodeZoomsBlockWrapConfig could not infer stage input features for a re-encoded zoom, "
                    f"got zoom={zoom}, decode_zoom={self.decode_zoom}, and in_zooms={list(current_in_zooms)}."
                )
            stage_input_features.append(reencode_feature)

        return stage_input_features

    def _partition_stage_input_zooms(self, current_in_zooms: Sequence[int]) -> Tuple[List[int], List[int]]:
        input_zooms = sorted(dict.fromkeys(int(zoom) for zoom in current_in_zooms))
        if self.decode_zoom is None:
            return input_zooms, []
        reencoded_zooms = [zoom for zoom in input_zooms if zoom <= int(self.decode_zoom)]
        passthrough_zooms = [zoom for zoom in input_zooms if zoom > int(self.decode_zoom)]
        return reencoded_zooms, passthrough_zooms

    def _resolve_stage_reencoded_highest_zoom(self, current_in_zooms: Sequence[int]) -> Optional[int]:
        reencoded_zooms, _ = self._partition_stage_input_zooms(current_in_zooms)
        if not reencoded_zooms:
            return None
        return max(reencoded_zooms)

    def _resolve_stage_output_zooms(self, current_in_zooms: Sequence[int]) -> List[int]:
        reencoded_zooms, passthrough_zooms = self._partition_stage_input_zooms(current_in_zooms)
        reencoded_highest_zoom = self._resolve_stage_reencoded_highest_zoom(current_in_zooms)
        requested_output_zooms = _normalize_unique_sorted_zooms(self.out_zooms)
        if not requested_output_zooms:
            default_output_zooms = []
            if reencoded_highest_zoom is not None:
                default_output_zooms.append(reencoded_highest_zoom)
            default_output_zooms.extend(passthrough_zooms)
            return sorted(dict.fromkeys(default_output_zooms))

        if reencoded_highest_zoom is None:
            invalid_output_zooms = [zoom for zoom in requested_output_zooms if zoom not in passthrough_zooms]
        else:
            invalid_output_zooms = [
                zoom for zoom in requested_output_zooms
                if zoom not in passthrough_zooms and zoom > reencoded_highest_zoom
            ]
        if invalid_output_zooms:
            raise ValueError(
                "ReencodeZoomsBlockWrapConfig requires requested output zooms above the re-encode limit to "
                "already exist as higher untouched inputs, "
                f"got decode_zoom={self.decode_zoom}, in_zooms={list(current_in_zooms)}, and out_zooms={requested_output_zooms}."
            )

        return sorted(dict.fromkeys(requested_output_zooms + passthrough_zooms))


class ReencodeZoomsBlockWrapOperation(BlockWrapOperation):
    operation_kind = "reencode_zooms"

    def __init__(
        self,
        *,
        decode_zoom: Optional[int] = None,
        out_zooms: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.decode_zoom = None if decode_zoom is None else int(decode_zoom)
        self.out_zooms = None if out_zooms is None else [int(zoom) for zoom in out_zooms]

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, None]:
        if not x_zooms_groups:
            return list(x_zooms_groups), None

        int_sample_configs = {
            int(zoom): cfg
            for zoom, cfg in context.sample_configs.items()
            if isinstance(zoom, int)
        }

        reencoded_highest_zoom = self._resolve_reencoded_highest_zoom(x_zooms_groups, int_sample_configs)
        output_zooms = self._resolve_output_zooms(
            x_zooms_groups=x_zooms_groups,
            reencoded_highest_zoom=reencoded_highest_zoom,
        )

        missing_sample_configs = [
            zoom for zoom in ([reencoded_highest_zoom] if reencoded_highest_zoom is not None else []) + output_zooms
            if zoom not in int_sample_configs
        ]
        if missing_sample_configs:
            raise ValueError(
                "ReencodeZoomsBlockWrapOperation requires sample_configs for all decoded/output zooms, "
                f"missing {sorted(dict.fromkeys(missing_sample_configs))}."
            )

        _validate_matching_timestep_counts(zooms=output_zooms, sample_configs=int_sample_configs)

        patch_index_zooms = _extract_patch_index_zooms(int_sample_configs)
        reencoded_groups: ZoomGroups = []
        for group_idx, x_zooms in enumerate(x_zooms_groups):
            if not x_zooms:
                reencoded_groups.append({})
                continue

            reencoded_group, passthrough_group = self._partition_group_zooms(x_zooms)
            group_outputs: ZoomGroup = {
                int(zoom): tensor
                for zoom, tensor in passthrough_group.items()
                if int(zoom) in output_zooms
            }

            reencoded_output_zooms = (
                [] if reencoded_highest_zoom is None else [zoom for zoom in output_zooms if zoom <= reencoded_highest_zoom]
            )
            if reencoded_highest_zoom is not None and reencoded_output_zooms:
                decoded_group = decode_zooms(
                    {int(zoom): tensor for zoom, tensor in reencoded_group.items()},
                    sample_configs=int_sample_configs,
                    out_zoom=reencoded_highest_zoom,
                )
                if reencoded_highest_zoom not in decoded_group:
                    raise ValueError(
                        f"ReencodeZoomsBlockWrapOperation failed to decode group {group_idx} to zoom {reencoded_highest_zoom}."
                    )

                decoded_highest = decoded_group[reencoded_highest_zoom]
                encoded_inputs: Dict[int, torch.Tensor] = {}
                for zoom in reencoded_output_zooms:
                    if zoom == reencoded_highest_zoom:
                        encoded_inputs[zoom] = decoded_highest
                        continue

                    encoded_inputs[zoom] = to_zoom(
                        decoded_highest,
                        in_zoom=reencoded_highest_zoom,
                        out_zoom=zoom,
                    )[0]

                if encoded_inputs:
                    group_outputs.update(
                        encode_zooms(
                            encoded_inputs,
                            sample_configs=int_sample_configs,
                            patch_index_zooms=patch_index_zooms,
                        )
                    )

            reencoded_groups.append({zoom: group_outputs[zoom] for zoom in output_zooms if zoom in group_outputs})

        return reencoded_groups, None

    def _partition_available_zooms(
        self,
        x_zooms_groups: Sequence[ZoomGroup],
    ) -> Tuple[List[int], List[int]]:
        available_zooms = sorted({
            int(zoom)
            for group in x_zooms_groups
            for zoom in group.keys()
        })
        if self.decode_zoom is None:
            return available_zooms, []
        reencoded_zooms = [zoom for zoom in available_zooms if zoom <= int(self.decode_zoom)]
        passthrough_zooms = [zoom for zoom in available_zooms if zoom > int(self.decode_zoom)]
        return reencoded_zooms, passthrough_zooms

    def _resolve_reencoded_highest_zoom(
        self,
        x_zooms_groups: Sequence[ZoomGroup],
        sample_configs: Mapping[int, Dict[str, Any]],
    ) -> Optional[int]:
        reencoded_zooms, passthrough_zooms = self._partition_available_zooms(x_zooms_groups)
        if reencoded_zooms:
            return max(reencoded_zooms)
        if passthrough_zooms:
            if self.decode_zoom is not None:
                return None
            return max(passthrough_zooms)

        if sample_configs:
            sample_zooms = sorted(int(zoom) for zoom in sample_configs.keys())
            if self.decode_zoom is None:
                return max(sample_zooms)
            filtered_sample_zooms = [zoom for zoom in sample_zooms if zoom <= int(self.decode_zoom)]
            return max(filtered_sample_zooms) if filtered_sample_zooms else None

        raise ValueError("ReencodeZoomsBlockWrapOperation could not infer a re-encoded zoom.")

    def _resolve_output_zooms(
        self,
        *,
        x_zooms_groups: Sequence[ZoomGroup],
        reencoded_highest_zoom: Optional[int],
    ) -> List[int]:
        _, passthrough_zooms = self._partition_available_zooms(x_zooms_groups)
        requested_output_zooms = _normalize_unique_sorted_zooms(self.out_zooms)
        if not requested_output_zooms:
            default_output_zooms = []
            if reencoded_highest_zoom is not None:
                default_output_zooms.append(reencoded_highest_zoom)
            default_output_zooms.extend(passthrough_zooms)
            return sorted(dict.fromkeys(default_output_zooms))

        if reencoded_highest_zoom is None:
            invalid_output_zooms = [zoom for zoom in requested_output_zooms if zoom not in passthrough_zooms]
        else:
            invalid_output_zooms = [
                zoom for zoom in requested_output_zooms
                if zoom not in passthrough_zooms and zoom > reencoded_highest_zoom
            ]
        if invalid_output_zooms:
            raise ValueError(
                "ReencodeZoomsBlockWrapOperation requires requested output zooms above the re-encode limit to "
                "already exist as higher untouched inputs, "
                f"got decode_zoom={self.decode_zoom}, input_zooms={sorted({int(zoom) for group in x_zooms_groups for zoom in group.keys()})}, "
                f"and out_zooms={requested_output_zooms}."
            )

        return sorted(dict.fromkeys(requested_output_zooms + passthrough_zooms))

    def _partition_group_zooms(self, x_zooms: ZoomGroup) -> Tuple[ZoomGroup, ZoomGroup]:
        if self.decode_zoom is None:
            return dict(x_zooms), {}
        reencoded_group = {
            int(zoom): tensor
            for zoom, tensor in x_zooms.items()
            if int(zoom) <= int(self.decode_zoom)
        }
        passthrough_group = {
            int(zoom): tensor
            for zoom, tensor in x_zooms.items()
            if int(zoom) > int(self.decode_zoom)
        }
        return reencoded_group, passthrough_group


class MergeGroupsBlockWrapConfig(BlockWrapConfig):
    operation_kind = "merge_groups"

    def __init__(
        self,
        variable_embedder_mode: str = "unique_per_depth",
        block_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.variable_embedder_mode: str
        self.block_kwargs: Dict[str, Any]

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            elif input_name == "block_kwargs":
                setattr(self, input_name, {} if value is None else dict(value))
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        del grid_layers
        return MergeGroupsBlockWrapOperation(variable_embedder_mode=self.variable_embedder_mode)

    def get_block_build_overrides(
        self,
        *,
        n_groups_variables: Sequence[int],
        n_groups_depths: Sequence[int],
        base_block_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        del base_block_kwargs
        merged_n_groups_variables = _merge_group_variable_counts(
            n_groups_variables=n_groups_variables,
            n_groups_depths=n_groups_depths,
            variable_embedder_mode=self.variable_embedder_mode,
        )
        overrides = {
            "n_groups_variables": merged_n_groups_variables,
            "n_groups_depths": [1],
        }
        overrides.update(self.block_kwargs)
        return overrides


class MergeGroupsBlockWrapOperation(BlockWrapOperation):
    operation_kind = "merge_groups"
    _VARIABLE_ID_KEYS = ("VariableEmbedder", "variables_sampled", "MGEmbedder")

    def __init__(self, variable_embedder_mode: str = "unique_per_depth") -> None:
        super().__init__()
        if variable_embedder_mode not in {"unique_per_depth", "shared_across_depth"}:
            raise ValueError(
                "variable_embedder_mode must be `unique_per_depth` or `shared_across_depth`, "
                f"got `{variable_embedder_mode}`."
            )
        self.variable_embedder_mode = variable_embedder_mode

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, MergeGroupsBlockWrapState]:
        if len(x_zooms_groups) <= 1:
            return x_zooms_groups, MergeGroupsBlockWrapState(
                original_mask_groups=context.mask_groups,
                original_emb_groups=context.emb_groups,
                group_shapes=[{zoom: tensor.shape for zoom, tensor in x_zooms.items()} for x_zooms in x_zooms_groups],
                zoom_order=sorted(x_zooms_groups[0].keys()) if x_zooms_groups else [],
            )

        zoom_order = _validate_group_zoom_keys(x_zooms_groups)
        group_shapes = [{zoom: tensor.shape for zoom, tensor in x_zooms.items()} for x_zooms in x_zooms_groups]
        merged_group = {
            zoom: torch.cat(
                [_fold_depth_into_variable(x_zooms[zoom]) for x_zooms in x_zooms_groups],
                dim=1,
            )
            for zoom in zoom_order
        }

        merged_mask_groups = None
        if context.mask_groups is not None:
            if any(mask_zooms is None for mask_zooms in context.mask_groups):
                raise ValueError(
                    "MergeGroupsBlockWrapOperation requires mask_groups to be either absent or present for every group."
                )
            merged_mask_groups = [
                {
                    zoom: torch.cat(
                        [_fold_depth_into_variable(mask_zooms[zoom]) for mask_zooms in context.mask_groups],
                        dim=1,
                    )
                    for zoom in zoom_order
                }
            ]

        merged_emb_groups = None
        if context.emb_groups is not None:
            merged_emb_groups = [self._merge_emb_groups(context.emb_groups, group_shapes)]

        state = MergeGroupsBlockWrapState(
            original_mask_groups=context.mask_groups,
            original_emb_groups=context.emb_groups,
            group_shapes=group_shapes,
            zoom_order=zoom_order,
        )
        context.mask_groups = merged_mask_groups
        context.emb_groups = merged_emb_groups
        return [merged_group], state

    def post(
        self,
        x_zooms_groups: ZoomGroups,
        state: MergeGroupsBlockWrapState,
        context: BlockWrapContext,
    ) -> ZoomGroups:
        context.mask_groups = state.original_mask_groups
        context.emb_groups = state.original_emb_groups

        if len(state.group_shapes) <= 1:
            return x_zooms_groups

        if len(x_zooms_groups) != 1:
            raise ValueError(
                f"MergeGroupsBlockWrapOperation.post expects exactly one merged group, got {len(x_zooms_groups)}."
            )

        merged_group = x_zooms_groups[0]
        restored_groups: ZoomGroups = []
        zoom_offsets = {zoom: 0 for zoom in state.zoom_order}
        for group_shapes in state.group_shapes:
            restored_group: ZoomGroup = {}
            for zoom in state.zoom_order:
                shape = group_shapes[zoom]
                flat_width = int(shape[1] * shape[4])
                start = zoom_offsets[zoom]
                stop = start + flat_width
                restored_group[zoom] = _unfold_variable_into_depth(merged_group[zoom][:, start:stop], shape)
                zoom_offsets[zoom] = stop
            restored_groups.append(restored_group)

        return restored_groups

    def _merge_emb_groups(
        self,
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        group_shapes: Sequence[Dict[int, torch.Size]],
    ) -> Dict[str, Any]:
        emb_groups_indexed = [
            (group_idx, emb_group)
            for group_idx, emb_group in enumerate(emb_groups)
            if emb_group is not None
        ]
        if not emb_groups_indexed:
            return {}

        merged_emb = {
            key: copy.deepcopy(value)
            for key, value in emb_groups_indexed[0][1].items()
            if key not in self._VARIABLE_ID_KEYS and key not in {"variable_names_sampled", GLOBAL_EMBEDDER_CACHE_KEY}
        }

        variable_name_pieces: List[str] = []
        merged_id_values: Dict[str, List[torch.Tensor]] = {key: [] for key in self._VARIABLE_ID_KEYS}
        next_unique_id = 0

        for group_idx, emb_group in emb_groups_indexed:
            group_shape = next(iter(group_shapes[group_idx].values()))
            n_variables = int(group_shape[1])
            n_depths = int(group_shape[4])

            if self.variable_embedder_mode == "unique_per_depth":
                merged_ids_unique = torch.arange(
                    next_unique_id,
                    next_unique_id + n_variables * n_depths,
                    device=_get_variable_id_tensor(emb_group, n_variables=n_variables).device,
                    dtype=_get_variable_id_tensor(emb_group, n_variables=n_variables).dtype,
                ).view(1, -1).expand(_get_variable_id_tensor(emb_group, n_variables=n_variables).shape[0], -1)
                next_unique_id += n_variables * n_depths

            for key in self._VARIABLE_ID_KEYS:
                if key in emb_group:
                    key_ids = _get_variable_id_tensor({key: emb_group[key]}, n_variables=n_variables)
                    if self.variable_embedder_mode == "unique_per_depth":
                        merged_id_values[key].append(merged_ids_unique)
                    else:
                        merged_id_values[key].append(
                            key_ids.unsqueeze(-1).expand(-1, -1, n_depths).reshape(key_ids.shape[0], -1)
                        )

            if "variable_names_sampled" in emb_group:
                names = list(emb_group["variable_names_sampled"])
                if len(names) != n_variables:
                    raise ValueError(
                        f"variable_names_sampled must have length {n_variables}, got {len(names)} for group {group_idx}."
                    )
                for name in names:
                    variable_name_pieces.extend([name] * n_depths)

        for key, pieces in merged_id_values.items():
            if pieces:
                merged_emb[key] = torch.cat(pieces, dim=1)

        if variable_name_pieces:
            merged_emb["variable_names_sampled"] = variable_name_pieces

        return merged_emb


def _resolve_shift_zooms(
    zooms: Optional[Sequence[int]],
    q_zooms: Union[Sequence[int], int],
    kv_zooms: Union[Sequence[int], int],
) -> List[int]:
    if zooms is not None:
        return [int(zoom) for zoom in zooms]

    ordered_zooms: List[int] = []
    for zoom_collection in (q_zooms, kv_zooms):
        if isinstance(zoom_collection, int):
            if zoom_collection == -1:
                continue
            ordered_zooms.append(int(zoom_collection))
            continue

        for zoom in zoom_collection:
            zoom_int = int(zoom)
            if zoom_int not in ordered_zooms:
                ordered_zooms.append(zoom_int)

    if not ordered_zooms:
        raise ValueError(
            "ShiftGroupsBlockWrapConfig requires either `zooms` or at least one configured q_zooms/kv_zooms entry."
        )

    return ordered_zooms


def _fold_depth_into_variable(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 6:
        raise ValueError(f"Expected a 6D tensor shaped like (b, v, t, n, d, f), got {tuple(x.shape)}.")
    return x.permute(0, 1, 4, 2, 3, 5).reshape(x.shape[0], x.shape[1] * x.shape[4], x.shape[2], x.shape[3], 1, x.shape[5])


def _unfold_variable_into_depth(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if x.ndim != 6:
        raise ValueError(f"Expected a 6D tensor shaped like (b, vd, t, n, 1, f), got {tuple(x.shape)}.")
    b, v, t, n, d, f = shape
    return x.reshape(b, v, d, t, n, f).permute(0, 1, 3, 4, 2, 5)


def _validate_group_zoom_keys(x_zooms_groups: Sequence[ZoomGroup]) -> List[int]:
    if not x_zooms_groups:
        return []

    zoom_order = sorted(x_zooms_groups[0].keys())
    for group_idx, x_zooms in enumerate(x_zooms_groups[1:], start=1):
        current_zoom_order = sorted(x_zooms.keys())
        if current_zoom_order != zoom_order:
            raise ValueError(
                "MergeGroupsBlockWrapOperation requires matching zoom keys across groups, "
                f"but group 0 has {zoom_order} and group {group_idx} has {current_zoom_order}."
            )
    return zoom_order


def _get_variable_id_tensor(emb_group: Mapping[str, Any], *, n_variables: int) -> torch.Tensor:
    for key in ("variables_sampled", "VariableEmbedder", "MGEmbedder"):
        if key not in emb_group:
            continue
        value = emb_group[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"`{key}` must be a tensor, got {type(value).__name__}.")
        if value.ndim == 1:
            value = value.view(1, -1)
        if value.ndim != 2 or value.shape[1] != n_variables:
            raise ValueError(
                f"`{key}` must have shape (batch, {n_variables}), got {tuple(value.shape)}."
            )
        return value
    raise KeyError("MergeGroupsBlockWrapOperation requires one of `variables_sampled`, `VariableEmbedder`, or `MGEmbedder`.")


def _embedding_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_embedding_values_equal(left[key], right[key]) for key in left.keys())
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_embedding_values_equal(left_item, right_item) for left_item, right_item in zip(left, right))
    return left == right


def _merge_group_variable_counts(
    *,
    n_groups_variables: Sequence[int],
    n_groups_depths: Sequence[int],
    variable_embedder_mode: str,
) -> List[int]:
    if len(n_groups_variables) != len(n_groups_depths):
        raise ValueError(
            "MergeGroupsBlockWrapConfig requires n_groups_variables and n_groups_depths to have the same length, "
            f"got {len(n_groups_variables)} and {len(n_groups_depths)}."
        )

    if variable_embedder_mode == "unique_per_depth":
        return [int(sum(int(n_variables) * int(n_depths) for n_variables, n_depths in zip(n_groups_variables, n_groups_depths)))]

    if variable_embedder_mode == "shared_across_depth":
        return [int(sum(int(n_variables) for n_variables in n_groups_variables))]

    raise ValueError(
        "variable_embedder_mode must be `unique_per_depth` or `shared_across_depth`, "
        f"got `{variable_embedder_mode}`."
    )
