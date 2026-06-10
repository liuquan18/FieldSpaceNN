from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ...modules.field_space.field_space_base import DiffDecoder
from .mg_base_model import MG_base_model, create_encoder_decoder_block, create_missing_zooms
from .residual_blocks import ResidualApplyBlock, ResidualSaveBlock

class MG_Transformer(MG_base_model):
    """
    Multi-grid transformer composed of configurable encoder/decoder blocks.
    """

    def __init__(
        self,
        mgrids: Sequence[Mapping[str, Any]],
        block_configs: Mapping[str, Any],
        in_zooms: Sequence[int],
        in_features: int = 1,
        n_groups_variables: Sequence[int] = [1],
        n_groups_depths: Optional[Sequence[int]] = None,
        shared_indexed_group_variables: Optional[Sequence[bool]] = None,
        shared_indexed_group_depths: Optional[Sequence[bool]] = None,
        shared_indexed_group_space: Optional[Sequence[bool]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the multi-grid transformer and its block stack.

        :param mgrids: Multi-grid configuration used by the base model.
        :param block_configs: Mapping of block configurations.
        :param in_zooms: Input zoom levels used by the model.
        :param in_features: Number of input features per variable.
        :param n_groups_variables: Number of variable groups for attention layers.
        :param kwargs: Additional arguments forwarded to block factories.
        :return: None.
        """
        super().__init__(mgrids)

        self.in_zooms: Sequence[int] = in_zooms
        self.in_features: int = in_features
        self.n_groups_depths: Sequence[int] = (
            list(n_groups_depths) if n_groups_depths is not None else [1] * len(n_groups_variables)
        )
        self.shared_indexed_group_variables: Sequence[bool] = (
            list(shared_indexed_group_variables)
            if shared_indexed_group_variables is not None
            else [False] * len(n_groups_variables)
        )
        self.shared_indexed_group_depths: Sequence[bool] = (
            list(shared_indexed_group_depths)
            if shared_indexed_group_depths is not None
            else [False] * len(n_groups_variables)
        )
        self.shared_indexed_group_space: Sequence[bool] = (
            list(shared_indexed_group_space)
            if shared_indexed_group_space is not None
            else [False] * len(n_groups_variables)
        )

        self.Blocks: nn.ModuleDict = nn.ModuleDict()

        in_features = [in_features] * len(in_zooms)

        for block_key, block_conf in block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = create_encoder_decoder_block(
                block_conf,
                in_zooms,
                in_features,
                n_groups_variables,
                self.grid_layers,
                self.n_groups_depths,
                self.shared_indexed_group_variables,
                self.shared_indexed_group_depths,
                self.shared_indexed_group_space,
            )

            self.Blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        if self.Blocks:
            list(self.Blocks.values())[-1].out_features = [in_features[0]]

        self.decoder: DiffDecoder = DiffDecoder()

    def decode(
        self,
        x_zooms: Dict[int, torch.Tensor],
        sample_configs: Mapping[int, Any],
        out_zoom: Optional[int] = None,
        emb: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Decode a single zoom-mapping into a requested zoom.

        :param x_zooms: Mapping from zoom level to tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :param emb: Optional embedding dictionary for decoding.
        :return: Decoded zoom mapping.
        """
        emb = emb or {}
        return self.decoder(x_zooms, emb=emb, sample_configs=sample_configs, out_zoom=out_zoom)


    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Dict[int, torch.Tensor]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        sample_configs: Mapping[int, Any] = {},
        out_zoom: Optional[int] = None,
    ) -> Sequence[Dict[int, torch.Tensor]]:
        """
        Forward pass through the multi-grid transformer.

        :param x_zooms_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param mask_zooms_groups: Optional list of mask mappings aligned with inputs, each
            matching the data tensor shape ``(b, v, t, n, d, f)``.
        :param emb_groups: Optional list of embedding dictionaries aligned with inputs.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :return: Output zoom-group mappings.
        """

        x_zooms_groups, mask_zooms_groups, emb_groups, sample_configs = create_missing_zooms(
            x_zooms_groups, self.in_zooms, mask_zooms_groups, emb_groups, sample_configs=sample_configs)

        saved_residual_groups: Optional[List[Dict[int, torch.Tensor]]] = None
        saved_mask_groups: Optional[List[Optional[Dict[int, torch.Tensor]]]] = None

        for block in self.Blocks.values():
            if isinstance(block, ResidualSaveBlock):
                saved_residual_groups = self.clone_zoom_groups(x_zooms_groups)
                saved_mask_groups = self.clone_mask_groups(mask_zooms_groups)
                continue

            if isinstance(block, ResidualApplyBlock):
                x_zooms_groups = self.apply_saved_residuals(
                    x_zooms_groups=x_zooms_groups,
                    saved_residual_groups=saved_residual_groups,
                    mask_zooms_groups=mask_zooms_groups,
                    saved_mask_groups=saved_mask_groups,
                    block=block,
                )
                if block.clear_after_apply:
                    saved_residual_groups = None
                    saved_mask_groups = None
                continue

            x_zooms_groups = block(
                x_zooms_groups,
                sample_configs=sample_configs,
                mask_groups=mask_zooms_groups,
                emb_groups=emb_groups,
            )

        if out_zoom is not None:
            for i, x_zooms in enumerate(x_zooms_groups):
                x_zooms_groups[i] = (
                    self.decoder(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)
                    if x_zooms
                    else {}
                )

        return x_zooms_groups

    def clone_zoom_groups(
        self,
        x_zooms_groups: Sequence[Dict[int, torch.Tensor]],
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Clone a sequence of zoom groups.

        :param x_zooms_groups: Sequence of zoom-to-tensor mappings.
        :return: Cloned zoom mappings.
        """
        return [
            {zoom: tensor.clone() for zoom, tensor in x_zooms.items()}
            for x_zooms in x_zooms_groups
        ]

    def clone_mask_groups(
        self,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
    ) -> Optional[List[Optional[Dict[int, torch.Tensor]]]]:
        """
        Clone a sequence of mask groups.

        :param mask_zooms_groups: Optional sequence of zoom-to-mask mappings.
        :return: Cloned mask mappings.
        """
        if mask_zooms_groups is None:
            return None

        return [
            None if mask_zooms is None else {zoom: mask.clone() for zoom, mask in mask_zooms.items()}
            for mask_zooms in mask_zooms_groups
        ]

    def apply_saved_residuals(
        self,
        x_zooms_groups: Sequence[Dict[int, torch.Tensor]],
        saved_residual_groups: Optional[Sequence[Dict[int, torch.Tensor]]],
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
        saved_mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
        block: ResidualApplyBlock,
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Apply a saved top-level residual state to the current zoom groups.

        :param x_zooms_groups: Current zoom mappings.
        :param saved_residual_groups: Previously saved zoom mappings.
        :param mask_zooms_groups: Current mask mappings.
        :param saved_mask_groups: Saved mask mappings aligned with the residual state.
        :param block: Residual apply marker containing mode settings.
        :return: Updated zoom mappings.
        """
        if saved_residual_groups is None:
            raise ValueError("ResidualApplyBlock requires a saved residual state.")

        if len(x_zooms_groups) != len(saved_residual_groups):
            raise ValueError(
                "ResidualApplyBlock requires the same number of groups in the saved and current states."
            )

        if block.mode not in {"add", "masked"}:
            raise ValueError(f"Unsupported residual mode `{block.mode}`.")

        x_zooms_groups_out = list(x_zooms_groups)
        for group_idx, (x_zooms, saved_zooms) in enumerate(zip(x_zooms_groups_out, saved_residual_groups)):
            current_zooms = set(x_zooms.keys())
            saved_zoom_keys = set(saved_zooms.keys())
            if current_zooms != saved_zoom_keys:
                raise ValueError(
                    f"ResidualApplyBlock requires matching zoom keys for group {group_idx}: "
                    f"{sorted(saved_zoom_keys)} != {sorted(current_zooms)}."
                )

            current_masks = None if mask_zooms_groups is None else mask_zooms_groups[group_idx]
            saved_masks = None if saved_mask_groups is None else saved_mask_groups[group_idx]
            if block.mode == "masked" and (current_masks is None or saved_masks is None):
                raise ValueError(
                    f"ResidualApplyBlock in masked mode requires masks for group {group_idx}."
                )

            for zoom in x_zooms.keys():
                current = x_zooms[zoom]
                saved = saved_zooms[zoom]
                if current.shape != saved.shape:
                    raise ValueError(
                        f"ResidualApplyBlock requires matching tensor shapes for group {group_idx}, "
                        f"zoom {zoom}: {tuple(saved.shape)} != {tuple(current.shape)}."
                    )

                if block.mode == "add":
                    x_zooms[zoom] = saved + current
                    continue

                if zoom not in current_masks or zoom not in saved_masks:
                    raise ValueError(
                        f"ResidualApplyBlock in masked mode requires masks for group {group_idx}, zoom {zoom}."
                    )

                mask = current_masks[zoom]
                saved_mask = saved_masks[zoom]
                if mask.shape != current.shape or saved_mask.shape != saved.shape:
                    raise ValueError(
                        f"ResidualApplyBlock in masked mode requires mask shapes to match tensor shapes "
                        f"for group {group_idx}, zoom {zoom}."
                    )

                if mask.dtype == torch.bool:
                    x_zooms[zoom] = torch.where(mask, current, saved)
                else:
                    x_zooms[zoom] = saved * (1 - mask) + current * mask

        return x_zooms_groups_out
