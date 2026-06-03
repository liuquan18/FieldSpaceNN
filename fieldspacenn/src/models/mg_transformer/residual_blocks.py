import copy
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn


class ResidualSaveConfig:
    def __init__(self, **kwargs: Any) -> None:
        """
        Store configuration for a residual save marker.

        :param kwargs: Additional keyword arguments assigned as attributes.
        :return: None.
        """
        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            else:
                setattr(self, input_name, value)


class ResidualApplyConfig:
    def __init__(
        self,
        mode: str = "add",
        clear_after_apply: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Store configuration for a residual apply marker.

        :param mode: Residual application mode, one of ``"add"`` or ``"masked"``.
        :param clear_after_apply: Whether to clear the saved residual after applying it.
        :param kwargs: Additional keyword arguments assigned as attributes.
        :return: None.
        """
        self.mode: str
        self.clear_after_apply: bool

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            else:
                setattr(self, input_name, value)


class ResidualSaveBlock(nn.Module):
    """
    Marker block that stores the current model state for a later residual apply.
    """

    def __init__(self, out_zooms: Sequence[int], out_features: Sequence[int]) -> None:
        super().__init__()
        self.out_zooms: List[int] = list(out_zooms)
        self.out_features: List[int] = list(out_features)

    def forward(self, x_zooms_groups: List[Dict[int, torch.Tensor]], **kwargs: Any) -> List[Dict[int, torch.Tensor]]:
        return x_zooms_groups


class ResidualApplyBlock(nn.Module):
    """
    Marker block that applies the previously saved residual state.
    """

    def __init__(
        self,
        out_zooms: Sequence[int],
        out_features: Sequence[int],
        mode: str = "add",
        clear_after_apply: bool = True,
    ) -> None:
        super().__init__()
        self.out_zooms: List[int] = list(out_zooms)
        self.out_features: List[int] = list(out_features)
        self.mode: str = mode
        self.clear_after_apply: bool = clear_after_apply

    def forward(self, x_zooms_groups: List[Dict[int, torch.Tensor]], **kwargs: Any) -> List[Dict[int, torch.Tensor]]:
        return x_zooms_groups
