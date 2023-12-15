from enum import Enum
from typing import Any

from esmpy.api.constants import ExtrapMethod
from esmpy.api.constants import UnmappedAction as _UnmappedAction


class UnmappedAction(Enum):
    RAISE = _UnmappedAction.ERROR
    IGNORE = _UnmappedAction.IGNORE


UNMAPPED_ACTIONS: dict[str, UnmappedAction] = {
    "raise": UnmappedAction.RAISE,
    "ignore": UnmappedAction.IGNORE,
}


def find_unmapped_action(value: str | UnmappedAction | None) -> UnmappedAction | None:
    value = "none" if value is None else value

    if isinstance(value, str):
        try:
            return UNMAPPED_ACTIONS[value]
        except KeyError:
            return None
    else:
        return value


def find_extrapolation_method(
    value: str | "_ExtrapolationMethod",
) -> "_ExtrapolationMethod":
    if isinstance(value, str):
        try:
            method = EXTRAPOLATION_METHODS[value]
        except KeyError:
            raise UnknownExtrapolationMethod(value) from None
        else:
            return method()
    else:
        return value


class _ExtrapolationMethod:
    code = None

    def asdict(self) -> dict[str, Any]:
        return {"extrap_method": self.code}


class ExtrapolateNone(_ExtrapolationMethod):
    code = ExtrapMethod.NONE


class ExtrapolateNearest(_ExtrapolationMethod):
    code = ExtrapMethod.NEAREST_STOD


class ExtrapolateInverseDistance(_ExtrapolationMethod):
    code = ExtrapMethod.NEAREST_IDAVG

    def __init__(
        self,
        n_source_points: int | None = None,
        distance_exponent: float | None = None,
    ):
        self._n_source_points = n_source_points
        self._distance_exponent = distance_exponent

    def asdict(self) -> dict[str, Any]:
        return super().asdict() | {
            "extrap_num_src_pnts": self._n_source_points,
            "extrap_dist_exponent": self._distance_exponent,
        }


class ExtrapolateCreep(_ExtrapolationMethod):
    code = ExtrapMethod.CREEP_FILL

    def __init__(self, n_levels: int | None = None):
        self._n_levels = n_levels

    def asdict(self) -> dict[str, Any]:
        return super().asdict() | {"extrap_num_levels": self._n_levels}


EXTRAPOLATION_METHODS: dict[str, type[_ExtrapolationMethod]] = {
    "none": ExtrapolateNone,
    "nearest": ExtrapolateNearest,
    "inverse": ExtrapolateInverseDistance,
    "creep": ExtrapolateInverseDistance,
}


class UnknownExtrapolationMethod(Exception):
    def __init__(self, name):
        self._name

    def __str__(self):
        return (
            f"unknown extrapolation method ({self._name}), not one of"
            f" {', '.join(sorted(EXTRAPOLATION_METHODS))}."
        )
