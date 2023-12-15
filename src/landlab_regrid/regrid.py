import esmpy
from esmpy.api.constants import MeshLoc
from esmpy.api.constants import RegridMethod
from landlab_regrid.constants import _ExtrapolationMethod
from landlab_regrid.constants import find_extrapolation_method
from landlab_regrid.constants import find_unmapped_action
from landlab_regrid.constants import UnmappedAction
from landlab_regrid.esmf import create


class _Regridder:
    method = None

    def __init__(
        self,
        src,
        dst,
        unmapped: str | UnmappedAction = "ignore",
        extrapolate: str | _ExtrapolationMethod = "none",
        from_: str = "node",
        to: str = "node",
    ):
        self._unmapped = find_unmapped_action(unmapped)
        self._from = self._validate_location(from_)
        self._to = self._validate_location(to)

        extrapolation_method = find_extrapolation_method(extrapolate)

        self._src_field = esmpy.Field(
            create(
                src,
                point="node" if from_ in ("node", "patch") else "corner",
                force_mesh=True,
            ),
            meshloc=MeshLoc.NODE if from_ in ("node", "corner") else MeshLoc.ELEMENT,
        )

        self._dst_field = esmpy.Field(
            create(
                dst,
                point="node" if to in ("node", "patch") else "corner",
                force_mesh=True,
            ),
            meshloc=MeshLoc.NODE if to in ("node", "corner") else MeshLoc.ELEMENT,
        )

        self._regrid = esmpy.Regrid(
            self._src_field,
            self._dst_field,
            **{"regrid_method": self.method, "unmapped_action": self._unmapped}
            | extrapolation_method.asdict(),
        )

    @staticmethod
    def _validate_location(at):
        if at not in {"node", "corner", "patch", "cell"}:
            raise ValueError(f"unknown location ({at})")
        else:
            return at

    def regrid(self, values):
        self._src_field.data[:] = values.reshape(-1)
        dst_field = self._regrid(self._src_field, self._dst_field)
        return dst_field.data.copy()

    def __call__(self, values):
        return self.regrid(values)


class PatchRegridder(_Regridder):
    method = RegridMethod.BILINEAR


class BilinearRegridder(_Regridder):
    method = RegridMethod.PATCH


class NearestNeighborRegridder(_Regridder):
    method = None

    def __init__(self, *args, dtos: bool = True, **kwds):
        if dtos:
            self.method = RegridMethod.NEAREST_DTOS
        else:
            self.method = RegridMethod.NEAREST_STOD

        super().__init__(*args, **kwds)


class ConserveRegridder(_Regridder):
    method = None

    def __init__(self, src, dst, order: int = 1, **kwds):
        if order == 1:
            self.method = RegridMethod.CONSERVE
        elif order == 2:
            self.method = RegridMethod.CONSERVE_2ND
        else:
            raise ValueError(f"order must be 1 or 2 ({order})")

        from_ = kwds.get("from_", "node")
        to = kwds.get("to", "node")

        if from_ not in ("patch", "cell") or to not in ("patch", "cell"):
            raise ValueError(
                "both 'from_' and 'to' arguments must be either 'patch' or 'cell'"
            )

        super().__init__(src, dst, **kwds)
