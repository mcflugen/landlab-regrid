import esmpy

from landlab_regrid.esmf import create_grid


class Regrid:
    def __init__(self, src, dst):
        self._src_field = esmpy.Field(create_grid(src), "value")
        self._dst_field = esmpy.Field(create_grid(dst), "value")

        self._regrid = esmpy.Regrid(
            self._src_field,
            self._dst_field,
            regrid_method=esmpy.RegridMethod.BILINEAR,
            unmapped_action=esmpy.UnmappedAction.IGNORE,
        )

    def regrid(self, values):
        self._src_field.data[:] = values.reshape(self._src_field.data.shape)
        dst_field = self._regrid(self._src_field, self._dst_field)
        return dst_field.data.copy()

    def __call__(self, values):
        return self.regrid(values)
