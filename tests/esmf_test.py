import numpy as np
import pytest
from landlab import RasterModelGrid
from landlab_regrid.esmf import _create_grid
from landlab_regrid.esmf import _create_mesh
from landlab_regrid.esmf import create_loc_stream
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("point", ("corner", "node"))
def test_to_grid(point):
    grid = RasterModelGrid((3, 4), xy_spacing=(1.0, 2.0), xy_of_lower_left=(1.0, 3.0))

    esmf_grid = _create_grid(grid, point=point)

    assert_array_equal(
        esmf_grid.get_coords(0).reshape(-1), getattr(grid, f"x_of_{point}")
    )
    assert_array_equal(
        esmf_grid.get_coords(1).reshape(-1), getattr(grid, f"y_of_{point}")
    )


@pytest.mark.parametrize("point", ("corner", "node"))
def test_to_mesh(point):
    grid = RasterModelGrid((5, 7), xy_spacing=(1.0, 2.0), xy_of_lower_left=(1.0, 3.0))

    esmf_mesh = _create_mesh(grid, point=point)

    assert_array_equal(
        esmf_mesh.get_coords(0).reshape(-1), getattr(grid, f"x_of_{point}")
    )
    assert_array_equal(
        esmf_mesh.get_coords(1).reshape(-1), getattr(grid, f"y_of_{point}")
    )


def test_to_loc_stream():
    x = np.linspace(0.0, 10.0)
    y = x**2
    loc_stream = create_loc_stream(np.c_[x, y])

    assert np.all(loc_stream["ESMF:X"] == x)
    assert np.all(loc_stream["x_of_point"] == x)
    assert np.all(loc_stream["ESMF:Y"] == y)
    assert np.all(loc_stream["y_of_point"] == y)
