import pytest
from landlab import RasterModelGrid
from landlab_regrid.esmf import create_grid
from landlab_regrid.esmf import create_mesh
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("point", ("corner", "node"))
def test_to_grid(point):
    grid = RasterModelGrid((3, 4), xy_spacing=(1.0, 2.0), xy_of_lower_left=(1.0, 3.0))

    esmf_grid = create_grid(grid, point=point)

    assert_array_equal(
        esmf_grid.get_coords(0).reshape(-1), getattr(grid, f"x_of_{point}")
    )
    assert_array_equal(
        esmf_grid.get_coords(1).reshape(-1), getattr(grid, f"y_of_{point}")
    )
