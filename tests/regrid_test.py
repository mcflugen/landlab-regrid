from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from landlab import HexModelGrid
from landlab import RasterModelGrid
from landlab_regrid.regrid import BilinearRegridder
from landlab_regrid.regrid import ConserveRegridder
from landlab_regrid.regrid import NearestNeighborRegridder
from landlab_regrid.regrid import PatchRegridder


@pytest.mark.parametrize("src_at", ("node", "corner", "patch", "cell"))
@pytest.mark.parametrize("dst_at", ("node", "corner", "patch", "cell"))
@pytest.mark.parametrize(
    "regridder",
    (
        BilinearRegridder,
        partial(NearestNeighborRegridder, dtos=True),
        partial(NearestNeighborRegridder, dtos=False),
        PatchRegridder,
    ),
)
def test_regridders(src_at, dst_at, regridder):
    src = RasterModelGrid((10, 20))
    dst = HexModelGrid((10, 20))

    regrid = regridder(src, dst, src_at=src_at, dst_at=dst_at)

    z = np.sqrt(
        getattr(src, f"xy_of_{src_at}")[:, 0] ** 2
        + getattr(src, f"xy_of_{src_at}")[:, 1] ** 2
    )

    zi = regrid.regrid(z)
    assert zi.shape == (dst.number_of_elements(dst_at),)


@pytest.mark.parametrize("src_at", ("patch", "cell"))
@pytest.mark.parametrize("dst_at", ("patch", "cell"))
@pytest.mark.parametrize(
    "regridder",
    (
        partial(ConserveRegridder, order=1),
        partial(ConserveRegridder, order=2),
    ),
)
def test_conserve_regridders(src_at, dst_at, regridder):
    src = RasterModelGrid((10, 20))
    dst = HexModelGrid((10, 20))

    regrid = regridder(src, dst, src_at=src_at, dst_at=dst_at)

    z = np.sqrt(
        getattr(src, f"xy_of_{src_at}")[:, 0] ** 2
        + getattr(src, f"xy_of_{src_at}")[:, 1] ** 2
    )

    zi = regrid.regrid(z)
    assert zi.shape == (dst.number_of_elements(dst_at),)
