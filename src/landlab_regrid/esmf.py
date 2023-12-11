import esmpy
import numpy as np
from landlab.graph.quantity.ext.of_element import count_of_children_at_parent


def create_grid(grid, point="node"):
    if point not in ("node", "corner"):
        raise ValueError(
            f"'point' keyword must be one of 'node' or 'corner' (got {point})"
        )

    if point == "node":
        rows_cols = grid.shape
    else:
        rows_cols = (grid.shape[0] - 1, grid.shape[1] - 1)

    esmpy_grid = esmpy.Grid(
        np.asarray(rows_cols),
        staggerloc=[esmpy.StaggerLoc.CENTER],
        coord_sys=esmpy.CoordSys.CART,
        coord_typekind=esmpy.TypeKind.R8,
    )

    x = esmpy_grid.get_coords(0)
    x[:] = getattr(grid, f"x_of_{point}").reshape(rows_cols)
    y = esmpy_grid.get_coords(1)
    y[:] = getattr(grid, f"y_of_{point}").reshape(rows_cols)

    return esmpy_grid


def create_mesh(grid, point="node"):
    if point not in ("node", "corner"):
        raise ValueError(
            f"'point' keyword must be one of 'node' or 'corner' (got {point})"
        )
    polygon = "patch" if point == "node" else "cell"

    mesh = esmpy.Mesh(parametric_dim=2, spatial_dim=2)

    mesh.add_nodes(
        getattr(grid, f"number_of_{point}s"),
        np.arange(getattr(grid, f"number_of_{point}s")),
        getattr(grid, f"xy_of_{point}s").reshape(-1),
        grid.zeros(at=point),
    )
    points_per_polygon = grid.empty(at=polygon)
    count_of_children_at_parent(
        getattr(grid, f"{point}s_at_{polygon}", points_per_polygon)
    )
    points_at_polygon = getattr(grid, f"{point}s_at_{polygon}").reshape(-1)

    mesh.add_elements(
        getattr(grid, f"number_of_{polygon}s"),
        np.arange(getattr(grid, f"number_of_{polygon}s")),
        points_per_polygon,
        points_at_polygon[points_at_polygon != -1],
    )

    return mesh


def create_loc_stream(xy_of_points):
    xy_of_points = np.asarray(xy_of_points)

    loc_stream = esmpy.LocStream(len(xy_of_points))

    loc_stream["x_of_point"] = xy_of_points[:, 0]
    loc_stream["y_of_point"] = xy_of_points[:, 1]
    loc_stream["ESMF:X"] = xy_of_points[:, 0]
    loc_stream["ESMF:Y"] = xy_of_points[:, 1]

    return loc_stream
