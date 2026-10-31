"""坐标、单应、椭圆与方向变换的固定预期。"""

from __future__ import annotations

import pytest

from behaviourlab.geometry import (
    Viewport,
    ellipse_contains,
    ellipse_normalized,
    point_in_polygon,
    quad_mapper,
    rotate_point,
    transform_geometry,
)

TOL = 1e-6


def close(actual, expected):
    assert abs(float(actual) - float(expected)) <= TOL, (actual, expected)


def test_viewport_maps_page_pointer_to_source_pixels():
    view = Viewport(640, 480, 934, 531, dpr=1, left=20, top=164)
    close(view.scale, 1.10625)
    close(view.offset[0], 113.0)
    close(view.offset[1], 0.0)
    assert view.bitmap_size == (934, 531)

    point = view.to_source(243.625, 296.75)
    assert point is not None
    close(point[0], 100.0)
    close(point[1], 120.0)

    retina = Viewport(640, 480, 934, 531, dpr=2, left=20, top=164)
    close(retina.scale, 1.10625)
    same = retina.to_source(243.625, 296.75)
    close(same[0], 100.0)
    close(same[1], 120.0)
    assert retina.bitmap_size == (1868, 1062)


def test_pointer_in_letterbox_is_rejected():
    view = Viewport(640, 480, 934, 531, dpr=1, left=20, top=164)
    assert view.to_source(20.0, 296.75) is None
    assert view.to_source(500.0, 160.0) is None


def test_quad_maps_to_physical_size():
    quad = [(20, 10), (180, 10), (180, 90), (20, 90)]
    mapper = quad_mapper(quad, 80, 40)
    x, y = mapper((100, 50))
    close(x, 40.0)
    close(y, 20.0)
    assert point_in_polygon((20, 50), quad) is True
    assert point_in_polygon((5, 50), quad) is False


def test_trapezoid_uses_homography_not_axis_scaling():
    quad = [(0, 0), (100, 0), (80, 100), (20, 100)]
    mapper = quad_mapper(quad, 100, 100)
    for corner, expected in zip(
        quad, [(0, 0), (100, 0), (100, 100), (0, 100)]
    ):
        point = mapper(corner)
        close(point[0], expected[0])
        close(point[1], expected[1])
    # 对角线交点保持中心：投影映射保交比，不按轴比例近似。
    center = mapper((50, 62.5))
    close(center[0], 50.0)
    close(center[1], 50.0)
    # 对称梯形上同一高度左右点应对称。
    left = mapper((10, 50))
    right = mapper((90, 50))
    close(left[0] + right[0], 100.0)
    close(left[1], right[1])


def test_quad_rejects_crossed_corners():
    crossed = [(0, 0), (100, 100), (100, 0), (0, 100)]
    with pytest.raises(Exception):
        quad_mapper(crossed, 10, 10)


def test_ellipse_normalized_and_boundary():
    u, v = ellipse_normalized((130, 50), (100, 50), (60, 30), 0)
    close(u, 0.5)
    close(v, 0.0)
    u, v = ellipse_normalized((100, 80), (100, 50), (60, 30), 90)
    close(u, 0.5)
    close(v, 0.0)
    assert ellipse_contains((160, 50), (100, 50), (60, 30), 0) is True
    assert ellipse_contains((161, 50), (100, 50), (60, 30), 0) is False


def test_orientation_round_trip_and_ellipse_angle():
    point = (100, 120)
    for orientation in (90, 180, 270):
        moved = rotate_point(point, 640, 480, 0, orientation)
        back = rotate_point(moved, 640, 480, orientation, 0)
        close(back[0], point[0])
        close(back[1], point[1])

    moved = rotate_point((100, 120), 640, 480, 0, 90)
    close(moved[0], 359.0)
    close(moved[1], 100.0)

    geometry = {
        "arena": {"type": "polygon", "points": [[10, 20], [30, 20], [30, 40]]},
        "platform": {"type": "ellipse", "center": [100, 120], "radii": [12, 8],
                     "angle": 10.0},
        "holes": {"type": "holes", "holes": [{"id": "1", "center": [50, 60]}]},
    }
    rotated = transform_geometry(geometry, 640, 480, 0, 90)
    close(rotated["arena"]["points"][0][0], 459.0)
    close(rotated["arena"]["points"][0][1], 10.0)
    close(rotated["platform"]["center"][0], 359.0)
    assert rotated["platform"]["radii"] == [12.0, 8.0]
    close(rotated["platform"]["angle"], 100.0)
    close(rotated["holes"]["holes"][0]["center"][0], 419.0)

    restored = transform_geometry(rotated, 640, 480, 90, 0)
    close(restored["arena"]["points"][0][0], 10.0)
    close(restored["arena"]["points"][0][1], 20.0)
    assert restored["platform"]["radii"] == [12.0, 8.0]
    close(restored["platform"]["angle"], 10.0)
