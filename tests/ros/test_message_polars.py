import polars as pl
import pytest

from aflux.utils import Key
from aflux.utils.ros import (
    convert_message_node_into_polars_dtype,
    convert_message_node_with_key_into_polars_dtype,
)
from aflux.utils.ros._message_node import (
    ArrayNode,
    LeafNode,
    ListNode,
    StructNode,
)


class TestConvertMessageNodeIntoPolarsDtype:
    def test_leaf_node(self) -> None:
        leaf = LeafNode("float64")
        assert convert_message_node_into_polars_dtype(leaf) == pl.Float64()

    def test_struct_node(self) -> None:
        leaf = LeafNode("float64")
        struct = StructNode("dummy", {"x": leaf, "y": leaf})
        assert convert_message_node_into_polars_dtype(struct) == pl.Struct({"x": pl.Float64(), "y": pl.Float64()})

    def test_array_node(self) -> None:
        leaf = LeafNode("float64")
        array = ArrayNode(leaf, 5)
        assert convert_message_node_into_polars_dtype(array) == pl.Array(pl.Float64(), 5)

    def test_list_node(self) -> None:
        leaf = LeafNode("float64")
        lst = ListNode(leaf)
        assert convert_message_node_into_polars_dtype(lst) == pl.List(pl.Float64())


class TestConvertMessageNodeWithKeyIntoPolarsDtype:
    @pytest.fixture
    def point_node(self) -> StructNode:
        return StructNode(
            "geometry_msgs/msg/Point",
            {
                "x": LeafNode("float64"),
                "y": LeafNode("float64"),
                "z": LeafNode("float64"),
            },
        )

    @pytest.fixture
    def point_cloud_node(self, point_node: StructNode) -> StructNode:
        return StructNode("sensor_msgs/msg/PointCloud", {"points": ListNode(point_node)})

    @pytest.fixture
    def polygon_node(self, point_node: StructNode) -> StructNode:
        return StructNode("custom_msgs/msg/Polygon", {"points": ArrayNode(point_node, 5)})

    def test_basic_attr(self, point_node: StructNode) -> None:
        dtype = convert_message_node_with_key_into_polars_dtype(point_node, Key("x"))
        assert dtype == pl.Float64()

    def test_pick_key(self, point_node: StructNode) -> None:
        dtype = convert_message_node_with_key_into_polars_dtype(point_node, Key("{x,y}"))
        assert dtype == pl.Array(pl.Float64(), 2)

    def test_list_iter(self, point_cloud_node: StructNode) -> None:
        dtype = convert_message_node_with_key_into_polars_dtype(point_cloud_node, Key("points[].x"))
        assert dtype == pl.List(pl.Float64())

    def test_list_iter_with_pick(self, point_cloud_node: StructNode) -> None:
        dtype = convert_message_node_with_key_into_polars_dtype(point_cloud_node, Key("points[].{x,y}"))
        assert dtype == pl.List(pl.Array(pl.Float64(), 2))

    def test_array_iter(self, polygon_node: StructNode) -> None:
        dtype = convert_message_node_with_key_into_polars_dtype(polygon_node, Key("points[].x"))
        assert dtype == pl.Array(pl.Float64(), 5)

    def test_array_iter_with_pick(self, polygon_node: StructNode) -> None:
        dtype = convert_message_node_with_key_into_polars_dtype(polygon_node, Key("points[].{x,y}"))
        assert dtype == pl.Array(pl.Array(pl.Float64(), 2), 5)

    def test_array_item(self, polygon_node: StructNode) -> None:
        dtype = convert_message_node_with_key_into_polars_dtype(polygon_node, Key("points[0].x"))
        assert dtype == pl.Float64()
