import polars as pl

from aflux.utils.ros._message_node import (
    ArrayNode,
    LeafNode,
    ListNode,
    StructNode,
)


class TestConvertMessageNodeIntoPolarsDtype:
    def test_leaf_node(self) -> None:
        leaf_node = LeafNode("float64")
        assert leaf_node.to_polars_dtype() == pl.Float64()

    def test_struct_node(self) -> None:
        leaf_node = LeafNode("float64")
        struct_node = StructNode("dummy", {"x": leaf_node, "y": leaf_node})
        assert struct_node.to_polars_dtype() == pl.Struct({"x": pl.Float64(), "y": pl.Float64()})

    def test_array_node(self) -> None:
        leaf_node = LeafNode("float64")
        array_node = ArrayNode(leaf_node, 5)
        assert array_node.to_polars_dtype() == pl.Array(pl.Float64(), 5)

    def test_list_node(self) -> None:
        leaf_node = LeafNode("float64")
        list_node = ListNode(leaf_node)
        assert list_node.to_polars_dtype() == pl.List(pl.Float64())
