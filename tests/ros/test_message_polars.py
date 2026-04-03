import polars as pl

from aflux.utils.ros import (
    convert_message_node_into_polars_dtype,
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
