import pytest
from pytest_mock import MockerFixture
from rosbags.interfaces import Nodetype

from aflux.utils import AttrKey, ItemKey, IterKey, PickKey
from aflux.utils.ros import (
    ArrayNode,
    LeafNode,
    ListNode,
    StructNode,
    parse_field_value_into_node,
    parse_msgtype_into_node,
)


class TestNodes:
    class TestLeafNode:
        def test_str(self) -> None:
            assert str(LeafNode("float64", 0)) == "float64"
            assert str(LeafNode("string", 10)) == "string<=10"

        def test_transition_raises(self) -> None:
            node = LeafNode("float64")
            with pytest.raises(ValueError, match="Invalid"):
                node.transition(AttrKey("x"))
            with pytest.raises(ValueError, match="Invalid"):
                node.transition(PickKey(["x", "y"]))

    class TestStructNode:
        def test_str(self) -> None:
            node = StructNode("geometry_msgs/msg/Point", {"x": LeafNode("float64")})
            assert str(node) == "geometry_msgs/msg/Point"

        def test_transition_attr_success(self) -> None:
            leaf_x = LeafNode("float64")
            node = StructNode("geometry_msgs/msg/Point", {"x": leaf_x})
            assert node.transition(AttrKey("x")) is leaf_x

        def test_transition_attr_missing(self) -> None:
            node = StructNode("geometry_msgs/msg/Point", {"x": LeafNode("float64")})
            with pytest.raises(ValueError, match="attribute does not exist"):
                node.transition(AttrKey("y"))

        def test_transition_item_raises(self) -> None:
            node = StructNode("geometry_msgs/msg/Point", {"x": LeafNode("float64")})
            with pytest.raises(ValueError, match="Invalid"):
                node.transition(ItemKey(0))

        def test_transition_pick_success(self) -> None:
            leaf_x = LeafNode("float64")
            leaf_y = LeafNode("float64")
            node = StructNode("geometry_msgs/msg/Point", {"x": leaf_x, "y": leaf_y})
            assert node.transition(PickKey(["x", "y"])) is leaf_x

        def test_transition_pick_missing(self) -> None:
            node = StructNode("geometry_msgs/msg/Point", {"x": LeafNode("float64")})
            with pytest.raises(ValueError, match="attribute does not exist"):
                node.transition(PickKey(["x", "y"]))

        def test_transition_pick_heterogeneous(self) -> None:
            node = StructNode("test_msgs/msg/Mixed", {"a": LeafNode("float64"), "b": LeafNode("int32")})
            with pytest.raises(ValueError, match="same type"):
                node.transition(PickKey(["a", "b"]))

    class TestArrayNode:
        def test_str(self) -> None:
            node = ArrayNode(LeafNode("float64"), 3)
            assert str(node) == "float64[3]"

        def test_transition_item_success(self) -> None:
            item_node = LeafNode("float64")
            node = ArrayNode(item_node, 3)
            assert node.transition(ItemKey(0)) is item_node
            assert node.transition(IterKey()) is item_node

        def test_transition_attr_raises(self) -> None:
            node = ArrayNode(LeafNode("float64"), 3)
            with pytest.raises(ValueError, match="Invalid"):
                node.transition(AttrKey("x"))

    class TestListNode:
        def test_str(self) -> None:
            assert str(ListNode(LeafNode("float64"))) == "float64[]"
            assert str(ListNode(LeafNode("float64"), 10)) == "float64[<=10]"

        def test_transition_item_success(self) -> None:
            item_node = LeafNode("float64")
            node = ListNode(item_node)
            assert node.transition(ItemKey(0)) is item_node
            assert node.transition(IterKey()) is item_node

        def test_transition_attr_raises(self) -> None:
            node = ListNode(LeafNode("float64"))
            with pytest.raises(ValueError, match="list"):
                node.transition(AttrKey("x"))


class TestParsing:
    @pytest.fixture
    def mock_typestore(self, mocker: MockerFixture):
        typestore = mocker.MagicMock()

        def get_msgdef(msgtype: str):
            msgdef = mocker.MagicMock()
            if msgtype == "geometry_msgs/msg/Point":
                msgdef.fields = [
                    ("x", (Nodetype.BASE, ("float64", 0))),
                    ("y", (Nodetype.BASE, ("float64", 0))),
                    ("z", (Nodetype.BASE, ("float64", 0))),
                ]
            else:
                msgdef.fields = []
            return msgdef

        typestore.get_msgdef.side_effect = get_msgdef
        return typestore

    class TestParseFieldValue:
        def test_base_node(self, mock_typestore) -> None:
            node = parse_field_value_into_node(mock_typestore, (Nodetype.BASE, ("float64", 0)))
            assert isinstance(node, LeafNode)
            assert node.dtype == "float64"
            assert node.max_length == 0

        def test_name_node(self, mock_typestore) -> None:
            node = parse_field_value_into_node(mock_typestore, (Nodetype.NAME, "geometry_msgs/msg/Point"))
            assert isinstance(node, StructNode)
            assert node.dtype == "geometry_msgs/msg/Point"
            assert "x" in node.field_node_map

        def test_array_node(self, mock_typestore) -> None:
            node = parse_field_value_into_node(mock_typestore, (Nodetype.ARRAY, ((Nodetype.BASE, ("float64", 0)), 3)))
            assert isinstance(node, ArrayNode)
            assert node.item_node.dtype == "float64"
            assert node.size == 3

        def test_sequence_node(self, mock_typestore) -> None:
            node = parse_field_value_into_node(mock_typestore, (Nodetype.SEQUENCE, ((Nodetype.BASE, ("uint8", 0)), 0)))
            assert isinstance(node, ListNode)
            assert node.item_node.dtype == "uint8"
            assert node.max_size == 0

    class TestParseMsgtype:
        def test_struct_node_creation(self, mock_typestore) -> None:
            node = parse_msgtype_into_node(mock_typestore, "geometry_msgs/msg/Point")
            assert isinstance(node, StructNode)
            assert node.dtype == "geometry_msgs/msg/Point"
            assert set(node.field_node_map.keys()) == {"x", "y", "z"}
