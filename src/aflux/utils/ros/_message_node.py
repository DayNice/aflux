from typing import Union

from pydantic import BaseModel, ConfigDict
from rosbags.interfaces import Nodetype, Typestore
from rosbags.interfaces.typing import Basename, FieldDesc

from aflux.utils import AttrKey, ItemKey, IterKey, Key


class BaseNode(BaseModel):
    model_config = ConfigDict(frozen=True)


class LeafNode(BaseNode):
    __match_args__ = ("dtype", "upper_bound")

    dtype: Basename
    upper_bound: int


class StructNode(BaseNode):
    __match_args__ = ("dtype", "field_node_map")

    dtype: str
    field_node_map: "dict[str, MessageNode]"


class ArrayNode(BaseNode):
    __match_args__ = ("item_node", "size")

    item_node: LeafNode | StructNode
    size: int

    @property
    def dtype(self) -> str:
        return f"{self.item_node.dtype}[{self.size}]"


class ListNode(BaseNode):
    __match_args__ = ("item_node",)

    item_node: LeafNode | StructNode

    @property
    def dtype(self) -> str:
        return f"{self.item_node.dtype}[]"


type MessageNode = Union[LeafNode, StructNode, ArrayNode, ListNode]


def parse_msgtype_into_node(typestore: Typestore, msgtype: str) -> StructNode:
    msgdef = typestore.get_msgdef(msgtype)
    field_node_map: dict[str, MessageNode] = {}
    for field_name, field_value in msgdef.fields:
        field_node = parse_field_value_into_node(typestore, field_value)
        field_node_map[field_name] = field_node
    return StructNode(dtype=msgtype, field_node_map=field_node_map)


def parse_field_value_into_node(typestore: Typestore, field_value: FieldDesc) -> MessageNode:
    match field_value:
        case (Nodetype.BASE, (dtype, upper_bound)):
            return LeafNode(dtype=dtype, upper_bound=upper_bound)
        case (Nodetype.NAME, dtype):
            return parse_msgtype_into_node(typestore, dtype)
        case (Nodetype.ARRAY, ((Nodetype.BASE, (dtype, upper_bound)), size)):
            item_node = LeafNode(dtype=dtype, upper_bound=upper_bound)
            return ArrayNode(item_node=item_node, size=size)
        case (Nodetype.ARRAY, ((Nodetype.NAME, dtype), size)):
            item_node = parse_msgtype_into_node(typestore, dtype)
            return ArrayNode(item_node=item_node, size=size)
        case (Nodetype.SEQUENCE, ((Nodetype.BASE, (dtype, upper_bound)), _)):
            item_node = LeafNode(dtype=dtype, upper_bound=upper_bound)
            return ListNode(item_node=item_node)
        case (Nodetype.SEQUENCE, ((Nodetype.NAME, dtype), _)):
            item_node = parse_msgtype_into_node(typestore, dtype)
            return ListNode(item_node=item_node)
        case _:
            raise ValueError(f"Unexpected field value: {field_value!r}")


def transition_node(node: MessageNode, key: AttrKey | ItemKey | IterKey):
    match node, key:
        case LeafNode(dtype), _:
            msg = f"Invalid attribute or item access against a leaf: {dtype!r}"
            raise ValueError(msg)
        case StructNode(dtype, field_node_map), AttrKey(field_name):
            field_node = field_node_map.get(field_name)
            if field_node is None:
                msg = f"Requested attribute does not exist: {dtype!r} {field_name!r}"
                raise ValueError(msg)
            return field_node
        case StructNode(), ItemKey() | IterKey():
            raise ValueError("Invalid item access against a struct.")
        case ArrayNode() | ListNode(), AttrKey():
            raise ValueError("Invalid attribute access against an array or list.")
        case ArrayNode() | ListNode() as node, ItemKey() | IterKey():
            return node.item_node
        case _:
            raise ValueError(f"Unknown node and key combination: {node!r} {key!r}")


def validate_message_field_getter(
    typestore: Typestore,
    msgtype: str,
    key: str | Key,
) -> Key:
    key = Key(key)
    node: MessageNode = parse_msgtype_into_node(typestore, msgtype)
    for part in key.parts:
        node = transition_node(node, part)
    return key
