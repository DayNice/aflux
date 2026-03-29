from typing import Union

from pydantic import BaseModel, ConfigDict
from rosbags.interfaces import Nodetype, Typestore
from rosbags.interfaces.typing import Basename, FieldDesc

from aflux.utils import AttrKey, ItemKey, Key, SpreadKey


class BaseNode(BaseModel):
    model_config = ConfigDict(frozen=True)


class LeafNode(BaseNode):
    __match_args__ = ("name", "upper_bound")

    name: Basename
    upper_bound: int


class StructNode(BaseNode):
    __match_args__ = ("name",)

    name: str


class ArrayNode(BaseNode):
    __match_args__ = ("item_node", "size")

    item_node: LeafNode | StructNode
    size: int


class ListNode(BaseNode):
    __match_args__ = ("item_node",)

    item_node: LeafNode | StructNode


MessageNode = Union[LeafNode, StructNode, ArrayNode, ListNode]


def parse_field_value_into_node(field_value: FieldDesc):
    match field_value:
        case (Nodetype.BASE, (name, upper_bound)):
            return LeafNode(name=name, upper_bound=upper_bound)
        case (Nodetype.NAME, name):
            return StructNode(name=name)
        case (Nodetype.ARRAY, ((Nodetype.BASE, (name, upper_bound)), size)):
            item_node = LeafNode(name=name, upper_bound=upper_bound)
            return ArrayNode(item_node=item_node, size=size)
        case (Nodetype.ARRAY, ((Nodetype.NAME, name), size)):
            item_node = StructNode(name=name)
            return ArrayNode(item_node=item_node, size=size)
        case (Nodetype.SEQUENCE, ((Nodetype.BASE, (name, upper_bound)), _)):
            item_node = LeafNode(name=name, upper_bound=upper_bound)
            return ListNode(item_node=item_node)
        case (Nodetype.SEQUENCE, ((Nodetype.NAME, name), _)):
            item_node = StructNode(name=name)
            return ListNode(item_node=item_node)
        case _:
            raise ValueError(f"Unexpected field value: {field_value!r}")


def transition_node(typestore: Typestore, node: MessageNode, key: AttrKey | ItemKey | SpreadKey):
    match node, key:
        case LeafNode(), _:
            raise ValueError("Invalid attribute or item access against a leaf.")
        case StructNode(msgtype), AttrKey(field_name):
            msgdef = typestore.get_msgdef(msgtype)
            for field_info in msgdef.fields:
                if field_info[0] == field_name:
                    break
            else:
                msg = f"Requested attribute does not exist: {msgtype!r} {field_name!r}"
                raise ValueError(msg)
            return parse_field_value_into_node(field_info[1])
        case StructNode(), ItemKey() | SpreadKey():
            raise ValueError("Invalid item access against a struct.")
        case ArrayNode() | ListNode(), AttrKey():
            raise ValueError("Invalid attribute access against an array or list.")
        case ArrayNode() | ListNode() as node, ItemKey() | SpreadKey():
            return node.item_node
        case _:
            raise ValueError(f"Unknown node and key combination: {node!r} {key!r}")


def validate_message_field_getter(
    typestore: Typestore,
    msgtype: str,
    key: str,
) -> Key:
    chain_key = Key.parse(key)
    node: MessageNode = StructNode(name=msgtype)
    for part in chain_key.parts:
        node = transition_node(typestore, node, part)
    return chain_key
