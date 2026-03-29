from abc import ABCMeta, abstractmethod
from typing import Any, Union, override

import numpy as np
from rosbags.interfaces import Nodetype, Typestore
from rosbags.interfaces.typing import Basename, FieldDesc

from aflux.utils import AttrKey, ItemKey, IterKey, Key, PickKey


class BaseNode(metaclass=ABCMeta):
    dtype: str

    @abstractmethod
    def __str__(self) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)!r})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        raise NotImplementedError

    @abstractmethod
    def transition(self, key: AttrKey | ItemKey | IterKey | PickKey) -> "BaseNode": ...

    @abstractmethod
    def dump_message(self, message: Any) -> Any: ...

    def dump_message_with_key(self, message: Any, key: str | Key) -> Any:
        key = Key(key)
        node = self
        for part in key.parts:
            node = node.transition(part)
        return node.dump_message(key(message))


class LeafNode(BaseNode):
    __match_args__ = ("dtype", "max_length")

    def __init__(self, dtype: Basename, max_length: int = 0) -> None:
        self._dtype = dtype
        self._max_length = max_length

    @property
    def dtype(self) -> Basename:
        return self._dtype

    @property
    def max_length(self) -> int:
        return self._max_length

    @override
    def __str__(self) -> str:
        if self.max_length != 0:
            return f"{self.dtype}<={self.max_length}"
        return f"{self.dtype}"

    @override
    def transition(self, key: AttrKey | ItemKey | IterKey | PickKey) -> "MessageNode":
        msg = f"Invalid attribute or item access against a leaf: {str(self)!r} {str(key)!r}"
        raise ValueError(msg)

    @override
    def dump_message(self, message: Any) -> Any:
        return message


class StructNode(BaseNode):
    __match_args__ = ("dtype", "field_node_map")

    def __init__(self, dtype: str, field_node_map: "dict[str, MessageNode]") -> None:
        self._dtype = dtype
        self._field_node_map = field_node_map

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def field_node_map(self) -> "dict[str, MessageNode]":
        return self._field_node_map

    @override
    def __str__(self) -> str:
        return self.dtype

    @override
    def transition(self, key: AttrKey | ItemKey | IterKey | PickKey) -> "MessageNode":
        if isinstance(key, AttrKey):
            field_node = self.field_node_map.get(key.name)
            if field_node is None:
                msg = f"Requested attribute does not exist: {str(self)!r} {str(key)!r}"
                raise ValueError(msg)
            return field_node

        if isinstance(key, PickKey):
            first_node = self.field_node_map.get(key.names[0])
            if first_node is None:
                msg = f"Requested attribute does not exist: {str(self)!r} {key.names[0]!r} in {str(key)!r}"
                raise ValueError(msg)

            for name in key.names[1:]:
                field_node = self.field_node_map.get(name)
                if field_node is None:
                    msg = f"Requested attribute does not exist: {str(self)!r} {name!r} in {str(key)!r}"
                    raise ValueError(msg)
                if field_node != first_node:
                    msg = (
                        "PickKey requires all requested attributes to have the same type."
                        f"Mismatch in {str(self)!r}: {str(first_node)!r} vs {str(field_node)!r}",
                    )
                    raise ValueError(msg)
            return first_node

        msg = f"Invalid item access against a struct: {str(self)!r} {str(key)!r}"
        raise ValueError(msg)

    @override
    def dump_message(self, message: Any) -> dict[str, Any]:
        return {
            field_name: field_node.dump_message(getattr(message, field_name))
            for field_name, field_node in self.field_node_map.items()
        }


class ArrayNode(BaseNode):
    __match_args__ = ("item_node", "size")

    def __init__(self, item_node: LeafNode | StructNode, size: int) -> None:
        self._item_node = item_node
        self._size = size

    @property
    def dtype(self) -> str:
        return f"{self.item_node.dtype}[{self.size}]"

    @property
    def item_node(self) -> LeafNode | StructNode:
        return self._item_node

    @property
    def size(self) -> int:
        return self._size

    @override
    def __str__(self) -> str:
        return self.dtype

    @override
    def transition(self, key: AttrKey | ItemKey | IterKey | PickKey) -> "MessageNode":
        if isinstance(key, (ItemKey, IterKey)):
            return self.item_node
        msg = f"Invalid attribute access against an array: {str(self)!r} {str(key)!r}"
        raise ValueError(msg)

    @override
    def dump_message(self, message: Any) -> np.ndarray | list[Any]:
        if isinstance(message, np.ndarray):
            return message
        return [self.item_node.dump_message(item) for item in message]


class ListNode(BaseNode):
    __match_args__ = ("item_node", "max_size")

    def __init__(self, item_node: LeafNode | StructNode, max_size: int = 0) -> None:
        self._item_node = item_node
        self._max_size = max_size

    @property
    def dtype(self) -> str:
        return f"{self.item_node.dtype}[]"

    @property
    def item_node(self) -> LeafNode | StructNode:
        return self._item_node

    @property
    def max_size(self) -> int:
        return self._max_size

    @override
    def __str__(self) -> str:
        if self.max_size != 0:
            return f"{self.item_node.dtype}[<={self.max_size}]"
        return f"{self.item_node.dtype}[]"

    @override
    def transition(self, key: AttrKey | ItemKey | IterKey | PickKey) -> "MessageNode":
        if isinstance(key, (ItemKey, IterKey)):
            return self.item_node
        msg = f"Invalid attribute access against a list: {str(self)!r} {str(key)!r}"
        raise ValueError(msg)

    @override
    def dump_message(self, message: Any) -> list[Any]:
        if isinstance(message, np.ndarray):
            return message
        return [self.item_node.dump_message(item) for item in message]


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
        case (Nodetype.BASE, (dtype, max_length)):
            return LeafNode(dtype, max_length)
        case (Nodetype.NAME, dtype):
            return parse_msgtype_into_node(typestore, dtype)
        case (Nodetype.ARRAY, ((Nodetype.BASE, (dtype, max_length)), size)):
            item_node = LeafNode(dtype, max_length)
            return ArrayNode(item_node, size)
        case (Nodetype.ARRAY, ((Nodetype.NAME, dtype), size)):
            item_node = parse_msgtype_into_node(typestore, dtype)
            return ArrayNode(item_node, size)
        case (Nodetype.SEQUENCE, ((Nodetype.BASE, (dtype, max_length)), max_size)):
            item_node = LeafNode(dtype, max_length)
            return ListNode(item_node, max_size)
        case (Nodetype.SEQUENCE, ((Nodetype.NAME, dtype), max_size)):
            item_node = parse_msgtype_into_node(typestore, dtype)
            return ListNode(item_node, max_size)
        case _:
            raise ValueError(f"Unexpected field value: {field_value!r}")


def validate_message_field_getter(
    typestore: Typestore,
    msgtype: str,
    key: str | Key,
) -> Key:
    key = Key(key)
    node: MessageNode = parse_msgtype_into_node(typestore, msgtype)
    for part in key.parts:
        node = node.transition(part)
    return key
