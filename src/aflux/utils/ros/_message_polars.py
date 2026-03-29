from typing import Literal, assert_never

import polars as pl
from rosbags.interfaces.typing import Basename

from aflux.utils import IterKey, Key, PickKey

from ._message_node import (
    ArrayNode,
    LeafNode,
    ListNode,
    MessageNode,
    StructNode,
)


def _basename_to_polars_dtype(basename: Basename) -> pl.DataType:
    mapping: dict[Basename, pl.DataType] = {
        "bool": pl.Boolean(),
        "byte": pl.UInt8(),
        "char": pl.Int8(),
        "int8": pl.Int8(),
        "int16": pl.Int16(),
        "int32": pl.Int32(),
        "int64": pl.Int64(),
        "uint8": pl.UInt8(),
        "uint16": pl.UInt16(),
        "uint32": pl.UInt32(),
        "uint64": pl.UInt64(),
        "float32": pl.Float32(),
        "float64": pl.Float64(),
        # "float128": pl.Float64(),  # polars doesn't have an equivalent dtype
        "string": pl.String(),
    }

    if basename not in mapping:
        msg = f"Unsupported ROS basename: {basename}"
        raise ValueError(msg)
    return mapping[basename]


def convert_message_node_into_polars_dtype(node: MessageNode) -> pl.DataType:
    match node:
        case LeafNode(dtype):
            return _basename_to_polars_dtype(dtype)
        case StructNode(_, field_node_map):
            field_dtype_map = {
                field_name: convert_message_node_into_polars_dtype(field_node)
                for field_name, field_node in field_node_map.items()
            }
            return pl.Struct(field_dtype_map)
        case ArrayNode(item_node):
            inner_dtype = convert_message_node_into_polars_dtype(item_node)
            return pl.Array(inner_dtype, shape=(node.size,))
        case ListNode(item_node):
            inner_dtype = convert_message_node_into_polars_dtype(item_node)
            return pl.List(inner_dtype)
        case _:
            assert_never(node)


def convert_message_node_with_key_into_polars_dtype(
    node: MessageNode,
    key: str | Key,
) -> pl.DataType:
    key = Key(key)
    wrapper_infos: list[tuple[Literal["array", "list"], int | None]] = []
    for part in key.parts:
        match part, node:
            case IterKey(), ArrayNode(_, size):
                wrapper_infos.append(("array", size))
            case IterKey(), ListNode():
                wrapper_infos.append(("list", None))
            case PickKey(names), StructNode():
                wrapper_infos.append(("array", len(names)))
        node = node.transition(part)

    dtype = convert_message_node_into_polars_dtype(node)
    for wrapper, size in reversed(wrapper_infos):
        match wrapper:
            case "array":
                assert size is not None
                dtype = pl.Array(dtype, size)
            case "list":
                dtype = pl.List(dtype)
            case _:
                assert_never(wrapper)
    return dtype
