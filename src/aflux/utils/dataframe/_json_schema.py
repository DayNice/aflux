from typing import Any, Literal

import polars as pl
from polars.datatypes import DataType, DataTypeClass


def convert_dtype_into_json_dtype(
    dtype: DataType | DataTypeClass,
) -> Literal[
    "boolean",
    "integer",
    "number",
    "string",
    "array",
    "object",
    "null",
]:
    match dtype:
        case pl.Boolean:
            return "boolean"
        case pl.Int8 | pl.Int16 | pl.Int32 | pl.Int64 | pl.UInt8 | pl.UInt16 | pl.UInt32 | pl.UInt64:
            return "integer"
        case pl.Float32 | pl.Float64:
            return "number"
        case pl.String | pl.Categorical | pl.Enum | pl.Date | pl.Datetime | pl.Time:
            return "string"
        case pl.List() | pl.Array():
            return "array"
        case pl.Struct():
            return "object"
        case pl.Null:
            return "null"
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def convert_dtype_into_json_schema(
    dtype: DataType | DataTypeClass,
    nullable: bool = False,
) -> dict[str, Any]:
    json_dtype = convert_dtype_into_json_dtype(dtype)
    if not dtype.is_nested():
        if json_dtype == "null" or not nullable:
            return {"type": json_dtype}
        return {"type": [json_dtype, "null"]}

    match dtype:
        case pl.List() | pl.Array():
            return {
                "type": json_dtype if not nullable else [json_dtype, "null"],
                "items": convert_dtype_into_json_schema(dtype.inner, nullable),
            }
        case pl.Struct():
            properties: dict[str, Any] = {}
            for field in dtype.fields:
                properties[field.name] = convert_dtype_into_json_schema(field.dtype, nullable)
            return {
                "type": json_dtype if not nullable else [json_dtype, "null"],
                "properties": properties,
            }
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def convert_schema_into_json_schema(
    schema: pl.Schema,
    nullable: bool = False,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, dtype in schema.items():
        properties[name] = convert_dtype_into_json_schema(dtype, nullable)
    return {
        "type": "object" if not nullable else ["object", "null"],
        "properties": properties,
    }
