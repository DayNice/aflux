from typing import Any

import polars as pl
from polars.datatypes import DataType, DataTypeClass


def convert_dtype_into_json_schema(
    dtype: DataType | DataTypeClass,
    nullable: bool = False,
) -> dict[str, Any]:
    match dtype:
        case pl.Boolean:
            return {"type": "boolean" if not nullable else ["boolean", "null"]}
        case pl.Int8 | pl.Int16 | pl.Int32 | pl.Int64 | pl.UInt8 | pl.UInt16 | pl.UInt32 | pl.UInt64:
            return {"type": "integer" if not nullable else ["integer", "null"]}
        case pl.Float32 | pl.Float64:
            return {"type": "number" if not nullable else ["number", "null"]}
        case pl.String | pl.Categorical | pl.Enum | pl.Date | pl.Datetime | pl.Time:
            return {"type": "string" if not nullable else ["string", "null"]}
        case pl.List() | pl.Array():
            return {
                "type": "array" if not nullable else ["array", "null"],
                "items": convert_dtype_into_json_schema(dtype.inner, nullable),
            }
        case pl.Struct():
            properties: dict[str, Any] = {}
            for field in dtype.fields:
                properties[field.name] = convert_dtype_into_json_schema(field.dtype, nullable)
            return {
                "type": "object" if not nullable else ["object", "null"],
                "properties": properties,
            }
        case pl.Null:
            return {"type": "null"}
        case _:
            msg = f"Unknown dtype: {dtype}"
            raise ValueError(msg)


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


def flatten_struct(
    root_expr: pl.Expr,
    root_dtype: DataType | DataTypeClass,
) -> pl.Expr:
    match root_dtype:
        case pl.Struct():
            field_exprs: list[pl.Expr] = []
            for field in root_dtype.fields:
                field_expr = flatten_struct(root_expr.struct.field(field.name), field.dtype)
                # workaround to avoid `pl.concat_list` flattening list of lists
                field_expr = field_expr.repeat_by(1)
                field_exprs.append(field_expr)
            return pl.concat_list(field_exprs).list.to_array(len(root_dtype.fields))
        case pl.List():
            item_expr = flatten_struct(pl.element(), root_dtype.inner)
            return root_expr.list.eval(item_expr)
        case pl.Array():
            item_expr = flatten_struct(pl.element(), root_dtype.inner)
            return root_expr.arr.eval(item_expr)
        case _:
            return root_expr
