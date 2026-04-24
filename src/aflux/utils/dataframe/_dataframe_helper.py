from typing import Any

import polars as pl
from polars.datatypes import DataType, DataTypeClass


def convert_dtype_into_json_schema(dtype: DataType | DataTypeClass) -> dict[str, Any]:
    match dtype:
        case pl.String | pl.Categorical | pl.Enum:
            return {"type": "string"}
        case pl.Int8 | pl.Int16 | pl.Int32 | pl.Int64 | pl.UInt8 | pl.UInt16 | pl.UInt32 | pl.UInt64:
            return {"type": "integer"}
        case pl.Float32 | pl.Float64:
            return {"type": "number"}
        case pl.Boolean:
            return {"type": "boolean"}
        case pl.Date:
            return {"type": "string", "format": "date"}
        case pl.Datetime:
            return {"type": "string", "format": "date-time"}
        case pl.Time:
            return {"type": "string", "format": "time"}
        case pl.List():
            return {"type": "array", "items": convert_dtype_into_json_schema(dtype.inner)}
        case pl.Array():
            return {
                "type": "array",
                "minItems": dtype.size,
                "maxItems": dtype.size,
                "items": convert_dtype_into_json_schema(dtype.inner),
            }
        case pl.Struct():
            return {
                "type": "object",
                "properties": {field.name: convert_dtype_into_json_schema(field.dtype) for field in dtype.fields},
            }
        case pl.Null:
            return {"type": "null"}
        case _:
            msg = f"Unknown dtype: {dtype}"
            raise ValueError(msg)


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
