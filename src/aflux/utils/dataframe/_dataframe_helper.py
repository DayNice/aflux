import polars as pl
from polars.datatypes import DataType, DataTypeClass


def flatten_struct(
    root_expr: pl.Expr,
    root_dtype: DataType | DataTypeClass,
) -> pl.Expr:
    match root_dtype:
        case pl.Struct():
            field_dtype_set = {el.dtype for el in root_dtype.fields}
            if len(field_dtype_set) != 1:
                msg = f"Struct with multiple field dtypes cannot be flattened: {field_dtype_set}"
                raise ValueError(msg)

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
