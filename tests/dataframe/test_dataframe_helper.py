import polars as pl
import pytest

from aflux.utils import dataframe as dataframe_utils


class TestConvertDtypeIntoJsonSchema:
    def test_string(self) -> None:
        assert dataframe_utils.convert_dtype_into_json_schema(pl.String) == {"type": "string"}
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Categorical) == {"type": "string"}
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Enum(["a", "b"])) == {"type": "string"}

    def test_integer(self) -> None:
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Int32) == {"type": "integer"}
        assert dataframe_utils.convert_dtype_into_json_schema(pl.UInt64) == {"type": "integer"}

    def test_float(self) -> None:
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Float32) == {"type": "number"}
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Float64) == {"type": "number"}

    def test_boolean(self) -> None:
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Boolean) == {"type": "boolean"}

    def test_temporal(self) -> None:
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Date) == {"type": "string", "format": "date"}
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Datetime) == {"type": "string", "format": "date-time"}
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Time) == {"type": "string", "format": "time"}

    def test_list(self) -> None:
        dtype = pl.List(pl.Int32)
        assert dataframe_utils.convert_dtype_into_json_schema(dtype) == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_array(self) -> None:
        dtype = pl.Array(pl.Float64, 3)
        assert dataframe_utils.convert_dtype_into_json_schema(dtype) == {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "number"},
        }

        dtype = pl.Array(pl.Float32, (3, 10, 12))
        assert dataframe_utils.convert_dtype_into_json_schema(dtype) == {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "array",
                "minItems": 10,
                "maxItems": 10,
                "items": {
                    "type": "array",
                    "minItems": 12,
                    "maxItems": 12,
                    "items": {"type": "number"},
                },
            },
        }

    def test_struct(self) -> None:
        dtype = pl.Struct({"a": pl.String, "b": pl.Int64})
        assert dataframe_utils.convert_dtype_into_json_schema(dtype) == {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }

    def test_null_type(self) -> None:
        assert dataframe_utils.convert_dtype_into_json_schema(pl.Null) == {"type": "null"}

    def test_unknown_dtype(self) -> None:
        with pytest.raises(ValueError, match="Unknown dtype:"):
            dataframe_utils.convert_dtype_into_json_schema(pl.Object)


class TestFlattenStruct:
    def test_homogeneous_struct(self) -> None:
        df = pl.DataFrame(
            {"data": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]},
            schema={"data": pl.Struct({"x": pl.Float64(), "y": pl.Float64()})},
        )
        expr = dataframe_utils.flatten_struct(pl.col("data"), df.schema["data"]).alias("data")
        result = df.select(expr)
        assert result.dtypes == [pl.Array(pl.Float64(), 2)]
        assert result["data"].to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_nested_list_of_structs(self) -> None:
        df = pl.DataFrame(
            {"data": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
            schema={"data": pl.List(pl.Struct({"x": pl.Float64(), "y": pl.Float64()}))},
        )
        expr = dataframe_utils.flatten_struct(pl.col("data"), df.schema["data"]).alias("data")
        result = df.select(expr)
        assert result.dtypes == [pl.List(pl.Array(pl.Float64(), 2))]
        assert result["data"].to_list() == [[[1.0, 2.0], [3.0, 4.0]]]

    def test_base_case_primitive_dtype(self) -> None:
        df = pl.DataFrame({"data": [1.0, 2.0]})
        expr = dataframe_utils.flatten_struct(pl.col("data"), pl.Float64()).alias("data")
        result = df.select(expr)
        assert result.dtypes == [pl.Float64()]
        assert result["data"].to_list() == [1.0, 2.0]
