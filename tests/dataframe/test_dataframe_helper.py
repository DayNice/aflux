import polars as pl

from aflux.utils import dataframe as dataframe_utils


class TestConvertDtypeIntoJsonSchema:
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
            "items": {"type": "number"},
        }

        dtype = pl.Array(pl.Float32, (3, 10, 12))
        assert dataframe_utils.convert_dtype_into_json_schema(dtype) == {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "array",
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


class TestConvertSchemaIntoJsonSchema:
    def test_primitive(self) -> None:
        schema = pl.Schema({"a": pl.String, "b": pl.Int64()})
        assert dataframe_utils.convert_schema_into_json_schema(schema) == {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }

        schema = pl.Schema({"a": pl.String, "b": pl.Int64()})
        assert dataframe_utils.convert_schema_into_json_schema(schema, True) == {
            "type": ["object", "null"],
            "properties": {
                "a": {"type": ["string", "null"]},
                "b": {"type": ["integer", "null"]},
            },
        }

    def test_struct(self) -> None:
        schema = pl.Schema({"a": pl.Struct({"x": pl.Float64(), "y": pl.Float64()})})
        assert dataframe_utils.convert_schema_into_json_schema(schema) == {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                },
            },
        }

    def test_list(self) -> None:
        schema = pl.Schema({"a": pl.List(pl.Float64())})
        assert dataframe_utils.convert_schema_into_json_schema(schema) == {
            "type": "object",
            "properties": {
                "a": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
        }

    def test_array(self) -> None:
        schema = pl.Schema({"a": pl.Array(pl.Float64(), 2)})
        assert dataframe_utils.convert_schema_into_json_schema(schema) == {
            "type": "object",
            "properties": {
                "a": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
        }


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
