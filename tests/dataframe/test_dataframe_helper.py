import polars as pl

from aflux.utils import dataframe as dataframe_utils


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
