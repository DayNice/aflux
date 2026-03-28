from types import SimpleNamespace

import pytest

from aflux.utils import AttrKey, ChainKey, ItemKey, SpreadKey


def ns(**kwargs: object) -> SimpleNamespace:
    """Shorthand for SimpleNamespace to keep test bodies concise."""
    return SimpleNamespace(**kwargs)


_VALID_PARSE_CASES = [
    # plain attribute chains
    ("a", [AttrKey(name="a")]),
    ("a.b", [AttrKey(name="a"), AttrKey(name="b")]),
    ("a.b.c", [AttrKey(name="a"), AttrKey(name="b"), AttrKey(name="c")]),
    # positive index
    ("a[0]", [AttrKey(name="a"), ItemKey(index=0)]),
    ("a[0].b", [AttrKey(name="a"), ItemKey(index=0), AttrKey(name="b")]),
    ("a[2].b.c", [AttrKey(name="a"), ItemKey(index=2), AttrKey(name="b"), AttrKey(name="c")]),
    # negative index
    ("a[-1]", [AttrKey(name="a"), ItemKey(index=-1)]),
    ("a[-1].b", [AttrKey(name="a"), ItemKey(index=-1), AttrKey(name="b")]),
    ("a[-42].b", [AttrKey(name="a"), ItemKey(index=-42), AttrKey(name="b")]),
    # spread
    ("a[]", [AttrKey(name="a"), SpreadKey()]),
    ("a[].b", [AttrKey(name="a"), SpreadKey(), AttrKey(name="b")]),
    # consecutive brackets
    ("a[0][]", [AttrKey(name="a"), ItemKey(index=0), SpreadKey()]),
    ("a[0][].b", [AttrKey(name="a"), ItemKey(index=0), SpreadKey(), AttrKey(name="b")]),
    ("a[][0]", [AttrKey(name="a"), SpreadKey(), ItemKey(index=0)]),
    ("a[][-1]", [AttrKey(name="a"), SpreadKey(), ItemKey(index=-1)]),
    # leading bracket (no initial name)
    ("[0]", [ItemKey(index=0)]),
    ("[0].b", [ItemKey(index=0), AttrKey(name="b")]),
    ("[]", [SpreadKey()]),
    ("[][-1]", [SpreadKey(), ItemKey(index=-1)]),
    ("[-1]", [ItemKey(index=-1)]),
]

_INVALID_PARSE_CASES = [
    "",  # empty string
    "a..b",  # double dot
    ".a",  # leading dot
    "a.",  # trailing dot
    "a b",  # whitespace
    "[-]",  # bare minus with no digits
    "a[",  # unclosed bracket
    "a]",  # unmatched closing bracket
    "a.[]",  # dot before a leading bracket
]


class TestChainKeyParse:
    @pytest.mark.parametrize(("text", "expected_parts"), _VALID_PARSE_CASES)
    def test_valid(self, text: str, expected_parts: list) -> None:
        assert ChainKey.parse(text).parts == expected_parts

    @pytest.mark.parametrize("text", _INVALID_PARSE_CASES)
    def test_invalid_raises_value_error(self, text: str) -> None:
        with pytest.raises(ValueError, match="invalid"):
            ChainKey.parse(text)

    def test_empty_parts_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ChainKey(parts=[])


class TestChainKeyGetter:
    class TestSingleStep:
        def test_attr(self) -> None:
            assert ChainKey.parse("x")(ns(x=42)) == 42

        def test_positive_index(self) -> None:
            assert ChainKey.parse("items[1]")(ns(items=[10, 20, 30])) == 20

        def test_negative_index(self) -> None:
            assert ChainKey.parse("items[-1]")(ns(items=[10, 20, 30])) == 30

        def test_spread_returns_list(self) -> None:
            assert ChainKey.parse("items[]")(ns(items=[10, 20, 30])) == [10, 20, 30]

    class TestChainedAttrs:
        def test_nested(self) -> None:
            assert ChainKey.parse("a.b")(ns(a=ns(b=99))) == 99

        def test_deeply_nested(self) -> None:
            assert ChainKey.parse("a.b.c")(ns(a=ns(b=ns(c="deep")))) == "deep"

    class TestIndexThenAttr:
        def test_positive(self) -> None:
            pts = [ns(x=1.0, y=2.0), ns(x=3.0, y=4.0)]
            assert ChainKey.parse("pts[0].x")(ns(pts=pts)) == 1.0

        def test_negative(self) -> None:
            pts = [ns(x=1.0), ns(x=9.0)]
            assert ChainKey.parse("pts[-1].x")(ns(pts=pts)) == 9.0

    class TestSpreadThenAttr:
        def test_basic(self) -> None:
            pts = [ns(x=1.0), ns(x=2.0), ns(x=3.0)]
            assert ChainKey.parse("pts[].x")(ns(pts=pts)) == [1.0, 2.0, 3.0]

        def test_preserves_order(self) -> None:
            items = [ns(v=i * 10) for i in range(5)]
            assert ChainKey.parse("items[].v")(ns(items=items)) == [0, 10, 20, 30, 40]

    class TestConsecutiveBrackets:
        def test_spread_then_index(self) -> None:
            assert ChainKey.parse("rows[][-1]")(ns(rows=[[1, 2, 3], [4, 5, 6]])) == [3, 6]

        def test_index_then_spread(self) -> None:
            assert ChainKey.parse("matrix[0][]")(ns(matrix=[[1, 2, 3], [4, 5, 6]])) == [1, 2, 3]

        def test_spread_then_attr_then_index(self) -> None:
            rows = [ns(vals=[10, 20]), ns(vals=[30, 40])]
            assert ChainKey.parse("rows[].vals[0]")(ns(rows=rows)) == [10, 30]

        def test_spread_then_negative_index(self) -> None:
            rows = [ns(vals=[1, 2, 3]), ns(vals=[4, 5])]
            assert ChainKey.parse("rows[].vals[-1]")(ns(rows=rows)) == [3, 5]

    class TestLeadingBracket:
        def test_index(self) -> None:
            assert ChainKey.parse("[1]")([10, 20, 30]) == 20

        def test_negative_index(self) -> None:
            assert ChainKey.parse("[-1]")([10, 20, 30]) == 30

        def test_spread(self) -> None:
            assert ChainKey.parse("[]")([10, 20, 30]) == [10, 20, 30]

        def test_spread_then_negative_index(self) -> None:
            assert ChainKey.parse("[][-1]")([[1, 2], [3, 4], [5]]) == [2, 4, 5]
