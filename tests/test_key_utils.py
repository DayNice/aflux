from types import SimpleNamespace

import pytest

from aflux.utils import AttrKey, ItemKey, IterKey, Key, PickKey


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
    # iterate
    ("a[]", [AttrKey(name="a"), IterKey()]),
    ("a[].b", [AttrKey(name="a"), IterKey(), AttrKey(name="b")]),
    # pick
    ("{x,y}", [PickKey(names=["x", "y"])]),
    ("a.{x,y}", [AttrKey(name="a"), PickKey(names=["x", "y"])]),
    ("{a,b}[0]", [PickKey(names=["a", "b"]), ItemKey(index=0)]),
    ("items[].{x,y}", [AttrKey(name="items"), IterKey(), PickKey(names=["x", "y"])]),
    # consecutive brackets
    ("a[0][]", [AttrKey(name="a"), ItemKey(index=0), IterKey()]),
    ("a[0][].b", [AttrKey(name="a"), ItemKey(index=0), IterKey(), AttrKey(name="b")]),
    ("a[][0]", [AttrKey(name="a"), IterKey(), ItemKey(index=0)]),
    ("a[][-1]", [AttrKey(name="a"), IterKey(), ItemKey(index=-1)]),
    # leading bracket (no initial name)
    ("[0]", [ItemKey(index=0)]),
    ("[0].b", [ItemKey(index=0), AttrKey(name="b")]),
    ("[]", [IterKey()]),
    ("[][-1]", [IterKey(), ItemKey(index=-1)]),
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
    "{}",  # empty pick
    "{a,}",  # trailing comma
    "{,b}",  # leading comma
    ".{a,b}",  # leading dot before a pick
    "a.{x, y}",  # whitespace in pick
]


class TestKeyParse:
    @pytest.mark.parametrize(("text", "expected_parts"), _VALID_PARSE_CASES)
    def test_valid(self, text: str, expected_parts: list) -> None:
        assert Key.parse(text) == expected_parts

    @pytest.mark.parametrize("text", _INVALID_PARSE_CASES)
    def test_invalid_raises_value_error(self, text: str) -> None:
        with pytest.raises(ValueError, match="invalid"):
            Key.parse(text)

    def test_empty_parts_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Key([])


class TestKeyGetter:
    class TestSingleStep:
        def test_attr(self) -> None:
            assert Key("x")(ns(x=42)) == 42

        def test_positive_index(self) -> None:
            assert Key("items[1]")(ns(items=[10, 20, 30])) == 20

        def test_negative_index(self) -> None:
            assert Key("items[-1]")(ns(items=[10, 20, 30])) == 30

        def test_iter_returns_list(self) -> None:
            assert Key("items[]")(ns(items=[10, 20, 30])) == [10, 20, 30]

    class TestChainedAttrs:
        def test_nested(self) -> None:
            assert Key("a.b")(ns(a=ns(b=99))) == 99

        def test_deeply_nested(self) -> None:
            assert Key("a.b.c")(ns(a=ns(b=ns(c="deep")))) == "deep"

    class TestIndexThenAttr:
        def test_positive(self) -> None:
            pts = [ns(x=1.0, y=2.0), ns(x=3.0, y=4.0)]
            assert Key("pts[0].x")(ns(pts=pts)) == 1.0

        def test_negative(self) -> None:
            pts = [ns(x=1.0), ns(x=9.0)]
            assert Key("pts[-1].x")(ns(pts=pts)) == 9.0

    class TestIterThenAttr:
        def test_basic(self) -> None:
            pts = [ns(x=1.0), ns(x=2.0), ns(x=3.0)]
            assert Key("pts[].x")(ns(pts=pts)) == [1.0, 2.0, 3.0]

        def test_preserves_order(self) -> None:
            items = [ns(v=i * 10) for i in range(5)]
            assert Key("items[].v")(ns(items=items)) == [0, 10, 20, 30, 40]

    class TestConsecutiveBrackets:
        def test_iter_then_index(self) -> None:
            assert Key("rows[][-1]")(ns(rows=[[1, 2, 3], [4, 5, 6]])) == [3, 6]

        def test_index_then_iter(self) -> None:
            assert Key("matrix[0][]")(ns(matrix=[[1, 2, 3], [4, 5, 6]])) == [1, 2, 3]

        def test_iter_then_attr_then_index(self) -> None:
            rows = [ns(vals=[10, 20]), ns(vals=[30, 40])]
            assert Key("rows[].vals[0]")(ns(rows=rows)) == [10, 30]

        def test_iter_then_negative_index(self) -> None:
            rows = [ns(vals=[1, 2, 3]), ns(vals=[4, 5])]
            assert Key("rows[].vals[-1]")(ns(rows=rows)) == [3, 5]

    class TestLeadingBracket:
        def test_index(self) -> None:
            assert Key("[1]")([10, 20, 30]) == 20

        def test_negative_index(self) -> None:
            assert Key("[-1]")([10, 20, 30]) == 30

        def test_iter(self) -> None:
            assert Key("[]")([10, 20, 30]) == [10, 20, 30]

        def test_iter_then_negative_index(self) -> None:
            assert Key("[][-1]")([[1, 2], [3, 4], [5]]) == [2, 4, 5]

    class TestPickKey:
        def test_pick_basic(self) -> None:
            assert Key("{x,y}")(ns(x=1, y=2, z=3)) == [1, 2]

        def test_pick_chained(self) -> None:
            assert Key("a.{x,y}")(ns(a=ns(x=10, y=20))) == [10, 20]

        def test_pick_then_attr(self) -> None:
            assert Key("{a,b}.val")(ns(a=ns(val=1), b=ns(val=2))) == [1, 2]

        def test_pick_then_index(self) -> None:
            assert Key("{a,b}[-1]")(ns(a=[1, 2], b=[3, 4])) == [2, 4]
