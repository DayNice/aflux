import re
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, override


class BaseKey(metaclass=ABCMeta):
    """A key for use as an object's getter."""

    @abstractmethod
    def __call__(self, obj: Any) -> Any: ...

    @abstractmethod
    def __str__(self) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)!r})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        raise NotImplementedError


class AttrKey(BaseKey):
    """A key for accessing an object's attribute."""

    __match_args__ = ("name",)

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @override
    def __call__(self, obj: Any) -> Any:
        return getattr(obj, self.name)

    @override
    def __str__(self) -> str:
        return self.name


class ItemKey(BaseKey):
    """A key for accessing an object's item."""

    __match_args__ = ("index",)

    def __init__(self, index: int) -> None:
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @override
    def __call__(self, obj: Any) -> Any:
        return obj[self.index]

    @override
    def __str__(self) -> str:
        return f"[{self.index}]"


class IterKey(BaseKey):
    """A key for iterating an object into a list."""

    @override
    def __call__(self, obj: Any) -> list[Any]:
        return list(obj)

    @override
    def __str__(self) -> str:
        return "[]"


class PickKey(BaseKey):
    """A key for picking an object into a list."""

    __match_args__ = ("names",)

    def __init__(self, names: Iterable[str]) -> None:
        self._names = tuple(names)
        if len(self._names) == 0:
            msg = "Provide at least one name."
            raise ValueError(msg)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @override
    def __call__(self, obj: Any) -> list[Any]:
        return [getattr(obj, name) for name in self.names]

    @override
    def __str__(self) -> str:
        return f"{{{','.join(self.names)}}}"


class Key(BaseKey):
    """A key for accessing attributes and items of an object."""

    __match_args__ = ("parts",)

    # (name | {name1,name2} | [index]) followed by (.name | .{name1,name2} | [index])
    _text_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:\w+|\{\w+(?:,\w+)*\}|\[(?:-\d+|\d*)\])(?:\.\w+|\.\{\w+(?:,\w+)*\}|\[(?:-\d+|\d*)\])*$"
    )
    _token_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"(?P<name>\w+)|\{(?P<pick>\w+(?:,\w+)*)\}|\[(?P<index>-?\d+)?\]"
    )

    def __init__(
        self,
        parts: "str | Iterable[AttrKey | ItemKey | IterKey | PickKey] | Key",
    ) -> None:
        """Create a key for accessing attributes and items of an object.

        Args:
            parts: A text representation of a key, or an iterable of its parts.
                Passing a Key instance is allowed for convenience.

        Examples:
            >>> Key("a[0][].b")
            Key('a[0][].b')
            >>> Key([AttrKey("a"), ItemKey(0), IterKey(), AttrKey("b")])
            Key('a[0][].b')
        """
        self._parts: tuple[AttrKey | ItemKey | IterKey | PickKey, ...]
        match parts:
            case Key() as key:
                self._parts = tuple(key.parts)
            case str() as text:
                self._parts = tuple(self.parse(text))
            case _:
                self._parts = tuple(parts)
        if len(self.parts) == 0:
            raise ValueError("Provide at least one key part.")
        self._getter: Callable[[Any], Any] | None = None

    @property
    def parts(self) -> tuple[AttrKey | ItemKey | IterKey | PickKey, ...]:
        return self._parts

    @override
    def __call__(self, obj: Any) -> Any:
        if self._getter is None:
            self._getter = self._build_getter()
        return self._getter(obj)

    @override
    def __str__(self) -> str:
        text_parts: list[str] = [str(self.parts[0])]
        for part in self.parts[1:]:
            if isinstance(part, (AttrKey, PickKey)):
                text_parts.append(".")
            text_parts.append(str(part))
        return "".join(text_parts)

    @classmethod
    def parse(cls, text: str) -> list[AttrKey | ItemKey | IterKey | PickKey]:
        """Parse a text representation of a key into corresponding parts.

        Args:
            text: A text representation of a key.

        Returns:
            A list of parsed key parts.

        Examples:
            >>> Key.parse("a.b")
            [AttrKey('a'), AttrKey('b')]
            >>> Key.parse("[][-1]")
            [IterKey('[]'), ItemKey('[-1]')]
            >>> Key.parse("a[0][].{x,y}")
            [AttrKey('a'), ItemKey('[0]'), IterKey('[]'), PickKey('{x,y}')]
        """
        if cls._text_pattern.fullmatch(text) is None:
            msg = f"Text representation of key is invalid: {text!r}"
            raise ValueError(msg)

        parts: list[AttrKey | ItemKey | IterKey | PickKey] = []
        for m in cls._token_pattern.finditer(text):
            name, pick, index_str = m.group("name"), m.group("pick"), m.group("index")
            if name is not None:
                parts.append(AttrKey(name=name))
            elif pick is not None:
                parts.append(PickKey(names=pick.split(",")))
            elif index_str is not None:
                parts.append(ItemKey(index=int(index_str)))
            else:
                parts.append(IterKey())
        return parts

    def _build_getter(self) -> Callable[[Any], Any]:
        def chain_key_with_getter(
            key: AttrKey | ItemKey | IterKey | PickKey,
            getter: Callable[[Any], Any],
        ) -> Callable[[Any], Any]:
            if isinstance(key, (IterKey, PickKey)):
                return lambda obj: [getter(el) for el in key(obj)]
            return lambda obj: getter(key(obj))

        getter = self.parts[-1]
        for part in reversed(self.parts[:-1]):
            getter = chain_key_with_getter(part, getter)
        return getter
