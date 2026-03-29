import re
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, override


class BaseKey(metaclass=ABCMeta):
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
    __match_args__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    @override
    def __call__(self, obj: Any) -> Any:
        return getattr(obj, self.name)

    @override
    def __str__(self) -> str:
        return self.name


class ItemKey(BaseKey):
    __match_args__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index

    @override
    def __call__(self, obj: Any) -> Any:
        return obj[self.index]

    @override
    def __str__(self) -> str:
        return f"[{self.index}]"


class IterKey(BaseKey):
    @override
    def __call__(self, obj: Any) -> list[Any]:
        return list(obj)

    @override
    def __str__(self) -> str:
        return "[]"


class Key(BaseKey):
    """A key for accessing attributes and items of an object."""

    __match_args__ = ("parts",)

    # (name | [index]) followed by (.name | [index])
    _text_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^(?:\w+|\[(?:-\d+|\d*)\])(?:\.\w+|\[(?:-\d+|\d*)\])*$")
    _token_pattern: ClassVar[re.Pattern[str]] = re.compile(r"(?P<name>\w+)|\[(?P<index>-?\d+)?\]")

    def __init__(
        self,
        parts: str | Iterable[AttrKey | ItemKey | IterKey],
    ) -> None:
        """Create a key for accessing attributes and items of an object.

        Args:
            parts: A text representation of a key, or an iterable of its parts.

        Examples:
            >>> Key("a[0][].b")
            Key('a[0][].b')
            >>> Key([AttrKey("a"), ItemKey(0), IterKey(), AttrKey("b")])
            Key('a[0][].b')
        """
        self.parts = self.parse(parts) if isinstance(parts, str) else list(parts)
        if len(self.parts) == 0:
            raise ValueError("Provide at least one key part.")
        self._getter: Callable[[Any], Any] | None = None

    @override
    def __call__(self, obj: Any) -> Any:
        if self._getter is None:
            self._getter = self._build_getter()
        return self._getter(obj)

    @override
    def __str__(self) -> str:
        text_parts: list[str] = [str(self.parts[0])]
        for part in self.parts[1:]:
            if isinstance(part, AttrKey):
                text_parts.append(".")
            text_parts.append(str(part))
        return "".join(text_parts)

    @classmethod
    def parse(cls, text: str) -> list[AttrKey | ItemKey | IterKey]:
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
            >>> Key.parse("a[0][].b")
            [AttrKey('a'), ItemKey('[0]'), IterKey('[]'), AttrKey('b')]
        """
        if cls._text_pattern.fullmatch(text) is None:
            msg = f"Text representation of key is invalid: {text!r}"
            raise ValueError(msg)

        parts: list[AttrKey | ItemKey | IterKey] = []
        for m in cls._token_pattern.finditer(text):
            name, index_str = m.group("name"), m.group("index")
            if name is not None:
                parts.append(AttrKey(name=name))
            elif index_str is not None:
                parts.append(ItemKey(index=int(index_str)))
            else:
                parts.append(IterKey())
        return parts

    def _build_getter(self) -> Callable[[Any], Any]:
        def chain_key_with_getter(
            key: AttrKey | ItemKey | IterKey,
            getter: Callable[[Any], Any],
        ) -> Callable[[Any], Any]:
            if isinstance(key, IterKey):
                return lambda obj: [getter(el) for el in key(obj)]
            return lambda obj: getter(key(obj))

        getter = self.parts[-1]
        for part in reversed(self.parts[:-1]):
            getter = chain_key_with_getter(part, getter)
        return getter
