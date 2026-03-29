import re
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, Self, override


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


class SpreadKey(BaseKey):
    @override
    def __call__(self, obj: Any) -> Any:
        return list(obj)

    @override
    def __str__(self) -> str:
        return "[]"


class ChainKey(BaseKey):
    __match_args__ = ("parts",)

    # (name | [index]) followed by (.name | [index])
    _text_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^(?:\w+|\[(?:-\d+|\d*)\])(?:\.\w+|\[(?:-\d+|\d*)\])*$")
    _token_pattern: ClassVar[re.Pattern[str]] = re.compile(r"(?P<name>\w+)|\[(?P<index>-?\d+)?\]")

    def __init__(self, parts: Iterable[AttrKey | ItemKey | SpreadKey]) -> None:
        self.parts = list(parts)
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
    def parse(cls, text: str) -> Self:
        """Parse a chain key from a text representation.

        Examples:
            >>> ChainKey.parse("a.b")
            ChainKey('a.b')
            >>> ChainKey.parse("[][-1]")
            ChainKey('[][-1]')
            >>> ChainKey.parse("a[0][].b")
            ChainKey('a[0][].b')
        """
        if cls._text_pattern.fullmatch(text) is None:
            msg = f"Text representation of key is invalid: {text!r}"
            raise ValueError(msg)

        parts: list[AttrKey | ItemKey | SpreadKey] = []
        for m in cls._token_pattern.finditer(text):
            name, index_str = m.group("name"), m.group("index")
            if name is not None:
                parts.append(AttrKey(name=name))
            elif index_str is not None:
                parts.append(ItemKey(index=int(index_str)))
            else:
                parts.append(SpreadKey())
        return cls(parts=parts)

    def _build_getter(self) -> Callable[[Any], Any]:
        def chain_key_with_getter(
            key: AttrKey | ItemKey | SpreadKey,
            getter: Callable[[Any], Any],
        ) -> Callable[[Any], Any]:
            if isinstance(key, SpreadKey):
                return lambda obj: [getter(el) for el in key(obj)]
            return lambda obj: getter(key(obj))

        getter = self.parts[-1]
        for part in reversed(self.parts[:-1]):
            getter = chain_key_with_getter(part, getter)
        return getter
