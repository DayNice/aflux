import functools
import operator
import pathlib
from collections.abc import Callable, Iterable, Iterator
from types import TracebackType
from typing import Any, Self, cast

from rosbags.highlevel import AnyReader
from rosbags.interfaces import Nodetype, TopicInfo
from rosbags.interfaces.typing import NameDesc
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore


def validate_message_field_getter(
    typestore: Typestore,
    msgtype: str,
    key: str,
) -> Callable[[Any], Any]:
    attrs = key.split(".")
    msgdef = typestore.get_msgdef(msgtype)

    for attr in attrs[:-1]:
        field = next((el for el in msgdef.fields if el[0] == attr), None)
        if field is None:
            msg = f"Requested attribute does not exist: {msgdef.name!r} {attr!r} "
            raise ValueError(msg)

        if field[1][0] != Nodetype.NAME:
            msg = f"Requested attribute is not a submessage: {msgdef.name!r} {attr!r} "
            raise ValueError(msg)
        field = cast(NameDesc, field)

        msgdef = typestore.get_msgdef(field[1][1])

    if attrs[-1] not in {x[0] for x in msgdef.fields}:
        msg = f"Requested attribute does not exist: {msgdef.name!r} {attrs[-1]!r} "
        raise ValueError(msg)

    return operator.attrgetter(key)


class BagReader:
    def __init__(
        self,
        bag_parent_dir: str | pathlib.Path,
        typestore: Typestore | None = None,
    ):
        self._bag_parent_dir = pathlib.Path(bag_parent_dir)
        self._bag_dirs = sorted(el.parent for el in self._bag_parent_dir.rglob("metadata.yaml"))
        if typestore is None:
            typestore = get_typestore(Stores.LATEST)
        self._reader = AnyReader(self._bag_dirs, default_typestore=typestore)
        self._reader.open()

    @functools.cached_property
    def topic_info_map(self) -> dict[str, TopicInfo]:
        return self._reader.topics.copy()

    def get_raw_bytes(self, topic: str) -> Iterator[tuple[int, str, bytes]]:
        topic_info = self.topic_info_map[topic]
        for connection, timestamp, rawdata in self._reader.messages(connections=topic_info.connections):
            yield timestamp, connection.msgtype, rawdata

    def get_messages(self, topic: str) -> Iterator[tuple[int, Any]]:
        for timestamp, msgtype, rawdata in self.get_raw_bytes(topic):
            message = self._reader.deserialize(rawdata, msgtype)
            yield timestamp, message

    def get_message_fields(self, topic: str, keys: Iterable[str]) -> Iterator[tuple[int, list[Any]]]:
        topic_info = self.topic_info_map[topic]

        msgtypes = sorted(set(el.msgtype for el in topic_info.connections))
        assert len(msgtypes) != 0, "Topic should have at least one message type."
        if len(msgtypes) != 1:
            raise ValueError("Topic has multiple message types, field access is not supported.")
        msgtype = msgtypes[0]

        getters = [validate_message_field_getter(self._reader.typestore, msgtype, key) for key in keys]
        for timestamp, message in self.get_messages(topic):
            yield timestamp, [el(message) for el in getters]

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
