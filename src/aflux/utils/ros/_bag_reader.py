import functools
import pathlib
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Any, Self

from rosbags.highlevel import AnyReader
from rosbags.interfaces import TopicInfo
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore

from ._message_node import validate_message_field_getter


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
