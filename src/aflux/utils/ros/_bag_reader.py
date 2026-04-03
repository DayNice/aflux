import functools
import pathlib
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Any, Self

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore

from aflux.types.ros import TopicInfo
from aflux.utils import Key

from ._message_node import (
    StructNode,
    parse_msgtype_into_node,
    validate_message_field_getter,
)


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
        topic_info_map: dict[str, TopicInfo] = {}
        for topic, raw_topic_info in self._reader.topics.items():
            if raw_topic_info.msgtype is None:
                msg = f"Topic with multiple message types is unsupported: {topic!r}"
                raise ValueError(msg)
            topic_info = TopicInfo(
                topic=topic,
                message_type=raw_topic_info.msgtype,
                num_messages=raw_topic_info.msgcount,
            )
            topic_info_map[topic] = topic_info
        return topic_info_map

    def get_message_node(self, topic: str) -> StructNode:
        topic_info = self.topic_info_map[topic]
        return parse_msgtype_into_node(self._reader.typestore, topic_info.message_type)

    def get_raw_bytes(self, topic: str) -> Iterator[tuple[int, str, bytes]]:
        connections = self._reader.topics[topic].connections
        for connection, timestamp, rawdata in self._reader.messages(connections):
            yield timestamp, connection.msgtype, rawdata

    def get_messages(self, topic: str) -> Iterator[tuple[int, Any]]:
        for timestamp, msgtype, rawdata in self.get_raw_bytes(topic):
            message = self._reader.deserialize(rawdata, msgtype)
            yield timestamp, message

    def dump_messages(self, topic: str) -> Iterator[tuple[int, dict[str, Any]]]:
        node = self.get_message_node(topic)
        for timestamp, message in self.get_messages(topic):
            yield timestamp, node.dump_message(message)

    def get_message_fields(self, topic: str, keys: Iterable[str | Key]) -> Iterator[tuple[int, list[Any]]]:
        topic_info = self.topic_info_map[topic]
        getters = [validate_message_field_getter(self._reader.typestore, topic_info.message_type, key) for key in keys]
        for timestamp, message in self.get_messages(topic):
            yield timestamp, [el(message) for el in getters]

    def dump_message_fields(self, topic: str, keys: Iterable[str | Key]) -> Iterator[tuple[int, list[Any]]]:
        node = self.get_message_node(topic)
        for timestamp, message in self.get_messages(topic):
            yield timestamp, [node.dump_message_with_key(message, key) for key in keys]

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
