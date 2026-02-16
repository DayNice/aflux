import fractions
import functools
import math
import pathlib
from collections.abc import Iterable, Iterator
from fractions import Fraction
from types import TracebackType
from typing import Self, cast

import av
import av.container
import av.stream
import av.video.reformatter
import numpy as np
import numpy.typing as npt

from aflux.types import VideoFrameInfo, VideoStatistics, VideoStreamInfo

from . import _stats_utils


class VideoReader:
    def __init__(self, video_file: str | pathlib.Path) -> None:
        self.file = pathlib.Path(video_file)
        self._container = av.open(self.file)
        if len(self._container.streams.video) == 0:
            msg = f"File should contain at least one video stream: {video_file}"
            raise ValueError(msg)
        self._stream = self._container.streams.video[0]

    def _seek_pts(self, pts: int, *, backward: bool = True) -> bool:
        try:
            self._container.seek(pts, backward=backward, stream=self._stream)
            return True
        except av.error.PermissionError as e:
            if e.args != (1, "Operation not permitted"):
                raise
            # FFmpeg exception for trying to seek beyond last keyframe
            return False

    def _demux_packets(self) -> Iterator[av.Packet]:
        return self._container.demux(self._stream)

    @functools.cached_property
    def _first_keyframe_pts(self) -> int:
        self._seek_pts(0)
        packet = next(self._demux_packets())
        assert isinstance(packet, av.Packet)
        assert packet.is_keyframe, "Packet should belong to a keyframe."
        assert packet.pts is not None, "Keyframe should have a pts."
        return packet.pts

    @functools.cached_property
    def _last_keyframe_pts(self) -> int:
        # compute initial search boundary
        high_pts = 1
        while self._seek_pts(high_pts, backward=False):
            high_pts = high_pts * 2
        low_pts = high_pts // 2

        # binary search for last keyframe
        while (high_pts - low_pts) > 1:
            mid_pts = low_pts + (high_pts - low_pts) // 2
            if self._seek_pts(mid_pts, backward=False):
                low_pts = mid_pts
            else:
                high_pts = mid_pts
        return low_pts

    @functools.cached_property
    def _frames_per_time_base(self) -> Fraction:
        assert self._stream.time_base is not None
        assert self._stream.average_rate is not None
        return self._stream.time_base * self._stream.average_rate

    @functools.cached_property
    def _stream_info(self) -> VideoStreamInfo:
        # stream attributes available in read mode
        assert self._stream.average_rate is not None
        assert self._stream.time_base is not None
        assert self._stream.pix_fmt is not None

        num_channels = len(self._stream.format.components)
        codec = self._stream.codec.canonical_name

        num_frames = self._stream.frames
        if num_frames == 0:
            num_frames = self._get_num_frames()

        video_stream_info = VideoStreamInfo(
            fps=self._stream.average_rate,
            time_base=self._stream.time_base,
            height=self._stream.height,
            width=self._stream.width,
            num_channels=num_channels,
            codec=codec,
            pixel_format=self._stream.pix_fmt,
            num_frames=num_frames,
        )
        return video_stream_info

    def _get_num_frames(self) -> int:
        # we assume the first frame is a keyframe
        min_pts = self._first_keyframe_pts
        max_pts = 0
        assert self._seek_pts(self._last_keyframe_pts)
        for packet in self._demux_packets():
            if packet.pts is None:
                continue
            max_pts = max(max_pts, packet.pts)
        return math.ceil((max_pts - min_pts) * self._frames_per_time_base) + 1

    def _estimate_frame_pts_by_index(self, frame_index: int) -> int:
        if frame_index < 0:
            msg = f"Frame index should be non-negative: {frame_index}"
            raise ValueError(msg)
        if frame_index >= self._stream_info.num_frames:
            msg = f"Frame index should be less than size: {frame_index}"
            raise ValueError(msg)

        # we assume the first frame is a keyframe
        first_frame_pts = self._first_keyframe_pts
        return first_frame_pts + math.ceil(frame_index / self._frames_per_time_base)

    def get_stream_info(self) -> VideoStreamInfo:
        return self._stream_info

    def get_frame_infos(self) -> list[VideoFrameInfo]:
        assert self._seek_pts(0)
        frame_infos = [el for el in self._demux_frame_infos()]
        frame_infos.sort(key=lambda el: el.timestamp)
        return frame_infos

    def get_first_frame_info(self) -> VideoFrameInfo:
        # we assume the first frame is a keyframe
        frame_info = self.get_first_keyframe_info()
        assert self._stream.start_time is not None, "Failed to determine start time."
        assert frame_info.pts == self._stream.start_time
        return frame_info

    def get_last_frame_info(self) -> VideoFrameInfo:
        found_frame_info: VideoFrameInfo | None = None

        assert self._seek_pts(self._last_keyframe_pts)
        for frame_info in self._demux_frame_infos():
            if found_frame_info is None or found_frame_info.pts < frame_info.pts:
                found_frame_info = frame_info

        assert found_frame_info is not None
        return found_frame_info

    def get_keyframe_infos(self) -> list[VideoFrameInfo]:
        keyframe_infos: list[VideoFrameInfo] = []
        prev_keyframe_pts: int = -1

        while self._seek_pts(prev_keyframe_pts + 1, backward=False):
            keyframe_info = next(self._demux_frame_infos())
            assert isinstance(keyframe_info, VideoFrameInfo)
            assert keyframe_info.is_keyframe, "Packet should belong to a keyframe."

            keyframe_infos.append(keyframe_info)
            prev_keyframe_pts = keyframe_info.pts

        return keyframe_infos

    def get_first_keyframe_info(self) -> VideoFrameInfo:
        assert self._seek_pts(0, backward=False)
        frame_info = next(self._demux_frame_infos())

        assert isinstance(frame_info, VideoFrameInfo)
        assert frame_info.is_keyframe, "Packet should belong to a keyframe."
        return frame_info

    def get_last_keyframe_info(self) -> VideoFrameInfo:
        assert self._seek_pts(self._last_keyframe_pts, backward=False)
        frame_info = next(self._demux_frame_infos())

        assert isinstance(frame_info, VideoFrameInfo)
        assert frame_info.is_keyframe, "Packet should belong to a keyframe."
        return frame_info

    def _demux_frame_infos(self) -> Iterator[VideoFrameInfo]:
        for packet in self._demux_packets():
            if packet.pts is None:
                continue
            frame_info = VideoFrameInfo(
                timestamp=packet.pts * self._stream_info.time_base,
                dts=packet.dts if packet.dts is not None else packet.pts,
                pts=packet.pts,
                is_keyframe=packet.is_keyframe,
            )
            yield frame_info

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def get_video_stream_info(video_file: str | pathlib.Path) -> VideoStreamInfo:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_stream_info()


def get_video_frame_infos(video_file: str | pathlib.Path) -> list[VideoFrameInfo]:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_frame_infos()


def get_video_last_frame_info(video_file: str | pathlib.Path) -> VideoFrameInfo:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_last_frame_info()


def get_video_keyframe_infos(video_file: str | pathlib.Path) -> list[VideoFrameInfo]:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_keyframe_infos()


def get_video_last_keyframe_info(video_file: str | pathlib.Path) -> VideoFrameInfo:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_last_keyframe_info()


def decode_video_frames(
    video_file: str | pathlib.Path,
    frame_indices: Iterable[int],
) -> list[av.VideoFrame]:
    frame_indices = list(frame_indices)
    if len(frame_indices) == 0:
        return []

    frame_index_to_original_pos = {frame_index: pos for pos, frame_index in enumerate(frame_indices)}
    frame_indices.sort()

    video_stream_info = get_video_stream_info(video_file)
    video_frame_infos = get_video_frame_infos(video_file)
    assert video_stream_info.num_frames == len(video_frame_infos)

    if frame_indices[0] < 0:
        msg = f"Frame index should be non-negative: {frame_indices[0]}"
        raise ValueError(msg)
    if frame_indices[-1] >= video_stream_info.num_frames:
        msg = f"Frame index should be less than size: {frame_indices[-1]}"
        raise ValueError(msg)

    keyframe_indices: list[int] = []
    keyframe_indices.append(0)  # frame at index 0 is an implicit keyframe
    for i in range(1, len(video_frame_infos)):
        if not video_frame_infos[i].is_keyframe:
            continue
        keyframe_indices.append(i)

    keyframe_pts_map: dict[int, list[int]] = {}
    for frame_index in frame_indices:
        # locate keyframe at or before current frame
        pos = np.searchsorted(keyframe_indices, frame_index, side="right") - 1
        keyframe_index = keyframe_indices[pos]
        keyframe_pts = video_frame_infos[keyframe_index].pts

        if keyframe_pts not in keyframe_pts_map:
            keyframe_pts_map[keyframe_pts] = []
        keyframe_pts_map[keyframe_pts].append(frame_index)

    found_frame_map: dict[int, av.VideoFrame] = {}
    with av.open(video_file, "r") as container:
        try:
            stream = container.streams.video[0]
        except IndexError:
            msg = f"File should contain at least one video stream: {video_file}"
            raise ValueError(msg)
        stream.thread_type = "AUTO"

        for keyframe_pts, target_indices in sorted(keyframe_pts_map.items()):
            target_pts_map = {video_frame_infos[i].pts: i for i in target_indices}
            assert len(target_pts_map) == len(target_indices), "All pts values within video should be unique."
            max_target_pts = max(target_pts_map.keys())

            container.seek(keyframe_pts, stream=stream)
            for frame in container.decode(stream):
                assert frame.pts is not None
                assert frame.pts <= max_target_pts, "Target pts should exist in video."

                if frame.pts in target_pts_map:
                    target_index = target_pts_map.pop(frame.pts)
                    found_frame_map[target_index] = frame
                if len(target_pts_map) == 0:
                    break

            assert len(target_pts_map) == 0, "Target pts should exist in video."

    found_frames: list[av.VideoFrame | None] = [None for _ in range(len(frame_indices))]
    for frame_index, frame in found_frame_map.items():
        pos = frame_index_to_original_pos[frame_index]
        found_frames[pos] = frame

    for frame in found_frames:
        assert frame is not None
    return cast(list[av.VideoFrame], found_frames)


def decode_video_frames_into_numpy(
    video_file: str | pathlib.Path,
    frame_indices: Iterable[int],
) -> npt.NDArray[np.float32]:
    frames = decode_video_frames(video_file, frame_indices)
    if len(frames) == 0:
        video_stream_info = get_video_stream_info(video_file)
        shape = (0, video_stream_info.num_channels, video_stream_info.height, video_stream_info.width)
        return np.empty(shape, dtype=np.float32)

    arr_list: list[npt.NDArray[np.uint8]] = []
    video_reformatter = av.video.reformatter.VideoReformatter()
    for frame in frames:
        frame = video_reformatter.reformat(frame, format="rgb24")
        arr = cast(npt.NDArray[np.uint8], frame.to_ndarray())
        assert len(arr.shape) == 3
        arr = arr.transpose(2, 0, 1)
        arr_list.append(arr)

    return np.stack(arr_list).astype(np.float32) / 255.0


def compute_video_statistics(video_file: str | pathlib.Path) -> VideoStatistics:
    video_stream_info = get_video_stream_info(video_file)
    indices = _stats_utils.compute_sample_indices(video_stream_info.num_frames)

    frames = decode_video_frames(video_file, indices)

    arr_list: list[npt.NDArray[np.uint8]] = []
    reformatter = av.video.reformatter.VideoReformatter()
    for frame in frames:
        arr = reformatter.reformat(frame, format="rgb24").to_ndarray()
        assert arr.dtype == np.uint8 and len(arr.shape) == 3 and arr.shape[-1] == 3
        arr = cast(npt.NDArray[np.uint8], arr)  # (H, W, 3)
        arr_list.append(arr)
    arr = cast(npt.NDArray[np.uint8], np.stack(arr_list))  # (N, H, W, 3)

    axis = (0, 1, 2)
    statistics = VideoStatistics(
        sample_size=arr.shape[0],
        min=tuple((arr.min(axis) / 255.0).tolist()),
        max=tuple((arr.max(axis) / 255.0).tolist()),
        mean=tuple((arr.mean(axis, dtype=np.float64) / 255.0).tolist()),
        std=tuple((arr.std(axis, dtype=np.float64) / 255.0).tolist()),
    )
    return statistics


def remux_video_into_mp4(
    input_file: str | pathlib.Path,
    output_file: str | pathlib.Path,
) -> None:
    with (
        av.open(input_file) as input_container,
        av.open(output_file, "w", format="mp4") as output_container,
    ):
        input_streams = [el for el in input_container.streams if isinstance(el, (av.VideoStream, av.AudioStream))]

        output_stream_map: dict[int, av.VideoStream | av.AudioStream] = {}
        for input_stream in input_streams:
            output_stream = output_container.add_stream_from_template(input_stream)
            output_stream_map[input_stream.index] = output_stream

            # prevent timestamp drift
            match input_stream:
                case av.VideoStream():
                    # 90,000Hz
                    output_stream.time_base = fractions.Fraction(1, 90000)
                case av.AudioStream():
                    output_stream.time_base = fractions.Fraction(1, input_stream.rate)
                case _:
                    assert input_stream.time_base is not None
                    output_stream.time_base = input_stream.time_base

        for packet in input_container.demux(input_streams):
            # ignore 'flush packet'
            if packet.size == 0 and packet.dts is None and packet.pts is None:
                continue

            output_stream = output_stream_map[packet.stream.index]
            packet.stream = output_stream
            output_container.mux(packet)
