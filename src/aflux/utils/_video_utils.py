import bisect
import fractions
import functools
import math
import operator
import pathlib
from collections.abc import Iterable, Iterator
from fractions import Fraction
from types import TracebackType
from typing import Self, TypedDict, cast

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
        self._stream.thread_type = "AUTO"

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

    def _decode_frames(self) -> Iterator[av.VideoFrame]:
        return self._container.decode(self._stream)

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

    @functools.cached_property
    def _keyframe_infos(self) -> list[VideoFrameInfo]:
        keyframe_infos: list[VideoFrameInfo] = []
        prev_keyframe_pts: int = -1

        while self._seek_pts(prev_keyframe_pts + 1, backward=False):
            keyframe_info = next(self._demux_frame_infos())
            assert isinstance(keyframe_info, VideoFrameInfo)
            assert keyframe_info.is_keyframe, "Packet should belong to a keyframe."

            keyframe_infos.append(keyframe_info)
            prev_keyframe_pts = keyframe_info.pts

        return keyframe_infos

    def _search_keyframe_info_by_pts(self, pts: int) -> VideoFrameInfo:
        if pts < 0:
            msg = "Target pts should be a non-negative integer."
            raise ValueError(msg)
        if pts <= self._keyframe_infos[0].pts:
            return self._keyframe_infos[0]

        pos = bisect.bisect_right(self._keyframe_infos, pts, key=operator.attrgetter("pts"))
        return self._keyframe_infos[pos - 1]

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

    def get_frame_info_by_index(self, frame_index: int) -> VideoFrameInfo:
        estimated_frame_pts = self._estimate_frame_pts_by_index(frame_index)

        pts_duration_per_frame = 1 / self._frames_per_time_base
        pts_tolerance = Fraction(1, 2) * pts_duration_per_frame
        # assume at least 1 keyframe per 720 frames
        pts_guard = 720 * pts_duration_per_frame

        found_frame_info: VideoFrameInfo | None = None
        assert self._seek_pts(estimated_frame_pts)
        for frame_info in self._demux_frame_infos():
            pts_diff = abs(frame_info.pts - estimated_frame_pts)
            if pts_diff <= pts_tolerance:
                found_frame_info = frame_info
                break
            assert pts_diff <= pts_guard, "Frame should be within demuxing range."

        assert found_frame_info is not None, "Frame should be within tolerance."
        return found_frame_info

    def get_keyframe_infos(self) -> list[VideoFrameInfo]:
        return self._keyframe_infos

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

    def decode_frames_by_indices(self, frame_indices: Iterable[int]) -> Iterator[av.VideoFrame]:
        frame_indices = list(frame_indices)
        if len(frame_indices) == 0:
            return

        if len(frame_indices) != len(set(frame_indices)):
            msg = "Frame indices should be unique."
            raise ValueError(msg)

        sorted_frame_indices = sorted(frame_indices)
        if sorted_frame_indices[0] < 0:
            msg = f"Frame index should be non-negative: {sorted_frame_indices[0]}"
            raise ValueError(msg)
        if sorted_frame_indices[-1] >= self._stream_info.num_frames:
            msg = f"Frame index should be less than size: {sorted_frame_indices[-1]}"
            raise ValueError(msg)

        class _FrameData(TypedDict):
            frame_index: int
            frame_info: VideoFrameInfo

        keyframe_map: dict[VideoFrameInfo, list[_FrameData]] = {}
        keyframe_reverse_map: dict[int, VideoFrameInfo] = {}
        for frame_index in sorted_frame_indices:
            frame_info = self.get_frame_info_by_index(frame_index)
            frame_data: _FrameData = {"frame_index": frame_index, "frame_info": frame_info}
            keyframe_info = self._search_keyframe_info_by_pts(frame_info.pts)

            if keyframe_info not in keyframe_map:
                keyframe_map[keyframe_info] = []
            keyframe_map[keyframe_info].append(frame_data)
            keyframe_reverse_map[frame_index] = keyframe_info

        found_frame_map: dict[int, av.VideoFrame] = {}
        for frame_index in frame_indices:
            if frame_index in found_frame_map:
                yield found_frame_map.pop(frame_index)
                continue

            keyframe_info = keyframe_reverse_map[frame_index]
            frame_data_list = keyframe_map[keyframe_info]

            frame_pts_map = {el["frame_info"].pts: el["frame_index"] for el in frame_data_list}
            assert len(frame_pts_map) == len(frame_data_list), "All pts values within video should be unique."
            max_frame_pts = max(frame_pts_map.keys())

            assert self._seek_pts(keyframe_info.pts)
            for frame in self._decode_frames():
                assert frame.pts is not None
                assert frame.pts <= max_frame_pts, "Target pts should exist in video."

                if frame.pts in frame_pts_map:
                    decoded_index = frame_pts_map.pop(frame.pts)
                    found_frame_map[decoded_index] = frame
                if len(frame_pts_map) == 0:
                    break
            assert len(frame_pts_map) == 0, "Target pts should exist in video."

            yield found_frame_map.pop(frame_index)

    def compute_statistics(self) -> VideoStatistics:
        sample_indices = _stats_utils.get_sample_indices(self._stream_info.num_frames)
        frames = self.decode_frames_by_indices(sample_indices)

        arr = self.convert_frames_into_rgb_numpy(frames)
        assert len(arr.shape) == 4 and arr.shape[3] == 3, "RGB array should be of shape (N, H, W, 3)."

        axis = (0, 1, 2)
        min_value = arr.min(axis) / 255.0
        max_value = arr.max(axis) / 255.0
        mean_value = arr.mean(axis, dtype=np.float64) / 255.0
        std_value = arr.std(axis, dtype=np.float64) / 255.0

        statistics = VideoStatistics(
            sample_size=arr.shape[0],
            min=tuple(min_value.tolist()),
            max=tuple(max_value.tolist()),
            mean=tuple(mean_value.tolist()),
            std=tuple(std_value.tolist()),
        )
        return statistics

    @staticmethod
    def convert_frames_into_rgb_numpy(frames: Iterable[av.VideoFrame]) -> npt.NDArray[np.uint8]:
        """Convert video frames into RGB numpy.

        Args:
            frames: An iterable of video frames.

        Returns:
            An RGB array of dtype `uint8` and shape `(N, H, W, 3)`.
        """

        reformatter = av.video.reformatter.VideoReformatter()

        arr_list: list[npt.NDArray[np.uint8]] = []
        for frame in frames:
            frame = reformatter.reformat(frame, format="rgb24")
            arr = frame.to_ndarray()

            assert arr.dtype == np.uint8, "RGB array should be type `uint8`."
            assert len(arr.shape) == 3 and arr.shape[2] == 3, "RGB array should be of shape `(H, W, 3)`."
            arr = cast(npt.NDArray[np.uint8], arr)

            arr_list.append(arr)
        return np.stack(arr_list)

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


def get_video_keyframe_infos(video_file: str | pathlib.Path) -> list[VideoFrameInfo]:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_keyframe_infos()


def decode_video_frames_by_indices(
    video_file: str | pathlib.Path,
    frame_indices: Iterable[int],
) -> Iterator[av.VideoFrame]:
    with VideoReader(video_file) as video_reader:
        yield from video_reader.decode_frames_by_indices(frame_indices)


def compute_video_statistics(video_file: str | pathlib.Path) -> VideoStatistics:
    with VideoReader(video_file) as video_reader:
        return video_reader.compute_statistics()


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
