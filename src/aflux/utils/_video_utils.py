import fractions
import pathlib
from typing import Iterable, cast

import av
import av.stream
import av.video.reformatter
import numpy as np
import numpy.typing as npt

from aflux.types import VideoFrameInfo, VideoStatistics, VideoStreamInfo

from . import _stats_utils


def get_video_stream_info(video_file: str | pathlib.Path) -> VideoStreamInfo:
    with av.open(video_file) as container:
        try:
            stream = container.streams.video[0]
        except IndexError:
            msg = f"File should contain at least one video stream: {video_file}"
            raise ValueError(msg)

        assert stream.average_rate is not None, f"Failed to determine video average rate: {video_file}"
        assert stream.time_base is not None, f"Failed to determine video time base: {video_file}"
        assert stream.pix_fmt is not None, f"Failed to determine video pixel format: {video_file}"

        fps = stream.average_rate
        time_base = stream.time_base
        height = stream.height
        width = stream.width
        num_channels = len(stream.format.components)
        codec = stream.codec.canonical_name
        pixel_format = stream.pix_fmt
        num_frames = stream.frames

    video_stream_info = VideoStreamInfo(
        fps=fps,
        time_base=time_base,
        height=height,
        width=width,
        num_channels=num_channels,
        codec=codec,
        pixel_format=pixel_format,
        num_frames=num_frames,
    )
    return video_stream_info


def get_video_frame_infos(video_file: str | pathlib.Path) -> list[VideoFrameInfo]:
    with av.open(video_file) as container:
        try:
            stream = container.streams.video[0]
        except IndexError:
            msg = f"File should contain at least one video stream: {video_file}"
            raise ValueError(msg)

        assert stream.time_base is not None, f"Failed to determine video time base: {video_file}"

        frame_infos: list[VideoFrameInfo] = []
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            frame_info = VideoFrameInfo(
                timestamp=float(packet.pts * stream.time_base),
                dts=packet.dts if packet.dts is not None else packet.pts,
                pts=packet.pts,
                is_keyframe=packet.is_keyframe,
            )
            frame_infos.append(frame_info)
        frame_infos.sort(key=lambda el: el.timestamp)

    return frame_infos


def get_video_keyframe_infos(video_file: str | pathlib.Path) -> list[VideoFrameInfo]:
    keyframe_infos: list[VideoFrameInfo] = []
    prev_keyframe_pts: int = -1

    with av.open(video_file, "r") as container:
        try:
            stream = container.streams.video[0]
        except IndexError:
            msg = f"File should contain at least one video stream: {video_file}"
            raise ValueError(msg)

        assert stream.time_base is not None, f"Failed to determine video time base: {video_file}"

        while True:
            try:
                container.seek(prev_keyframe_pts + 1, backward=False, stream=stream)
            except av.error.PermissionError as e:
                if e.args != (1, "Operation not permitted"):
                    raise
                # FFmpeg exception for trying to access beyond last frame
                break

            found_packet: av.Packet | None = None
            for packet in container.demux(stream):
                if packet.is_keyframe:
                    found_packet = packet
                    break
            if found_packet is None:
                break

            assert found_packet.is_keyframe, "Packet should belong to a keyframe."
            assert found_packet.pts is not None, "Keyframe should have a pts."
            assert found_packet.pts > prev_keyframe_pts, "Forward seek should find next keyframe."

            keyframe_info = VideoFrameInfo(
                timestamp=float(found_packet.pts * stream.time_base),
                dts=found_packet.dts if found_packet.dts is not None else found_packet.pts,
                pts=found_packet.pts,
                is_keyframe=found_packet.is_keyframe,
            )
            keyframe_infos.append(keyframe_info)
            prev_keyframe_pts = keyframe_info.pts

    return keyframe_infos


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
