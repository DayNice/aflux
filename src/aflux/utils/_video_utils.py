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

        fps = round(float(stream.average_rate), 3)  # tolerate fractions introduced while remuxing
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


def decode_video_frames(
    video_file: str | pathlib.Path,
    frame_indices: Iterable[int],
) -> list[av.VideoFrame]:
    frame_indices = list(frame_indices)
    if not frame_indices:
        return []

    video_stream_info = get_video_stream_info(video_file)
    video_frame_infos = get_video_frame_infos(video_file)

    target_timestamps = [video_frame_infos[i].timestamp for i in frame_indices]
    from_timestamp = min(target_timestamps)
    to_timestamp = max(target_timestamps)
    tolerance = 0.5 / video_stream_info.fps

    selected_frames: list[av.VideoFrame | None] = [None for _ in range(len(target_timestamps))]
    min_distances: list[float] = [float("inf") for _ in range(len(target_timestamps))]
    closest_timestamps: list[float | None] = [None for _ in range(len(target_timestamps))]
    with av.open(video_file, "r") as container:
        try:
            stream = container.streams.video[0]
        except IndexError:
            msg = f"File should contain at least one video stream: {video_file}"
            raise ValueError(msg)
        assert stream.time_base is not None
        stream.thread_type = "AUTO"

        # seek keyframe before first timestamp.
        offset = int(round(from_timestamp / stream.time_base))
        container.seek(offset, stream=stream)

        # Decode frames in the window
        for frame in container.decode(stream):
            assert frame.pts is not None
            timestamp = float(frame.pts * stream.time_base)

            if timestamp < from_timestamp - tolerance:
                continue
            if timestamp > to_timestamp + tolerance:
                break

            for i, target_timestamp in enumerate(target_timestamps):
                distance = abs(timestamp - target_timestamp)
                if distance >= min_distances[i]:
                    continue
                min_distances[i] = distance
                selected_frames[i] = frame
                closest_timestamps[i] = timestamp

    min_dist = np.array(min_distances)
    if not np.all(min_dist <= tolerance):
        bad_indices = np.where(min_dist > tolerance)[0]
        original_frame_indices = [frame_indices[i] for i in bad_indices]
        queried_ts = [target_timestamps[i] for i in bad_indices]
        closest_ts = [closest_timestamps[i] for i in bad_indices]

        msg = (
            f"Could not find a close enough frame for all indices (tolerance: {tolerance:.4f}s).\n"
            f"Problematic indices: {original_frame_indices}\n"
            f"Their timestamps: {queried_ts}\n"
            f"Closest found timestamps: {closest_ts}"
        )
        raise ValueError(msg)

    for frame in selected_frames:
        assert frame is not None
    return cast(list[av.VideoFrame], selected_frames)


def decode_video_frames_into_numpy(
    video_file: str | pathlib.Path,
    frame_indices: Iterable[int],
) -> npt.NDArray[np.float32]:
    frames = decode_video_frames(video_file, frame_indices)
    if len(frames) == 0:
        video_stream_info = get_video_stream_info(video_file)
        shape = (0, video_stream_info.num_channels, video_stream_info.height, video_stream_info.width)
        return np.empty(shape, dtype=np.float32)

    arr_list: list[npt.NDArray[np.float32]] = []
    video_reformatter = av.video.reformatter.VideoReformatter()
    for frame in frames:
        frame = video_reformatter.reformat(frame, format="rgb24")
        arr = frame.to_ndarray()
        assert len(arr.shape) == 3
        arr = arr.transpose(2, 0, 1)
        arr = arr.astype(np.float32) / 255.0
        arr_list.append(arr)

    return np.stack(arr_list)


def compute_video_statistics(video_file: str | pathlib.Path) -> VideoStatistics:
    video_stream_info = get_video_stream_info(video_file)
    indices = _stats_utils.compute_sample_indices(video_stream_info.num_frames)

    frames = decode_video_frames_into_numpy(video_file, indices)
    assert len(frames.shape) == 4  # (N, C, H, W)

    axis = (0, 2, 3)
    statistics = VideoStatistics(
        sample_size=frames.shape[0],
        min=tuple(frames.min(axis).tolist()),
        max=tuple(frames.max(axis).tolist()),
        mean=tuple(frames.mean(axis).tolist()),
        std=tuple(frames.std(axis).tolist()),
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
