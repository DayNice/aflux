import fractions
import pathlib
from collections.abc import Iterable, Iterator

import av
import av.container
import av.stream
import av.video.reformatter

from aflux.types import VideoFrameInfo, VideoStatistics, VideoStreamInfo

from ._video_reader import VideoReader


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
