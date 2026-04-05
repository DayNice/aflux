import itertools
import math
import pathlib
from collections.abc import Iterable, Iterator
from fractions import Fraction
from typing import cast

import av
import av.container
import av.stream
import av.video.reformatter
import PIL.Image

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


def decode_video_frames(
    video_file: str | pathlib.Path,
    frame_indices: Iterable[int],
) -> Iterator[av.VideoFrame]:
    with VideoReader(video_file) as video_reader:
        yield from video_reader.decode_frames(frame_indices)


def compute_video_statistics(video_file: str | pathlib.Path) -> VideoStatistics:
    with VideoReader(video_file) as video_reader:
        return video_reader.compute_statistics()


def merge_video_statistics_list(video_statistics_list: Iterable[VideoStatistics]) -> VideoStatistics:
    video_statistics_list = list(video_statistics_list)
    if len(video_statistics_list) == 0:
        msg = "Provide at least one video statistics."
        raise ValueError(msg)
    first_video_statistics = video_statistics_list[0]
    if len(video_statistics_list) == 1:
        return first_video_statistics

    total_sample_size = sum(el.sample_size for el in video_statistics_list)
    total_min = []
    for i in range(len(first_video_statistics.min)):
        min_scalar = min(el.min[i] for el in video_statistics_list)
        total_min.append(min_scalar)
    total_min = tuple(total_min)

    total_max = []
    for i in range(len(first_video_statistics.max)):
        max_scalar = max(el.max[i] for el in video_statistics_list)
        total_max.append(max_scalar)
    total_max = tuple(total_max)

    total_mean = []
    for i in range(len(first_video_statistics.mean)):
        mean_scalar = sum(el.sample_size * el.mean[i] for el in video_statistics_list) / total_sample_size
        total_mean.append(mean_scalar)
    total_mean = tuple(total_mean)

    total_std = []
    for i in range(len(first_video_statistics.std)):
        square_scalar = 0
        for el in video_statistics_list:
            square_scalar += (el.sample_size - 1) * (el.std[i] ** 2)
            square_scalar += el.sample_size * (el.mean[i] ** 2)
        square_scalar /= total_sample_size

        var_scalar = (square_scalar - total_mean[i] ** 2) * (total_sample_size / (total_sample_size - 1))
        std_scalar = math.sqrt(max(0, var_scalar))
        total_std.append(std_scalar)
    total_std = tuple(total_std)

    return VideoStatistics(
        sample_size=total_sample_size,
        min=total_min,
        max=total_max,
        mean=total_mean,
        std=total_std,
    )


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
                    output_stream.time_base = Fraction(1, 90000)
                case av.AudioStream():
                    output_stream.time_base = Fraction(1, input_stream.rate)
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


def encode_images_into_mp4(
    images: Iterable[PIL.Image.Image],
    output_file: str | pathlib.Path,
    *,
    fps: Fraction = Fraction(30, 1),
    bits_per_pixel: Fraction = Fraction(1, 25),
):
    output_file = pathlib.Path(output_file)

    images = iter(images)
    sample_image = next(images, None)
    if sample_image is None:
        raise ValueError("Provide at least one image.")
    images = itertools.chain([sample_image], images)

    bits_per_sec = sample_image.width * sample_image.height * fps * bits_per_pixel
    output_file.parent.mkdir(parents=True, exist_ok=True)

    encoder_options = {
        "preset": "6",
        "crf": "26",
        "svtav1-params": "tune=0",
        "maxrate": f"{round(bits_per_sec)}",
        "bufsize": f"{round(bits_per_sec * 2)}",
    }

    with av.open(output_file, "w", format="mp4") as container:
        stream = cast(av.VideoStream, container.add_stream("libsvtav1", fps, encoder_options))
        stream.width = sample_image.width
        stream.height = sample_image.height
        stream.pix_fmt = "yuv420p10le"
        stream.gop_size = round(fps * 2)
        stream.time_base = Fraction(1, 90000)

        for image in images:
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
