import itertools
import math
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import av.container
import av.stream
import av.video.reformatter
import PIL.Image

from aflux.types.video import VideoStatistics


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


def infer_target_bits_per_pixel(
    num_pixels: int,
    fps: int | Fraction,
    *,
    complexity_constant: float = 16.908,
    spatial_scaling_factor: float = 0.687,
    temporal_scaling_factor: float = 0.560,
) -> float:
    """Infer target bits per pixel for a given (num_pixels, fps) combination.

    Assumes the following relation:

        bit_rate = complexity_constant
            * (num_pixels ^ spatial_scaling_factor)
            * (fps ^ temporal_scaling_factor)

        bit_rate = bits_per_pixel * num_pixels * fps

    Default parameters were set using `scipy.optimize.curve_fit` against the following values.
        - 480p 30fps: 1/15
        - 720p 30fps: 1/20
        - 1080p 30fps: 1/25
        - 1080p 60fps: 1/35
        - 4k 60fps: 1/50
    """
    if num_pixels <= 0:
        raise ValueError("Number of pixels should be a positive value.")
    if fps <= 0:
        raise ValueError("Frame rate should a positive value.")

    bits_per_pixel = complexity_constant
    bits_per_pixel *= num_pixels ** (spatial_scaling_factor - 1.0)
    bits_per_pixel *= float(fps) ** (temporal_scaling_factor - 1.0)
    return bits_per_pixel


def remux_video_into_mp4(
    input_file: str | Path,
    output_file: str | Path,
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
    output_file: str | Path,
    *,
    fps: Fraction = Fraction(30, 1),
    bits_per_pixel: Fraction = Fraction(1, 25),
):
    output_file = Path(output_file)

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
