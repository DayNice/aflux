from fractions import Fraction
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

import aflux.utils.video as video_utils
from aflux.utils.video import VideoReader


@pytest.fixture
def tmp_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "test.mp4"
    red_image = PIL.Image.new("RGB", (128, 128), color=(255, 0, 0))
    blue_image = PIL.Image.new("RGB", (128, 128), color=(0, 0, 255))

    images = [red_image] * 5 + [blue_image] * 5
    video_utils.encode_images_into_mp4(images, video_path, fps=Fraction(30, 1))
    return video_path


class TestVideoReader:
    def test_get_stream_info(self, tmp_video: Path) -> None:
        with VideoReader(tmp_video) as video_reader:
            info = video_reader.get_stream_info()

        assert info.width == 128
        assert info.height == 128
        assert info.num_frames == 10
        assert info.fps == Fraction(30, 1)

    def test_decode_frames(self, tmp_video: Path) -> None:
        with VideoReader(tmp_video) as reader:
            frames = list(reader.decode_frames([9, 0, 5]))

        assert len(frames) == 3

        arr_9 = frames[0].to_ndarray(format="rgb24")
        arr_0 = frames[1].to_ndarray(format="rgb24")
        arr_5 = frames[2].to_ndarray(format="rgb24")

        # Frame 9 is blue
        assert np.mean(arr_9[:, :, 0]) < 50
        assert np.mean(arr_9[:, :, 2]) > 200

        # Frame 0 is red
        assert np.mean(arr_0[:, :, 0]) > 200
        assert np.mean(arr_0[:, :, 2]) < 50

        # Frame 5 is blue
        assert np.mean(arr_5[:, :, 0]) < 50
        assert np.mean(arr_5[:, :, 2]) > 200

    def test_get_frame_infos_with_indices(self, tmp_video: Path) -> None:
        with VideoReader(tmp_video) as reader:
            infos = reader.get_frame_infos([8, 2, 4])

        assert len(infos) == 3
        assert infos[0].frame_index == 8
        assert infos[1].frame_index == 2
        assert infos[2].frame_index == 4

    def test_invalid_frame_indices(self, tmp_video: Path) -> None:
        with VideoReader(tmp_video) as reader:
            with pytest.raises(ValueError, match="should be unique"):
                reader.get_frame_infos([0, 0, 1])

            with pytest.raises(ValueError, match="should be non-negative"):
                reader.get_frame_infos([-1, 0])

            with pytest.raises(ValueError, match="should be less than size"):
                reader.get_frame_infos([0, 9999])

    def test_compute_statistics(self, tmp_video: Path) -> None:
        with VideoReader(tmp_video) as reader:
            stats = reader.compute_statistics()

        assert stats.sample_size == 10

        # Red channel
        assert stats.mean[0] == pytest.approx(0.5, abs=0.1)
        assert stats.max[0] > 0.8

        # Green channel
        assert stats.mean[1] == pytest.approx(0.0, abs=0.1)
        assert stats.max[1] < 0.2

        # Blue channel
        assert stats.mean[2] == pytest.approx(0.5, abs=0.1)
        assert stats.max[2] > 0.8


class TestVideoStatisticsMerge:
    def test_merge_video_statistics_list(self, tmp_path: Path) -> None:
        video_path1 = tmp_path / "test1.mp4"
        video_path2 = tmp_path / "test2.mp4"

        red_image = PIL.Image.new("RGB", (128, 128), color=(255, 0, 0))
        blue_image = PIL.Image.new("RGB", (128, 128), color=(0, 0, 255))

        video_utils.encode_images_into_mp4([red_image] * 5, video_path1, fps=Fraction(30, 1))
        video_utils.encode_images_into_mp4([blue_image] * 5, video_path2, fps=Fraction(30, 1))

        with VideoReader(video_path1) as reader1:
            stats1 = reader1.compute_statistics()

        with VideoReader(video_path2) as reader2:
            stats2 = reader2.compute_statistics()

        merged_stats = video_utils.merge_video_statistics_list([stats1, stats2])

        assert merged_stats.sample_size == stats1.sample_size + stats2.sample_size

        # Red channel (from first video)
        assert merged_stats.mean[0] == pytest.approx(0.5, abs=0.1)
        assert merged_stats.max[0] > 0.8

        # Green channel
        assert merged_stats.mean[1] == pytest.approx(0.0, abs=0.1)
        assert merged_stats.max[1] < 0.2

        # Blue channel (from second video)
        assert merged_stats.mean[2] == pytest.approx(0.5, abs=0.1)
        assert merged_stats.max[2] > 0.8
