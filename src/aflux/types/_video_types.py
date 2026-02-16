from fractions import Fraction

from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, PositiveInt


class VideoStreamInfo(BaseModel):
    fps: Fraction
    time_base: Fraction
    height: PositiveInt
    width: PositiveInt
    num_channels: PositiveInt
    codec: str
    pixel_format: str
    num_frames: PositiveInt


class VideoFrameInfo(BaseModel):
    timestamp: NonNegativeFloat
    dts: int
    pts: NonNegativeInt
    is_keyframe: bool


class VideoStatistics(BaseModel):
    sample_size: int
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
