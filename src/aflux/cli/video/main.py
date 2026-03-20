import json
import pathlib
from collections.abc import Iterator
from typing import cast

import av
import rich.progress
from cyclopts import App
from rich.console import Console

from aflux.utils import decode_video_frames_by_indices

from ..parameters import Indices, VideoFile
from .info import app as info_app

app = App(help="Inspect a video.")
app.command(info_app, name="info")


@app.command
def frames(
    video_file: VideoFile,
    output_dir: pathlib.Path,
    indices: Indices,
    *,
    progress_bar: bool = False,
) -> None:
    """Save frames at given indices as image files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    iterator = zip(indices, decode_video_frames_by_indices(video_file, indices))
    iterator = cast(Iterator[tuple[int, av.VideoFrame]], iterator)
    iterator = rich.progress.track(
        iterator,
        total=len(indices),
        console=Console(stderr=True),
        disable=not progress_bar,
    )

    for index, frame in iterator:
        output_name = f"frame_{index:06d}.png"
        output_file = output_dir / output_name
        frame.to_image().save(output_file)
        print(json.dumps({"frame_index": index, "output_name": output_name}))
