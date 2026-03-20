from cyclopts import App

from aflux import utils

from ..parameters import VideoFile

app = App(help="Inspect video metadata.")


@app.command
def stream(video: VideoFile) -> None:
    """Get video stream information."""
    info = utils.get_video_stream_info(video)
    print(info.model_dump_json())


@app.command
def frames(video: VideoFile, *, keyframes_only: bool = False) -> None:
    """Get video frame informations."""
    if keyframes_only:
        frame_infos = utils.get_video_keyframe_infos(video)
    else:
        frame_infos = utils.get_video_frame_infos(video)
    for frame_info in frame_infos:
        print(frame_info.model_dump_json())
