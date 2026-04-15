from cyclopts import App

from .bucket import app as bucket_app
from .ros import app as ros_app
from .video import app as video_app

app = App(name="aflux")
app.command(bucket_app, name="bucket")
app.command(ros_app, name="ros")
app.command(video_app, name="video")

if __name__ == "__main__":
    app()
