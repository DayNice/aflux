from cyclopts import App

from .ros import app as ros_app
from .video import app as video_app

app = App(name="aflux")
app.command(ros_app, name="ros")
app.command(video_app, name="video")

if __name__ == "__main__":
    app()
