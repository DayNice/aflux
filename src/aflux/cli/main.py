from cyclopts import App

from .video import app as video_app

app = App(name="aflux")
app.command(video_app, name="video")

if __name__ == "__main__":
    app()
