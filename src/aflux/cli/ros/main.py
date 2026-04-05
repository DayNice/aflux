from cyclopts import App

from .bag import app as bag_app

app = App(help="Inspect Robot Operating System (ROS) artifacts.")
app.command(bag_app, name="bag")

if __name__ == "__main__":
    app()
