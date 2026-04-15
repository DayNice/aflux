from cyclopts import App

from .s3 import app as s3_app

app = App(help="Inspect object storage buckets.")
app.command(s3_app, name="s3")

if __name__ == "__main__":
    app()
