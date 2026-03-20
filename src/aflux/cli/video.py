import pathlib
from typing import Annotated

from cyclopts import App, Parameter, validators

app = App(help="Inspect and process video files.")


VideoFile = Annotated[
    pathlib.Path,
    Parameter(validator=validators.Path(exists=True)),
]
