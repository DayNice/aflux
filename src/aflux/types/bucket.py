from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, NonNegativeInt


class BucketFileMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Annotated[str, Field(min_length=1)]
    size: NonNegativeInt
    last_modified: AwareDatetime
