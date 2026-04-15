import shutil
from typing import TYPE_CHECKING

import boto3
import botocore
import botocore.config
from cyclopts import App

from aflux.utils.bucket import S3Bucket

from ..parameters import InputFile, OutputFile

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client
else:
    S3Client = object

app = App(help="Inspect an S3 bucket.")


def _get_anonymous_s3_client() -> S3Client:
    s3_client = boto3.client(
        "s3",
        config=botocore.config.Config(
            signature_version=botocore.UNSIGNED,
        ),
    )
    return s3_client


def _get_s3_bucket(bucket_name: str, bucket_prefix: str, anonymous: bool) -> S3Bucket:
    s3_client = _get_anonymous_s3_client() if anonymous else None
    s3_bucket = S3Bucket(bucket_name, bucket_prefix, s3_client=s3_client)
    return s3_bucket


@app.command
def check_file_exists(
    remote_path: str,
    *,
    bucket_name: str,
    bucket_prefix: str = "",
    anonymous: bool = False,
) -> None:
    s3_bucket = _get_s3_bucket(bucket_name, bucket_prefix, anonymous)
    file_exists = s3_bucket.check_file_exists(remote_path)
    print("true" if file_exists else "false")


@app.command
def get_file_meta(
    remote_path: str,
    *,
    bucket_name: str,
    bucket_prefix: str = "",
    anonymous: bool = False,
) -> None:
    s3_bucket = _get_s3_bucket(bucket_name, bucket_prefix, anonymous)
    file_meta = s3_bucket.get_file_meta(remote_path)
    print(file_meta.model_dump_json())


@app.command
def get_file_metas(
    *,
    bucket_name: str,
    bucket_prefix: str = "",
    anonymous: bool = False,
) -> None:
    s3_bucket = _get_s3_bucket(bucket_name, bucket_prefix, anonymous)
    for file_meta in s3_bucket.get_file_metas():
        print(file_meta.model_dump_json())


@app.command
def get_file(
    remote_path: str,
    output_file: OutputFile,
    *,
    bucket_name: str,
    bucket_prefix: str = "",
    anonymous: bool = False,
) -> None:
    s3_bucket = _get_s3_bucket(bucket_name, bucket_prefix, anonymous)
    local_file = s3_bucket.get_file(remote_path)
    shutil.move(local_file, output_file)


@app.command
def put_file(
    input_file: InputFile,
    remote_path: str,
    *,
    bucket_name: str,
    bucket_prefix: str = "",
    anonymous: bool = False,
) -> None:
    s3_bucket = _get_s3_bucket(bucket_name, bucket_prefix, anonymous)
    s3_bucket.put_file(input_file, remote_path)


@app.command
def delete_file(
    remote_path: str,
    *,
    bucket_name: str,
    bucket_prefix: str = "",
    anonymous: bool = False,
) -> None:
    s3_bucket = _get_s3_bucket(bucket_name, bucket_prefix, anonymous)
    s3_bucket.delete_file(remote_path)
