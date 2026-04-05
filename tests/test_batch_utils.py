import asyncio
import concurrent.futures
import time

import pytest

from aflux.utils._batch_utils import arun_batch, run_batch


@pytest.fixture
def anyio_backend():
    return "asyncio"


class TestSyncBatch:
    def test_order_preservation(self) -> None:
        def slow_task(index: int, sleep_time: float) -> int:
            time.sleep(sleep_time)
            return index

        kwargs_list = [
            {"index": 0, "sleep_time": 0.03},
            {"index": 1, "sleep_time": 0.02},
            {"index": 2, "sleep_time": 0.01},
            {"index": 3, "sleep_time": 0.00},
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = run_batch(executor, slow_task, kwargs_list, batch_size=4)

        assert results == [0, 1, 2, 3]


class TestAsyncBatch:
    @pytest.mark.anyio
    async def test_order_preservation(self) -> None:
        async def slow_task(index: int, sleep_time: float) -> int:
            await asyncio.sleep(sleep_time)
            return index

        kwargs_list = [
            {"index": 0, "sleep_time": 0.03},
            {"index": 1, "sleep_time": 0.02},
            {"index": 2, "sleep_time": 0.01},
            {"index": 3, "sleep_time": 0.00},
        ]

        async with asyncio.TaskGroup() as task_group:
            results = await arun_batch(task_group, slow_task, kwargs_list, batch_size=4)

        assert results == [0, 1, 2, 3]
