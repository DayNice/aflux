import asyncio
import concurrent.futures
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import Any, cast


def _sync_indexed_wrapper[T, **P](
    index: int,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[int, T]:
    return index, func(*args, **kwargs)


def iter_batch[T](
    executor: concurrent.futures.Executor,
    func: Callable[..., T],
    kwargs_list: Iterable[dict[str, Any]],
    batch_size: int = 32,
) -> Iterator[tuple[int, T]]:
    batch_size = max(1, batch_size)

    pending: set[concurrent.futures.Future[tuple[int, T]]] = set()
    done: set[concurrent.futures.Future[tuple[int, T]]]

    for i, kwargs in enumerate(kwargs_list):
        while len(pending) >= batch_size:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                yield future.result()

        future = executor.submit(_sync_indexed_wrapper, i, func, **kwargs)
        pending.add(future)  # ty: ignore[invalid-argument-type]

    for future in concurrent.futures.as_completed(pending):
        yield future.result()


def run_batch[T](
    executor: concurrent.futures.Executor,
    func: Callable[..., T],
    kwargs_list: Sequence[dict[str, Any]],
    batch_size: int = 32,
) -> list[T]:
    results: list[T | None] = [None] * len(kwargs_list)

    for index, result in iter_batch(executor, func, kwargs_list, batch_size):
        results[index] = result

    return cast(list[T], results)


async def _async_indexed_wrapper[T, **P](
    index: int,
    func: Callable[P, Awaitable[T]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[int, T]:
    return index, await func(*args, **kwargs)


async def aiter_batch[T](
    task_group: asyncio.TaskGroup,
    func: Callable[..., Awaitable[T]],
    kwargs_list: Iterable[dict[str, Any]],
    batch_size: int = 32,
) -> AsyncIterator[tuple[int, T]]:
    batch_size = max(1, batch_size)

    pending: set[asyncio.Task[tuple[int, T]]] = set()
    done: set[asyncio.Task[tuple[int, T]]]

    for i, kwargs in enumerate(kwargs_list):
        while len(pending) >= batch_size:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                yield task.result()

        task = task_group.create_task(_async_indexed_wrapper(i, func, **kwargs))
        pending.add(task)

    for coro in asyncio.as_completed(pending):
        yield coro.result()


async def arun_batch[T](
    task_group: asyncio.TaskGroup,
    func: Callable[..., Awaitable[T]],
    kwargs_list: Sequence[dict[str, Any]],
    batch_size: int = 32,
) -> list[T]:
    results: list[T | None] = [None] * len(kwargs_list)

    async for index, result in aiter_batch(task_group, func, kwargs_list, batch_size):
        results[index] = result

    return cast(list[T], results)
