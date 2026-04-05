import pytest

from aflux.utils._stats_utils import get_sample_indices, get_sample_size


class TestGetSampleSize:
    @pytest.mark.parametrize(
        ("population_size", "expected"),
        [
            (-10, 0),
            (0, 0),
            (100, 100),
        ],
    )
    def test_bounds(self, population_size: int, expected: int) -> None:
        assert get_sample_size(population_size) == expected

    def test_large_population(self) -> None:
        size = get_sample_size(1_000_000)
        assert size < 1_000_000
        assert size >= 384


class TestGetSampleIndices:
    def test_spacing(self) -> None:
        population_size = 1000
        indices = get_sample_indices(population_size)
        expected_size = get_sample_size(population_size)

        assert len(indices) == expected_size
        assert indices[0] == 0
        assert indices[-1] == population_size - 1

        for i in range(1, len(indices)):
            assert indices[i] > indices[i - 1]
