import pytest

from mps.data.simulate import simulate_dataset


@pytest.fixture(scope="session")
def small_dataset():
    """8 teams, 2 short seasons: fast enough for tests, enough rows to train."""
    return simulate_dataset([2023, 2024], games_per_team=60, seed=11, n_teams=8)
