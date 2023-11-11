import shutil
from pathlib import Path

import torch

import utils
from config import NUM_NOISE_VARIATIONS_TO_CREATE, NoiseLevel, NOISE_LEVELS_TO_CREATE
from utils import save_bitmaps, get_experiments_data_path


def save_noisy_digits_separately(
    noisy_bitmaps: torch.Tensor, noisy_path: Path, num_variations: int
) -> None:
    """
    Save the noisy bitmaps in separate folders for easier parsing
    """
    for i in range(10):
        digit_path = noisy_path / str(i)
        digit_path.mkdir(exist_ok=True, parents=True)
        # There are NUM_NOISE_VARIATIONS_TO_CREATE noisy versions for each digit, and they are ordered from 0 to 9
        # i.e. the first NUM_NOISE_VARIATIONS_TO_CREATE noisy versions are for digit 0,
        # the next NUM_NOISE_VARIATIONS_TO_CREATE noisy versions are for digit 1, etc.
        digit_bitmaps = noisy_bitmaps[i * num_variations : (i + 1) * num_variations]
        save_bitmaps(digit_bitmaps, digit_path)


def prepare_number_bitmaps_experiment(
    experiments_data_path: Path,
    num_variations: int = NUM_NOISE_VARIATIONS_TO_CREATE,
    noise_levels: list[NoiseLevel] = NOISE_LEVELS_TO_CREATE,
) -> None:
    """
    Prepare the number bitmaps experiment data
    """
    bitmaps = utils.get_6x6_numbers_bitmaps()
    bitmaps_path = experiments_data_path / "numbers"
    bitmaps_path.mkdir(exist_ok=True)

    originals_path = bitmaps_path / "originals"
    originals_path.mkdir(exist_ok=True)
    save_bitmaps(bitmaps, originals_path)

    noisy_path = bitmaps_path / "discrete"
    noisy_path.mkdir(exist_ok=True)
    for noise_level in noise_levels:
        noisy_bitmaps = utils.noise_bitmaps(
            bitmaps,
            num_variations_per_bitmap=num_variations,
            noise_level=noise_level,
            is_continuous=False,
            flatten=True,
        )
        noisy_bitmaps = noisy_bitmaps.reshape(
            noisy_bitmaps.shape[0], 1, bitmaps.shape[1], bitmaps.shape[2]
        )
        save_noisy_digits_separately(
            noisy_bitmaps, noisy_path / noise_level.value, num_variations
        )

    continuous_noisy_path = bitmaps_path / "continuous"
    continuous_noisy_path.mkdir(exist_ok=True)
    for noise_level in noise_levels:
        continuous_noisy_bitmaps = utils.noise_bitmaps(
            bitmaps,
            num_variations_per_bitmap=num_variations,
            noise_level=noise_level,
            is_continuous=True,
            flatten=True,
        )
        continuous_noisy_bitmaps = continuous_noisy_bitmaps.reshape(
            continuous_noisy_bitmaps.shape[0], 1, bitmaps.shape[1], bitmaps.shape[2]
        )
        save_noisy_digits_separately(
            continuous_noisy_bitmaps,
            continuous_noisy_path / noise_level.value,
            num_variations,
        )

    balanced_discrete_noisy_path = bitmaps_path / "balanced_discrete"
    balanced_discrete_noisy_path.mkdir(exist_ok=True)
    for noise_level in noise_levels:
        balanced_discrete_noisy_bitmaps = utils.noise_bitmaps(
            bitmaps,
            num_variations_per_bitmap=num_variations,
            noise_level=noise_level,
            is_continuous=False,
            flatten=True,
            max_attempts_to_balance=100,
        )
        balanced_discrete_noisy_bitmaps = balanced_discrete_noisy_bitmaps.reshape(
            balanced_discrete_noisy_bitmaps.shape[0],
            1,
            bitmaps.shape[1],
            bitmaps.shape[2],
        )
        save_noisy_digits_separately(
            balanced_discrete_noisy_bitmaps,
            balanced_discrete_noisy_path / noise_level.value,
            num_variations,
        )

    balanced_continuous_noisy_path = bitmaps_path / "balanced_continuous"
    balanced_continuous_noisy_path.mkdir(exist_ok=True)
    for noise_level in noise_levels:
        balanced_continuous_noisy_bitmaps = utils.noise_bitmaps(
            bitmaps,
            num_variations_per_bitmap=num_variations,
            noise_level=noise_level,
            is_continuous=True,
            flatten=True,
            max_attempts_to_balance=100,
        )
        balanced_continuous_noisy_bitmaps = balanced_continuous_noisy_bitmaps.reshape(
            balanced_continuous_noisy_bitmaps.shape[0],
            1,
            bitmaps.shape[1],
            bitmaps.shape[2],
        )
        save_noisy_digits_separately(
            balanced_continuous_noisy_bitmaps,
            balanced_continuous_noisy_path / noise_level.value,
            num_variations,
        )


def reset_experiments_data() -> Path:
    """
    Delete the experiments data directory and create a new one
    """
    experiments_data_path = get_experiments_data_path()
    if experiments_data_path.exists():
        shutil.rmtree(experiments_data_path)

    experiments_data_path.mkdir()

    return experiments_data_path


def main():
    experiments_data_path = reset_experiments_data()

    prepare_number_bitmaps_experiment(experiments_data_path)


if __name__ == "__main__":
    main()
