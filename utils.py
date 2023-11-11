from pathlib import Path
from typing import Optional

import jax
import numpy as np
import torch
import torchvision
from jax import numpy as jnp

from config import NoiseLevel, NOISE_LEVEL_TO_VARIANCE, NoiseType


def get_6x6_numbers_bitmaps() -> torch.Tensor:
    zero = torch.Tensor(
        [
            [1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
        ]
    )

    one = torch.Tensor(
        [
            [1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1],
        ]
    )

    two = torch.Tensor(
        [
            [1, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
        ]
    )

    three = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
        ]
    )

    four = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
        ]
    )

    five = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 0],
        ]
    )

    six = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
        ]
    )

    seven = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1],
        ]
    )

    eight = torch.Tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
        ]
    )

    nine = torch.Tensor(
        [
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1],
        ]
    )

    return torch.stack([zero, one, two, three, four, five, six, seven, eight, nine])


def change_random_bits_in_image(
    img: torch.Tensor,
    num_bits_to_change: int,
    lower_bound: float = 0,
    upper_bound: float = 1,
    variance: float = 0.3,
) -> torch.Tensor:
    """
    Changes num_bits_to_change random bits in a flattened image
    """
    noisy_img = img.clone()
    indices = np.random.choice(len(noisy_img), num_bits_to_change, replace=False)
    noisy_img[indices] = torch.clamp(
        noisy_img[indices] + variance * torch.randn(num_bits_to_change).float(),
        lower_bound,
        upper_bound,
    )

    return noisy_img


def noise_bitmaps(
    bitmaps: torch.Tensor,
    num_variations_per_bitmap: int,
    noise_level: NoiseLevel,
    is_continuous: bool,
    flatten: bool,
    max_attempts_to_balance: int = 1,
) -> torch.Tensor:
    """
    Adds noise to the given bitmaps
    """
    noisy_bitmaps: list[torch.Tensor] = []

    for bitmap in bitmaps:
        for i in range(num_variations_per_bitmap):
            # If the closest bitmap to the noisy bitmap from the original bitmaps is not its own original bitmap,
            # we try again for max_attempts_to_balance times. This is used to create a balanced dataset in which
            # every noisy bitmap has the original bitmap as its closest bitmap. If we fail to create a balanced
            # dataset, we just use the last noisy bitmap we created.
            noisy_bitmap = None
            for attempt in range(max_attempts_to_balance):
                noisy_bitmap = bitmap.flatten().clone()
                noise_variance = NOISE_LEVEL_TO_VARIANCE[noise_level]
                noisy_bitmap += torch.clamp(
                    torch.randn(noisy_bitmap.shape) * noise_variance, 0, 1
                )

                if not is_continuous:
                    noisy_bitmap = torch.clamp(noisy_bitmap, 0, 1).round()

                noisy_bitmap = (
                    noisy_bitmap if flatten else reshape_flatten_image(noisy_bitmap)
                )

                similarities = bitmaps.reshape(
                    bitmaps.shape[0], -1
                ) @ noisy_bitmap.flatten().unsqueeze(1)

                # if there is more than one bitmap with the same similarity, we continue to the next iteration
                if torch.sum(similarities == torch.max(similarities)) > 1:
                    continue

                closest_bitmap = bitmaps[torch.argmax(similarities)]
                if torch.equal(closest_bitmap, bitmap):
                    break

            if max_attempts_to_balance > 1 and attempt == max_attempts_to_balance - 1:
                raise ValueError(
                    "Tried to create a balanced dataset but failed, some digits are probably too similar"
                )
            if noisy_bitmap is None:
                raise ValueError(
                    f"Failed to create a balanced dataset for bitmap {bitmap}."
                    f"Did you set max_attempts_to_balance to a value less than 1?"
                )
            noisy_bitmaps.append(noisy_bitmap)

    return torch.stack(noisy_bitmaps)


def reshape_flatten_image(img: torch.Tensor) -> torch.Tensor:
    """
    Reshapes a flattened image to a square matrix
    """
    dim = int(np.sqrt(len(img)))
    return img.reshape(dim, dim)


def get_train_bitmap_shape(train_bitmap: torch.Tensor) -> tuple[int, int]:
    """
    Returns the shape of the given train bitmap, assuming it's square
    """
    dim = int(np.sqrt(len(train_bitmap)))
    return dim, dim


def generate_random_flat_bitmap(
    width: int,
    height: int,
    is_continuous: bool,
    lower_bound: float = 0,
    upper_bound: float = 1,
    discrete_values: tuple[int] = (0, 1),
) -> torch.Tensor:
    """
    Generates a random bitmap of the given width and height
    """
    if is_continuous:
        return torch.rand(width * height) * (upper_bound - lower_bound) + lower_bound
    else:
        return torch.from_numpy(
            np.random.choice(discrete_values, size=(width * height))
        ).to(torch.float)


def generate_random_train_bitmap_data(
    width: int, height: int, num_images: int, is_continuous: bool
) -> torch.Tensor:
    """
    Generates a list of random bitmaps for training
    """
    if is_continuous:
        bitmaps = [
            generate_random_flat_bitmap(width, height, is_continuous=True)
            for _ in range(num_images)
        ]
    else:
        bitmaps = [
            generate_random_flat_bitmap(width, height, is_continuous=False)
            for _ in range(num_images)
        ]

    return torch.stack(bitmaps)


def save_bitmaps(bitmaps: torch.Tensor, path: Path) -> None:
    """
    Saves a list of bitmaps as images
    """
    for i, bitmap in enumerate(bitmaps):
        torch.save(bitmap, path / f"{i}.pt")
        # Save a grayscale image
        torchvision.utils.save_image(bitmap.unsqueeze(0), path / f"{i}.png")


def get_experiments_data_path() -> Path:
    """
    Get the path to the experiments data directory
    """
    return Path(__file__).parent / "experiments_data"


def tensor_to_jax(train_data: torch.Tensor) -> jax.Array:
    return jnp.array(train_data.numpy(), dtype=jnp.float32)


def get_correct_and_incorrect_num_memories(
    digits_to_test: list[int], num_examples_per_digit: int
):
    return len(digits_to_test), len(digits_to_test) * num_examples_per_digit


def get_train_data(
    noise_type: NoiseType,
    digits_to_test: list[int],
    num_noise_variations: Optional[int] = None,
    noise_level: Optional[NoiseLevel] = None,
) -> torch.Tensor:
    """
    :param noise_type: one of 'none', 'discrete' or 'continuous_noise'
    :param digits_to_test: list of digits to test
    :param num_noise_variations: number of noisy variations for each digit
    :param noise_level: noise level
    :return: train data
    """
    experiments_path = get_experiments_data_path()
    numbers_path = experiments_path / "numbers"

    # Handle original data differently
    if noise_type == NoiseType.NONE:
        data_path = numbers_path / "originals"
        sorted_files = sorted(data_path / f"{digit}.pt" for digit in digits_to_test)
        return torch.stack([torch.load(file).flatten() for file in sorted_files])

    if not num_noise_variations:
        raise ValueError("num_noise_variations must be specified for noisy data")

    if not noise_level:
        raise ValueError("noise_level must be specified for noisy data")

    if noise_type == NoiseType.DISCRETE:
        data_path = numbers_path / "discrete"
    elif noise_type == NoiseType.BALANCED_DISCRETE:
        data_path = numbers_path / "balanced_discrete"
    elif noise_type == NoiseType.CONTINUOUS:
        data_path = numbers_path / "continuous"
    elif noise_type == NoiseType.BALANCED_CONTINUOUS:
        data_path = numbers_path / "balanced_continuous"
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")

    data_path = data_path / noise_level.value

    data_paths = [data_path / str(digit) for digit in digits_to_test]

    # Load num_noise_variations of .pt files and flatten them
    data = []
    for path in data_paths:
        data += [
            torch.load(file).flatten()
            for i, file in enumerate(path.glob("*.pt"))
            if i < num_noise_variations
        ]
    return torch.stack(data)
