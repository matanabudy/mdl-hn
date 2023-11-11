import functools
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional

from config import (
    BITMAPS_MEMORIZATION_LIMIT,
    NoiseType,
    CACHE_SIZE,
    GrammarEncodingScheme,
    GRAMMAR_ENCODING_SCHEME,
    D_GIVEN_G_WEIGHT,
    SHOULD_SCALE_D_GIVEN_G_BY_TRAINING_SET_SIZE,
    TRAINING_SET_SCALE_FACTOR,
)
from utils import (
    reshape_flatten_image,
    get_train_bitmap_shape,
    generate_random_train_bitmap_data,
    change_random_bits_in_image,
    generate_random_flat_bitmap,
    get_train_data,
)


@dataclass(frozen=True)
class ModernHN:
    memorized_patterns: torch.Tensor

    def retrieve(self, z: torch.Tensor, beta: float = 8) -> torch.Tensor:
        """
        Retrieve predictions for each bitmap in z using the softmax function
        """
        probabilities = functional.softmax(beta * self.memorized_patterns @ z.T, dim=0)
        return (self.memorized_patterns.T @ probabilities).T

    def retrieve_idx(self, z: torch.Tensor) -> torch.Tensor:
        """
        Retrieve predictions for each bitmap in z
        """
        indices = torch.argmax(self.memorized_patterns @ z.T, dim=0)
        return indices.permute(*torch.arange(indices.ndim - 1, -1, -1))

    def save(self, path: Path):
        torch.save(self.memorized_patterns, path)


def align_memorized_patterns(mhn1: ModernHN, mhn2: ModernHN) -> ModernHN:
    memorized_patterns_copy1 = mhn1.memorized_patterns.clone()
    memorized_patterns_copy2 = mhn2.memorized_patterns.clone()
    indices = torch.argmax(memorized_patterns_copy1 @ memorized_patterns_copy2.T, dim=1)
    sorted_indices = torch.argsort(indices)
    memorized_patterns = memorized_patterns_copy1[sorted_indices]
    return ModernHN(memorized_patterns)


def extract_memorized_bitmaps_from_solution(
    solution: np.ndarray, width: int, height: int
) -> ModernHN:
    """
    A solution is a stacked matrices of memorized bitmaps. Each bitmap is width x height
    Convert it to torch.Tensor and return a ModernHN
    """
    return ModernHN(torch.Tensor(solution.reshape(-1, width * height)))


def stack_memorized_bitmaps(bitmaps: torch.Tensor) -> np.ndarray:
    """
    Opposite to extract_memorized_bitmaps_from_solution
    """
    return np.hstack(bitmaps).flatten()


@functools.lru_cache(maxsize=CACHE_SIZE)
def get_modern_hn_encoding_length(
    hypothesis: ModernHN,
    encoding_scheme: GrammarEncodingScheme = GRAMMAR_ENCODING_SCHEME,
) -> float:
    """
    Return sum of size of memorized patterns
    """
    memorized_patterns_copy = hypothesis.memorized_patterns.clone()
    if encoding_scheme == GrammarEncodingScheme.NAIVE:
        num_patterns = hypothesis.memorized_patterns.size(0)
        pattern_size = hypothesis.memorized_patterns.size(1)
        return num_patterns * pattern_size
    elif encoding_scheme == GrammarEncodingScheme.COMPRESSION_PROXY:
        # Encode weights which favor 0 and 1 and give the other values a score which tries to proxy the compression
        # Note that although it has a justification where we want 0 and 1 to be favored, we may encounter issues
        # when dealing with grayscale images, since we will favor black and white pixels
        res = memorized_patterns_copy * (1 - memorized_patterns_copy)
        return (
            # This function and consts are arbitrary, besides adding 1 to res which states that even 0 and 1 should have
            # a score of 1 (to proxy for counting the number of memories when memories are sharp)
            (1 + 10 * res)
            .pow(2)
            .sum()
            .item()
        )
    else:
        raise ValueError(f"Invalid encoding scheme: {encoding_scheme}")


def get_data_given_grammar_encoding(hypothesis: ModernHN, data: torch.Tensor) -> float:
    """
    Get D given G encoding length
    """
    num_patterns_to_retrieve = data.size(0)

    retrieved_data = hypothesis.retrieve(data)
    distance = torch.dist(data, retrieved_data, p=1).item()

    num_memories_to_choose_from = hypothesis.memorized_patterns.size(0)
    encoded_num_memories_to_choose_from = int(np.log2(num_memories_to_choose_from)) + 1

    shape_size = data.size(1)
    num_bits_to_encode_location_in_memory = int(np.log2(shape_size)) + 1

    encoding = (
        distance * num_bits_to_encode_location_in_memory
        + encoded_num_memories_to_choose_from * num_patterns_to_retrieve
    )

    return (
        encoding / (num_patterns_to_retrieve * TRAINING_SET_SCALE_FACTOR)
        if SHOULD_SCALE_D_GIVEN_G_BY_TRAINING_SET_SIZE
        else encoding
    )


def get_mdl_details(hypothesis: ModernHN, data: torch.Tensor) -> str:
    g_length = get_modern_hn_encoding_length(hypothesis)
    d_given_g = get_data_given_grammar_encoding(hypothesis, data)
    mdl_score = g_length + d_given_g
    return f"G: {g_length:.2f}, D:G: {d_given_g:.2f}, MDL: {mdl_score:.2f}"


def calc_modern_hn_mdl_score(
    hypothesis: ModernHN, data: torch.Tensor
) -> tuple[float, float]:
    """
    Calculate the MDL score for a ModernHN hypothesis
    Returns a tuple of (g_score, d_given_g_score)
    """
    g_score = get_modern_hn_encoding_length(hypothesis)
    d_given_g_score = get_data_given_grammar_encoding(hypothesis, data)
    return g_score, d_given_g_score * D_GIVEN_G_WEIGHT


def plot_mdl_mhn(
    hypothesis: ModernHN, should_show: bool = False, should_save: bool = False
):
    """
    Plot the memorized bitmaps in a grid
    """
    bitmaps = [
        reshape_flatten_image(bitmap) for bitmap in hypothesis.memorized_patterns
    ]
    num_bitmaps = len(bitmaps)
    if num_bitmaps == 1:
        plt.imshow(bitmaps[0], cmap="gray")
        plt.title("MDL-MHN")
        plt.show()
        return

    num_rows = int(np.sqrt(num_bitmaps))
    num_cols = int(np.ceil(num_bitmaps / num_rows))
    fig, axs = plt.subplots(num_rows, num_cols)
    for i, bitmap in enumerate(bitmaps):
        axs_index = (i // num_cols, i % num_cols) if num_rows > 1 else i % num_cols
        axs[axs_index].imshow(bitmap, cmap="gray")
        axs[axs_index].set_title(f"Bitmap {i}")
        axs[axs_index].set_axis_off()

    fig.suptitle("MDL-MHN")

    if should_save:
        # Save the plot with current date and time
        plt.savefig(f"mdl_mhn_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

    if should_show:
        plt.show()


def plot_prediction(
    hypothesis: ModernHN,
    data: torch.Tensor,
    should_show: bool = False,
    should_save: bool = False,
    annotation: str = None,
    golden_mhn: ModernHN = None,
):
    """
    For each bitmap in the data, plot the bitmap, and then next to it the chosen bitmap from the hypothesis
    """
    memorized_patterns_copy = hypothesis.memorized_patterns.clone()
    n_cols = len(memorized_patterns_copy)
    argmaxes = hypothesis.retrieve_idx(data)
    columns = [
        [bitmap for i, bitmap in enumerate(data) if argmaxes[i] == idx]
        for idx in range(n_cols)
    ]
    averages = [torch.zeros_like(memorized_patterns_copy[0]) for _ in columns]
    for id_col, column in enumerate(columns):
        if column:
            averages[id_col] = torch.mean(torch.stack(column, dim=0), dim=0)
    average_mhn = ModernHN(torch.stack(averages))

    n_rows = 2 + max(len(bitmaps) for bitmaps in columns)
    fig, axs = plt.subplots(n_rows, n_cols)
    if n_cols == 1:
        axs = axs.reshape(-1, 1)

    axs[0, 0].annotate(
        "Memory:",
        xy=(-0.2, 1),
        xycoords="axes fraction",
        ha="right",
        va="top",
    )
    axs[1, 0].annotate(
        "Data:",
        xy=(-0.2, 1),
        xycoords="axes fraction",
        ha="right",
        va="top",
    )
    axs[n_rows - 1, 0].annotate(
        "Averages:",
        xy=(-0.2, 1),
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    for idx_col, col in enumerate(columns):
        axs[0, idx_col].imshow(
            reshape_flatten_image(memorized_patterns_copy[idx_col]), cmap="gray"
        )
        axs[n_rows - 1, idx_col].imshow(
            reshape_flatten_image(averages[idx_col]), cmap="gray"
        )

        for idx_num, bitmap in enumerate(col):
            axs[1 + idx_num, idx_col].imshow(reshape_flatten_image(bitmap), cmap="gray")
        for idx_num in range(n_rows):
            axs[idx_num, idx_col].set_axis_off()

    plot_title = get_mdl_details(hypothesis, data)
    if annotation is not None:
        plot_title = f"{annotation} ({plot_title})"
    if golden_mhn is not None:
        golden_details = get_mdl_details(golden_mhn, data)
        plot_title += f"\n(Golden: {golden_details})"
    fig.suptitle(plot_title)

    average_mdl_details = f"average: {get_mdl_details(average_mhn, data)}"
    plt.figtext(0.5, 0.02, average_mdl_details, ha="center", fontsize=12)

    if should_save:
        # Save the plot with current date and time
        title = f"mdl_mhn_prediction_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        plt.savefig(f"{title}.png", bbox_inches="tight", pad_inches=0.01, dpi=1200)

        plt.savefig(f"{title}.svg", bbox_inches="tight", pad_inches=0.01, dpi=1200)

    if should_show:
        plt.show()

    # Close the figure to save memory
    plt.close(fig)


def plot_prediction_and_gold(
    hypothesis: ModernHN,
    data: torch.Tensor,
    should_show: bool = False,
    should_save: bool = False,
    golden_mhn: ModernHN = None,
):
    plot_prediction(
        golden_mhn,
        data,
        should_show=should_show,
        should_save=should_save,
        annotation="Golden",
    )
    hypothesis = align_memorized_patterns(hypothesis, golden_mhn)
    plot_prediction(
        hypothesis,
        data,
        should_show=should_show,
        should_save=should_save,
        annotation="Trained",
        golden_mhn=golden_mhn,
    )


def get_initial_train_mdl_mhn(train_data: torch.Tensor) -> ModernHN:
    """
    Get a ModernHN hypothesis which memorizes the train data
    """
    return ModernHN(train_data)


def crossover_random_bitmaps(hypothesis: ModernHN) -> ModernHN:
    num_memorized_patterns = len(hypothesis.memorized_patterns)
    if num_memorized_patterns < 2:
        return hypothesis

    memorized_patterns_copy = hypothesis.memorized_patterns.clone()
    indices = np.arange(len(memorized_patterns_copy))
    random_indices = np.random.choice(indices, size=2, replace=False)
    idx1 = min(random_indices)
    idx2 = max(random_indices)
    new_bitmap = (memorized_patterns_copy[idx1] + memorized_patterns_copy[idx2]) / 2

    # Assert all values are between 0 and 1
    assert torch.all(new_bitmap >= 0) and torch.all(new_bitmap <= 1)

    memorized_patterns_copy = torch.cat(
        (
            memorized_patterns_copy[:idx1],
            memorized_patterns_copy[idx1 + 1 : idx2],
            memorized_patterns_copy[idx2 + 1 :],
            new_bitmap.reshape(1, -1),
        )
    )
    return ModernHN(memorized_patterns_copy)


def remove_random_bitmap(hypothesis: ModernHN) -> ModernHN:
    """
    Remove a random bitmap from the hypothesis
    """
    if len(hypothesis.memorized_patterns) == 1:
        return hypothesis

    memorized_patterns_copy = hypothesis.memorized_patterns.clone()
    random_bitmap_idx = np.random.randint(len(memorized_patterns_copy))
    memorized_patterns_copy = torch.cat(
        (
            memorized_patterns_copy[:random_bitmap_idx],
            memorized_patterns_copy[random_bitmap_idx + 1 :],
        )
    )
    return ModernHN(memorized_patterns_copy)


def add_random_bitmap(hypothesis: ModernHN) -> ModernHN:
    """
    Add a random bitmap to the hypothesis
    """
    if len(hypothesis.memorized_patterns) == BITMAPS_MEMORIZATION_LIMIT:
        return hypothesis

    memorized_patterns_copy = hypothesis.memorized_patterns.clone()
    width, height = get_train_bitmap_shape(memorized_patterns_copy[0])
    random_bitmap = generate_random_flat_bitmap(width, height, is_continuous=False)
    memorized_patterns_copy = torch.cat(
        (memorized_patterns_copy, random_bitmap.reshape(1, -1))
    )
    return ModernHN(memorized_patterns_copy)


def add_random_continuous_bitmap(hypothesis: ModernHN) -> ModernHN:
    """
    Add a random continuous bitmap to the hypothesis
    """
    if len(hypothesis.memorized_patterns) == BITMAPS_MEMORIZATION_LIMIT:
        return hypothesis

    memorized_patterns_copy = hypothesis.memorized_patterns.clone()
    width, height = get_train_bitmap_shape(memorized_patterns_copy[0])
    random_bitmap = generate_random_flat_bitmap(width, height, is_continuous=True)
    memorized_patterns_copy = torch.cat(
        (memorized_patterns_copy, random_bitmap.reshape(1, -1))
    )
    return ModernHN(memorized_patterns_copy)


def add_random_exemplar(hypothesis: ModernHN, data: torch.Tensor) -> ModernHN:
    """
    Add a random continuous bitmap to the hypothesis
    """
    if len(hypothesis.memorized_patterns) == BITMAPS_MEMORIZATION_LIMIT:
        return hypothesis

    memorized_patterns_copy = hypothesis.memorized_patterns.clone()
    random_idx = np.random.randint(0, data.shape[0])
    random_bitmap = data[random_idx]
    memorized_patterns_copy = torch.cat(
        (memorized_patterns_copy, random_bitmap.reshape(1, -1))
    )
    return ModernHN(memorized_patterns_copy)


def change_random_bitmap(hypothesis: ModernHN) -> ModernHN:
    """
    Change a random bitmap in the hypothesis
    """
    memorized_patterns_copy = hypothesis.memorized_patterns.clone()
    random_bitmap_idx = np.random.randint(len(memorized_patterns_copy))
    width, height = get_train_bitmap_shape(memorized_patterns_copy[0])
    num_bits_to_change = np.random.randint(1, width * height + 1)
    memorized_patterns_copy[random_bitmap_idx] = change_random_bits_in_image(
        memorized_patterns_copy[random_bitmap_idx],
        num_bits_to_change,
    )
    return ModernHN(memorized_patterns_copy)


def get_mutation_funcs(data: torch.Tensor):
    return [
        remove_random_bitmap,
        functools.partial(add_random_exemplar, data=data),
        change_random_bitmap,
        crossover_random_bitmaps,
    ]


def mutate_mdl_mhn(hypothesis: ModernHN, data: torch.Tensor) -> ModernHN:
    """
    Mutate a hypothesis by a random mutation function out of the following:
    """
    mutation_funcs = get_mutation_funcs(data)
    mutation_func = random.choice(mutation_funcs)
    return mutation_func(hypothesis)


def get_random_mhn(width: int, height: int) -> ModernHN:
    """
    Get a random ModernHN hypothesis
    """
    num_bitmaps_to_memorize = np.random.randint(1, BITMAPS_MEMORIZATION_LIMIT + 1)
    initial_memorized_bitmaps = generate_random_train_bitmap_data(
        width, height, num_bitmaps_to_memorize, is_continuous=False
    )
    return ModernHN(initial_memorized_bitmaps)


def get_random_continuous_mhn(width: int, height: int) -> ModernHN:
    """
    Get a random ModernHN hypothesis
    """
    num_bitmaps_to_memorize = np.random.randint(1, BITMAPS_MEMORIZATION_LIMIT + 1)
    initial_memorized_bitmaps = generate_random_train_bitmap_data(
        width, height, num_bitmaps_to_memorize, is_continuous=True
    )
    return ModernHN(initial_memorized_bitmaps)


def get_golden_mhn(digits_to_test: list[int]) -> ModernHN:
    # Get a ModernHN hypothesis which memorizes the original data
    original_data = get_train_data(NoiseType.NONE, digits_to_test)
    return ModernHN(original_data)
