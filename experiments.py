import csv
import itertools
import multiprocessing
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import train_mhn
import train_sa
import utils
from config import (
    NoiseLevel,
    NoiseType,
    INITIAL_TEMPERATURE,
    THRESHOLD,
    COOLING_RATE,
    EARLY_STOP_ITERATIONS,
    SA_RESTARTS,
    LOGGING_INTERVAL,
    NUM_WORKERS,
)
from mdl_mhn import (
    ModernHN,
    calc_modern_hn_mdl_score,
    plot_prediction_and_gold,
    get_golden_mhn,
)
from utils import get_train_data


class ExperimentType(Enum):
    MHN = "mhn"
    MDL_MHN = "mdlmhn"


@dataclass
class ExperimentResult:
    experiment_type: str
    seed: int
    noise_type: str
    test_digits: list[int]
    num_examples_per_digit: int
    num_memories: int
    noise_level: str
    g_score: float
    d_given_g_score: float
    golden_mhn_g_score: float
    golden_mhn_d_given_g_score: float
    golden_data: torch.Tensor
    train_data: torch.Tensor
    golden_mhn: ModernHN
    best_mhn: ModernHN


def create_results_dir(out_dir: Path, experiment_name: str) -> Path:
    """
    Create a directory for the experiment results
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = out_dir / f"{experiment_name}_{current_datetime}"
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir


def save_experiment_results(
    out_dir: Path, results: list[ExperimentResult], experiment_type: ExperimentType
) -> None:
    experiment_type_dir = out_dir / experiment_type.value
    experiment_type_dir.mkdir(exist_ok=True, parents=True)

    results_csv_path = experiment_type_dir / "results.csv"
    with open(results_csv_path, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "noise_type",
                "test_digits",
                "num_examples_per_digit",
                "num_memories",
                "noise_level",
                "g_score",
                "d_given_g_score",
                "golden_mhn_g_score",
                "golden_mhn_d_given_g_score",
                "golden_data_path",
                "train_data_path",
                "golden_mhn_path",
                "best_mhn_path",
            ],
        )
        writer.writeheader()
        for result in results:
            result_uuid = uuid.uuid4()

            golden_data_path = experiment_type_dir / f"{result_uuid}_golden_data.pt"
            torch.save(result.golden_data, golden_data_path)

            train_data_path = experiment_type_dir / f"{result_uuid}_train_data.pt"
            torch.save(result.train_data, train_data_path)

            golden_mhn_path = experiment_type_dir / f"{result_uuid}_golden_mhn.pt"
            result.golden_mhn.save(golden_mhn_path)

            best_mhn_path = experiment_type_dir / f"{result_uuid}_best_mhn.pt"
            result.best_mhn.save(best_mhn_path)

            writer.writerow(
                {
                    "seed": result.seed,
                    "noise_type": result.noise_type,
                    "test_digits": result.test_digits,
                    "num_examples_per_digit": result.num_examples_per_digit,
                    "num_memories": result.num_memories,
                    "noise_level": result.noise_level,
                    "g_score": result.g_score,
                    "d_given_g_score": result.d_given_g_score,
                    "golden_mhn_g_score": result.golden_mhn_g_score,
                    "golden_mhn_d_given_g_score": result.golden_mhn_d_given_g_score,
                    "golden_data_path": golden_data_path.name,
                    "train_data_path": train_data_path.name,
                    "golden_mhn_path": golden_mhn_path.name,
                    "best_mhn_path": best_mhn_path.name,
                }
            )


def run_experiment_for_subset(
    experiment_type: ExperimentType,
    numbers_subset: list[int],
    num_examples_per_digit: int,
    noise_type: NoiseType,
    noise_level: NoiseLevel,
    should_save: bool,
    should_show: bool,
) -> list[ExperimentResult]:
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    logger.info(
        f"Testing subset: {numbers_subset},"
        f"noise level: {noise_level.value},"
        f"num examples per digit: {num_examples_per_digit}"
    )
    train_data = get_train_data(
        noise_type,
        numbers_subset,
        num_examples_per_digit,
        noise_level,
    )
    golden_mhn = get_golden_mhn(numbers_subset)
    golden_mhn_g_score, golden_mhn_d_given_g_score = calc_modern_hn_mdl_score(
        golden_mhn, train_data
    )

    seed = np.random.randint(0, 100000)
    trained_mhns = []
    results = []
    if experiment_type == ExperimentType.MHN:
        (
            correct_num_memories,
            incorrect_num_memories,
        ) = utils.get_correct_and_incorrect_num_memories(
            numbers_subset, num_examples_per_digit
        )
        train_jax = utils.tensor_to_jax(train_data)
        for num_memories in [correct_num_memories, incorrect_num_memories]:
            logger.info(f"Training with {num_memories} memories")
            ham = train_mhn.train(seed, train_jax, num_memories)
            best_mhn = train_mhn.ham_to_mhn(ham)
            trained_mhns.append(best_mhn)

    elif experiment_type == ExperimentType.MDL_MHN:
        best_mhn, _ = train_sa.train(
            seed,
            train_data,
            initial_temperature=INITIAL_TEMPERATURE,
            threshold=THRESHOLD,
            cooling_rate=COOLING_RATE,
            early_stop_iterations=EARLY_STOP_ITERATIONS,
            num_restarts=SA_RESTARTS,
            logging_interval=LOGGING_INTERVAL,
        )
        trained_mhns.append(best_mhn)
    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}")

    for trained_mhn in trained_mhns:
        if should_show or should_save:
            plot_prediction_and_gold(
                trained_mhn,
                train_data,
                should_show=should_show,
                should_save=should_save,
                golden_mhn=golden_mhn,
            )

        g_score, d_given_g_score = calc_modern_hn_mdl_score(trained_mhn, train_data)

        results.append(
            ExperimentResult(
                experiment_type=experiment_type.value,
                seed=seed,
                test_digits=numbers_subset,
                num_examples_per_digit=num_examples_per_digit,
                num_memories=len(trained_mhn.memorized_patterns),
                noise_type=NoiseType.DISCRETE.value,
                noise_level=noise_level.value,
                g_score=g_score,
                d_given_g_score=d_given_g_score,
                golden_mhn_g_score=golden_mhn_g_score,
                golden_mhn_d_given_g_score=golden_mhn_d_given_g_score,
                golden_mhn=golden_mhn,
                best_mhn=trained_mhn,
                train_data=train_data,
                golden_data=get_train_data(NoiseType.NONE, numbers_subset),
            )
        )

    return results


def run_experiment(
    experiment_type: ExperimentType,
    digits_to_test: list[int],
    noise_type: NoiseType,
    possible_noise_levels: list[NoiseLevel],
    possible_num_examples_per_digit: list[int],
    num_combinations_per_subset: int,
    should_save: bool,
    should_show: bool,
) -> list[ExperimentResult]:
    possible_subset_lengths = range(1, len(digits_to_test) + 1)
    possible_number_subsets = []
    for subset_length in possible_subset_lengths:
        possible_combinations = list(
            itertools.combinations(digits_to_test, subset_length)
        )
        indices = np.random.choice(
            len(possible_combinations),
            min(num_combinations_per_subset, len(possible_combinations)),
            replace=False,
        )
        possible_number_subsets.extend([possible_combinations[i] for i in indices])

    all_params = itertools.product(
        possible_number_subsets, possible_num_examples_per_digit, possible_noise_levels
    )

    with multiprocessing.Pool(NUM_WORKERS) as p:
        results = p.starmap(
            run_experiment_for_subset,
            [
                (
                    experiment_type,
                    number_subset,
                    num_examples_per_digit,
                    noise_type,
                    noise_level,
                    should_save,
                    should_show,
                )
                for number_subset, num_examples_per_digit, noise_level in all_params
            ],
        )

    # Flatten the list of lists
    return [result for sublist in results for result in sublist]
