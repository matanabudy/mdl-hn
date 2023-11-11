import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import RESULTS_DIR, NoiseLevel, NUM_COMBINATIONS_PER_SUBSET, NoiseType
from experiments import (
    ExperimentType,
    run_experiment,
    save_experiment_results,
)


def generic_run_experiment(
    out_dir: Path,
    experiments_types: list[ExperimentType],
    noise_type: NoiseType,
    should_show: bool = False,
    should_save: bool = False,
    experiment_name: Optional[str] = None,
):
    experiment_name = experiment_name or noise_type.value
    results_dir = out_dir / experiment_name

    digits_to_test = list(range(10))
    possible_num_examples_per_digit = [1, 5, 10, 30]
    if noise_type == NoiseType.NONE:
        possible_noise_levels = [NoiseLevel.NONE]
    else:
        possible_noise_levels = [
            NoiseLevel.LOW,
            NoiseLevel.MEDIUM,
        ]

    for experiments_type in experiments_types:
        results = run_experiment(
            experiments_type,
            digits_to_test,
            noise_type,
            possible_noise_levels,
            possible_num_examples_per_digit,
            NUM_COMBINATIONS_PER_SUBSET,
            should_save,
            should_show,
        )

        save_experiment_results(results_dir, results, experiments_type)


def run_golden_experiment(
    out_dir: Path,
    experiments_types: list[ExperimentType],
):
    generic_run_experiment(
        out_dir, experiments_types, NoiseType.NONE, experiment_name="golden"
    )


def run_discrete_experiment(
    out_dir: Path,
    experiments_types: list[ExperimentType],
):
    generic_run_experiment(out_dir, experiments_types, NoiseType.DISCRETE)


def run_continuous_experiment(
    out_dir: Path,
    experiments_types: list[ExperimentType],
):
    generic_run_experiment(out_dir, experiments_types, NoiseType.CONTINUOUS)


def run_balanced_discrete_experiment(
    out_dir: Path,
    experiments_types: list[ExperimentType],
):
    generic_run_experiment(out_dir, experiments_types, NoiseType.BALANCED_DISCRETE)


def run_balanced_continuous_experiment(
    out_dir: Path,
    experiments_types: list[ExperimentType],
):
    generic_run_experiment(out_dir, experiments_types, NoiseType.BALANCED_CONTINUOUS)


def main():
    # If GPU is available, use it
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Defaults to 0.9 * TOTAL_MEM

    digits_dir = RESULTS_DIR / "digits" / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    digits_dir.mkdir(exist_ok=True, parents=True)

    experiments_types = [ExperimentType.MHN, ExperimentType.MDL_MHN]

    experiments_funcs = [
        run_golden_experiment,
        run_discrete_experiment,
        run_balanced_discrete_experiment,
        run_continuous_experiment,
        run_balanced_continuous_experiment,
    ]
    for experiment_func in experiments_funcs:
        experiment_func(digits_dir, experiments_types)


if __name__ == "__main__":
    main()
