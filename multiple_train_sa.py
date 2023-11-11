"""
Launching multiple processes to train the model, each with a different seed
"""
import multiprocessing
import sys

import numpy as np
from loguru import logger

from config import (
    NUM_WORKERS,
    INITIAL_TEMPERATURE,
    THRESHOLD,
    COOLING_RATE,
    EARLY_STOP_ITERATIONS,
    SA_RESTARTS,
    LOGGING_INTERVAL,
    NOISE_TYPE,
    DIGITS_TO_TEST,
    NUM_NOISE_VARIATIONS_TO_TRAIN,
    NOISE_LEVEL_TO_TRAIN,
    NUM_SEEDS,
)
from mdl_mhn import (
    plot_prediction_and_gold,
    get_golden_mhn,
)
from train_sa import train
from utils import get_train_data


def main():
    # TODO - Better logging
    train_data = get_train_data(
        NOISE_TYPE, DIGITS_TO_TEST, NUM_NOISE_VARIATIONS_TO_TRAIN, NOISE_LEVEL_TO_TRAIN
    )
    golden_mhn = get_golden_mhn(DIGITS_TO_TEST)

    seeds = [np.random.randint(0, 100000) for _ in range(NUM_SEEDS)]
    logger.info(f"Seeds: {seeds}")
    params = [
        (
            seed,
            train_data,
            INITIAL_TEMPERATURE,
            THRESHOLD,
            COOLING_RATE,
            EARLY_STOP_ITERATIONS,
            SA_RESTARTS,
            LOGGING_INTERVAL,
        )
        for seed in seeds
    ]
    with multiprocessing.Pool(NUM_WORKERS) as p:
        results = p.starmap(train, params)

    best_mdl_score = sys.maxsize
    best_mhn = None
    for mhn, mdl_score in results:
        if mdl_score < best_mdl_score:
            best_mdl_score = mdl_score
            best_mhn = mhn

    logger.info(f"Best MDL score: {best_mdl_score}")

    plot_prediction_and_gold(
        best_mhn,
        train_data,
        should_show=True,
        should_save=True,
        golden_mhn=golden_mhn,
    )


if __name__ == "__main__":
    main()
