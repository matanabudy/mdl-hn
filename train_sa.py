import random

import numpy as np
import torch
from loguru import logger

from config import (
    DEFAULT_SEED,
    SA_RESTARTS,
    INITIAL_TEMPERATURE,
    THRESHOLD,
    COOLING_RATE,
    EARLY_STOP_ITERATIONS,
    LOGGING_INTERVAL,
    NOISE_TYPE,
    INITIAL_HYPOTHESIS,
    InitialHypothesis,
    NUM_NOISE_VARIATIONS_TO_TRAIN,
    DIGITS_TO_TEST,
    NOISE_LEVEL_TO_TRAIN,
)
from mdl_mhn import (
    calc_modern_hn_mdl_score,
    get_mdl_details,
    plot_prediction_and_gold,
    mutate_mdl_mhn,
    get_random_mhn,
    get_initial_train_mdl_mhn,
    ModernHN,
    get_golden_mhn,
)
from simulated_annealer import SimulatedAnnealing
from utils import (
    get_train_bitmap_shape,
    get_train_data,
)


def get_mdl_mhn_energy(hypothesis, data):
    grammar_score, data_given_grammar_score = calc_modern_hn_mdl_score(hypothesis, data)
    return grammar_score + data_given_grammar_score


def get_mdl_mhn_neighbour(hypothesis, data):
    return mutate_mdl_mhn(hypothesis, data)


def train(
    seed: int,
    train_data: torch.Tensor,
    initial_temperature: int,
    threshold: float,
    cooling_rate: float,
    early_stop_iterations: int,
    num_restarts: int,
    logging_interval: int,
) -> tuple[ModernHN, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_width, train_height = get_train_bitmap_shape(train_data[0])
    initial_hypothesis = (
        get_initial_train_mdl_mhn(train_data)
        if INITIAL_HYPOTHESIS == InitialHypothesis.TRAIN
        else get_random_mhn(train_width, train_height)
    )

    annealer = SimulatedAnnealing(
        initial_temperature,
        threshold,
        cooling_rate,
        get_energy_func=get_mdl_mhn_energy,
        get_neighbour_func=get_mdl_mhn_neighbour,
        calc_mdl_score_func=calc_modern_hn_mdl_score,
        early_stop_iterations=early_stop_iterations,
        num_restarts=num_restarts,
        logging_interval=logging_interval,
    )
    _, best_hypothesis = annealer.run(initial_hypothesis, train_data)
    grammar_score, data_given_grammar_score = calc_modern_hn_mdl_score(
        best_hypothesis, train_data
    )
    mdl_score = grammar_score + data_given_grammar_score
    logger.info(f"SA training finished. {get_mdl_details(best_hypothesis, train_data)}")

    return best_hypothesis, mdl_score


def main():
    train_data = get_train_data(
        NOISE_TYPE, DIGITS_TO_TEST, NUM_NOISE_VARIATIONS_TO_TRAIN, NOISE_LEVEL_TO_TRAIN
    )
    golden_mhn = get_golden_mhn(DIGITS_TO_TEST)

    best_mhn, _ = train(
        DEFAULT_SEED,
        train_data,
        initial_temperature=INITIAL_TEMPERATURE,
        threshold=THRESHOLD,
        cooling_rate=COOLING_RATE,
        early_stop_iterations=EARLY_STOP_ITERATIONS,
        num_restarts=SA_RESTARTS,
        logging_interval=LOGGING_INTERVAL,
    )
    plot_prediction_and_gold(
        best_mhn, train_data, should_show=True, should_save=False, golden_mhn=golden_mhn
    )


if __name__ == "__main__":
    main()
