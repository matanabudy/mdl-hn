import math
import random
import sys

from loguru import logger


class SimulatedAnnealing:
    def __init__(
        self,
        initial_temperature,
        threshold,
        cooling_rate,
        get_energy_func,
        get_neighbour_func,
        calc_mdl_score_func,
        max_iterations=sys.maxsize,
        early_stop_iterations=sys.maxsize,
        num_restarts=1,
        logging_interval=5000,
    ):
        self.initial_temperature = initial_temperature
        self.threshold = threshold
        self.cooling_rate = cooling_rate
        self.best_hypothesis = None
        self.best_energy = sys.maxsize
        self.early_stop_iterations = early_stop_iterations
        self.max_iterations = max_iterations
        self.get_energy_func = get_energy_func
        self.get_neighbour_func = get_neighbour_func
        self.calc_mdl_score_func = calc_mdl_score_func
        self.num_restarts = num_restarts
        self.logging_interval = logging_interval

    def run(self, initial_hypothesis, data):
        current_hypothesis = initial_hypothesis
        current_temperature = self.initial_temperature

        current_iteration = 0
        current_restart = 0
        num_iterations_without_improvement = 0
        should_stop = False
        try:
            while True:
                if current_temperature < self.threshold:
                    logger.debug(
                        f"Reached threshold of {self.threshold}, stopping after this iteration"
                    )
                    should_stop = True
                if should_stop:
                    if current_restart < self.num_restarts:
                        logger.debug(
                            f"Restarting SA, current restart {current_restart}"
                        )
                        current_restart += 1
                        current_iteration = 0
                        num_iterations_without_improvement = 0
                        current_temperature = self.initial_temperature
                        current_hypothesis = self.best_hypothesis
                        should_stop = False
                        continue
                    else:
                        logger.debug(
                            f"Reached max restarts of {self.num_restarts}, stopping"
                        )
                        break
                if current_iteration >= self.max_iterations:
                    logger.debug(f"Reached max iterations of {self.max_iterations}")
                    should_stop = True
                    continue

                current_iteration += 1
                current_energy = self.get_energy_func(current_hypothesis, data)

                if current_energy < self.best_energy:
                    num_iterations_without_improvement = 0
                    self.best_hypothesis = current_hypothesis
                    self.best_energy = current_energy
                else:
                    num_iterations_without_improvement += 1
                    if num_iterations_without_improvement >= self.early_stop_iterations:
                        logger.debug(
                            f"Reached early stop iterations of {self.early_stop_iterations}, stopping"
                        )
                        should_stop = True

                new_hypothesis = self.get_neighbour_func(current_hypothesis, data)

                (
                    best_grammar_score,
                    best_data_given_grammar_score,
                ) = self.calc_mdl_score_func(self.best_hypothesis, data)

                log_iteration_msg = (
                    f"Iteration {current_iteration}, Best G: {best_grammar_score}, "
                    f"Best D|G: {best_data_given_grammar_score},"
                    f"Best MDL: {best_grammar_score + best_data_given_grammar_score}"
                )
                if current_iteration % self.logging_interval == 0:
                    logger.debug(log_iteration_msg)

                new_energy = self.get_energy_func(new_hypothesis, data)
                energy_delta = new_energy - current_energy
                if energy_delta < 0:
                    probability_to_change = 1
                else:
                    probability_to_change = math.exp(
                        -energy_delta * 1.0 / current_temperature
                    )
                if random.random() <= probability_to_change:
                    current_hypothesis = new_hypothesis
                current_temperature = self.cooling_rate * current_temperature
        except KeyboardInterrupt:
            logger.debug(
                "Keyboard interrupt, returning current hypothesis and best hypothesis"
            )
            return current_hypothesis, self.best_hypothesis

        return current_hypothesis, self.best_hypothesis
