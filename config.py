from enum import Enum
from pathlib import Path

# General
CACHE_SIZE = 100_000
DEFAULT_SEED = 42
RESULTS_DIR = Path("results")


# Data
class NoiseType(Enum):
    NONE = "none"
    DISCRETE = "discrete"
    BALANCED_DISCRETE = "balanced_discrete"
    CONTINUOUS = "continuous"
    BALANCED_CONTINUOUS = "balanced_continuous"


class NoiseLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


NOISE_LEVEL_TO_VARIANCE = {
    NoiseLevel.LOW: 0.2,
    NoiseLevel.MEDIUM: 0.3,
    NoiseLevel.HIGH: 0.4,
}


NUM_NOISE_VARIATIONS_TO_CREATE = 30
NUM_NOISE_VARIATIONS_TO_TRAIN = 5
DIGITS_TO_TEST = [0, 1, 2, 3, 4]
NOISE_TYPE = NoiseType.DISCRETE
NOISE_LEVELS_TO_CREATE = [NoiseLevel.LOW, NoiseLevel.MEDIUM, NoiseLevel.HIGH]
NOISE_LEVEL_TO_TRAIN = NoiseLevel.LOW


# MDL-MHN
class InitialHypothesis(Enum):
    RANDOM = "random"
    TRAIN = "train"


BITMAPS_MEMORIZATION_LIMIT = 100
INITIAL_HYPOTHESIS = InitialHypothesis.RANDOM


class GrammarEncodingScheme(Enum):
    NAIVE = "naive"
    COMPRESSION_PROXY = "compression_proxy"


GRAMMAR_ENCODING_SCHEME = GrammarEncodingScheme.COMPRESSION_PROXY
D_GIVEN_G_WEIGHT = 1
SHOULD_SCALE_D_GIVEN_G_BY_TRAINING_SET_SIZE = False
TRAINING_SET_SCALE_FACTOR = 1


# SA Params
LOGGING_INTERVAL = 5000
SA_RESTARTS = 50

INITIAL_TEMPERATURE = 8
THRESHOLD = 10**-300
COOLING_RATE = 0.95
EARLY_STOP_ITERATIONS = 100_000

# Multi SA Params
NUM_WORKERS = None  # None means use all available CPUs
NUM_SEEDS = 10

# Experiments
NUM_COMBINATIONS_PER_SUBSET = 5

# MHN Experiments
NUM_EPOCHS = 100
TRAIN_WIDTH = 6
TRAIN_HEIGHT = 6
