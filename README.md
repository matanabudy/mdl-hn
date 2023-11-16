<h1 align="center">Minimum Description Length Hopfield Networks</h1>

<h3 align="center">
    <a href="https://matanabudy.com">Matan Abudy</a>, 
    <a href="https://www.cs.tau.ac.il/~nurlan/">Nur Lan</a>, 
    <a href="http://www.emmanuel.chemla.free.fr">Emmanuel Chemla</a>,
    <a href="https://english.tau.ac.il/profile/rkatzir">Roni Katzir</a>
</h3>

<h4 align="center">
    <a href="https://neurips.cc/virtual/2023/78187">Associative Memory & Hopfield Networks Workshop, NeurIPS 2023</a>
</h4>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <a href="https://arxiv.org/abs/2311.06518"><img src="https://img.shields.io/badge/arXiv-2311.06518-b31b1b.svg" alt="arXiv"></a>
</p>

##

Code for [Minimum Description Length Hopfield Networks](https://arxiv.org/abs/2311.06518) paper.

## Getting Started

### Setup

Note that it has been tested with Python 3.11, but any version after 3.9 should work.

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt  
```

Before training or running experiments, please make sure to run `python prepare_data.py` to obtain all the required
datasets. Additionally, note that all the scripts below rely on configuration parameters specified in `config.py`. For
instance, the default training of a specific model is performed with the following parameters:

- `DIGITS_TO_TEST = [0, 1, 2, 3, 4]`
- `NOISE_TYPE = NoiseType.DISCRETE`
- `NOISE_LEVEL_TO_TRAIN = NoiseLevel.LOW`
- `NUM_NOISE_VARIATIONS_TO_TRAIN = 5`

Which means running on 5 digits (0, 1, 2, 3, 4) with discrete low noise, and having 5 noisy examples per digit.

### Training a Modern Hopfield Network

Adapted from [eqx-hamux](https://github.com/bhoov/eqx-hamux), use the `train_mhn.py` script to train a Modern Hopfield
Network based on the provided configuration.

### Training a Minimum Description Length Modern Hopfield Network

You can train using Simulated Annealing either single-processed in `train_sa.py` or multi-processed
using `multiple_train_sa.py`. The number of workers is determined by `NUM_WORKERS` in `config.py`.

## Running Experiments

To run the experiments as mentioned in the paper, one should execute `run_experiments.py`. This script runs both MDL-MHN
experiments and MHN experiments. The results are saved in a default directory named `results` and can be parsed
using `parse_results.py`.

### Parsing Results

Parsing results will provide the following statistics and graphs for each dataset you ran against in your `results`
directory. By default, this includes golden, discrete noise, balanced discrete noise, continuous noise, and balanced
continuous noise.

- **If `show_samples=True:`**
    - For MHN experiments, two consecutive experiments of MHN will be sampled: one with the correct number of
      pre-determined clusters for training, and one with an incorrect number.
    - For MDL-MHN experiments, 2 random sub-experiments will be chosen. These samples will be displayed as organized
      plots along with the golden solution.
- **Distance distributions per noise level:** These distributions show the distances from the trained MHN memories for
  both the golden digits and the training set digits. We can observe the shift in MDL-MHN, where the distances
  concentrate towards lower values after training. The statistics of mean and median distances per noise level are also
  provided.
- **MDL-MHN only:**
    - A bar plot of clusters in the trained MHNs, along with the golden baseline.
    - A line plot of the number of memories vs. the number of digits tested will also be shown, with color representing
      the number of training examples and line type representing the level of noise.
    - Accuracies - hard accuracy denotes the number of times MDL-MHN correctly identified the number of clusters, and
      soft accuracy denotes the number of times it correctly identified the number of clusters or a lower number.

## Citation

```bib
@article{abudy2023minimum,
      title={Minimum Description Length Hopfield Networks}, 
      author={Matan Abudy and Nur Lan and Emmanuel Chemla and Roni Katzir},
      year={2023},
      eprint={2311.06518},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

