# Adapted from training notebook in https://github.com/bhoov/eqx-hamux
import os
import random

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
import torch
from tqdm.auto import tqdm

import hamux as hmx
from config import (
    TRAIN_WIDTH,
    TRAIN_HEIGHT,
    NUM_EPOCHS,
    DIGITS_TO_TEST,
    DEFAULT_SEED,
    NOISE_TYPE,
    NUM_NOISE_VARIATIONS_TO_TRAIN,
    NOISE_LEVEL_TO_TRAIN,
)
from mdl_mhn import ModernHN, plot_prediction_and_gold, get_golden_mhn
from utils import get_correct_and_incorrect_num_memories, get_train_data

# If GPU is available, use it
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Defaults to 0.9 * TOTAL_MEM


# Train Functions
class DenseSynapseHid(eqx.Module):
    W: jax.Array

    def __init__(self, key, d1: int, d2: int):
        super().__init__()
        self.W = jax.random.normal(key, (d1, d2)) * 0.02 + 0.2

    @property
    def nW(self):
        nc = jnp.sqrt(jnp.sum(self.W**2, axis=0, keepdims=True))
        return self.W / nc

    def __call__(self, g1):
        """Compute the energy of the synapse"""
        x2 = g1 @ self.nW
        beta = 1e1
        return -1 / beta * jax.nn.logsumexp(beta * x2, axis=-1)


def lossf(ham, xs, key, nsteps=1, alpha=1.0):
    """Given a noisy initial image, descend the energy and try to reconstruct the original image at the end of the dynamics.

    Works best with fewer steps due to the vanishing gradient problem"""
    img = xs["input"]
    xs["input"] = img + jr.normal(key, img.shape) * 0.3
    gs = ham.activations(xs)

    for i in range(nsteps):
        # Construct noisy image to final prediction
        evalue, egrad = ham.dEdg(gs, xs, return_energy=True)
        xs = jtu.tree_map(lambda x, dEdg: x - alpha * dEdg, xs, egrad)
        gs = ham.activations(xs)

    # 1step prediction means gradient == image
    img_final = gs["input"]
    loss = ((img_final - img) ** 2).mean()

    logs = {
        "loss": loss,
    }

    return loss, logs


@eqx.filter_jit
def step(img, ham, opt_state, key, opt):
    xs = ham.init_states(bs=img.shape[0])
    xs["input"] = img

    (loss, logs), grads = eqx.filter_value_and_grad(lossf, has_aux=True)(ham, xs, key)
    updates, opt_state = opt.update(grads, opt_state, ham)
    newparams = optax.apply_updates(eqx.filter(ham, eqx.is_array), updates)
    ham = eqx.combine(newparams, ham)
    return ham, opt_state, logs


def train(
    seed: int,
    train_jax: jax.Array,
    num_memories_to_train: int,
    n_epochs: int = NUM_EPOCHS,
    train_width: int = TRAIN_WIDTH,
    train_height: int = TRAIN_HEIGHT,
):
    # Settings seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    neurons = {
        "input": hmx.Neurons(hmx.lagr_spherical_norm, (train_width * train_height,)),
    }
    synapses = {
        "s1": DenseSynapseHid(key, train_width * train_height, num_memories_to_train),
    }
    connections = [(["input"], "s1")]

    ham = hmx.HAM(neurons, synapses, connections)
    xs = ham.init_states()
    gs = ham.activations(xs)
    opt = optax.adam(4e-2)

    pbar = tqdm(range(n_epochs), total=n_epochs)
    img = train_jax[:]
    batch_size = 1

    ham = ham.vectorize()
    opt_state = opt.init(eqx.filter(ham, eqx.is_array))

    noise_rng = jr.PRNGKey(100)
    batch_rng = jr.PRNGKey(10)
    for e in pbar:
        batch_key, batch_rng = jr.split(batch_rng)
        idxs = jr.permutation(batch_key, jnp.arange(img.shape[0]))
        i = 0

        while i < img.shape[0]:
            noise_key, noise_rng = jr.split(noise_rng)
            batch = img[idxs[i : i + batch_size]]
            ham, opt_state, logs = step(batch, ham, opt_state, noise_key, opt)
            i = i + batch_size

        pbar.set_description(
            f'epoch = {e + 1:03d}/{n_epochs:03d}, loss = {logs["loss"].item():2.6f}'
        )

    return ham


def ham_to_mhn(ham) -> ModernHN:
    # Convert ham weights to torch.Tensor
    weights = torch.from_numpy(np.array(ham.synapses["s1"].nW.T))
    return ModernHN(weights)


def main():
    train_data = get_train_data(
        NOISE_TYPE, DIGITS_TO_TEST, NUM_NOISE_VARIATIONS_TO_TRAIN, NOISE_LEVEL_TO_TRAIN
    )
    train_jax = jnp.array(train_data)

    golden_mhn = get_golden_mhn(DIGITS_TO_TEST)

    (
        correct_num_memories,
        incorrect_num_memories,
    ) = get_correct_and_incorrect_num_memories(
        DIGITS_TO_TEST, NUM_NOISE_VARIATIONS_TO_TRAIN
    )

    correct_num_memories_best_mhn = train(DEFAULT_SEED, train_jax, correct_num_memories)
    plot_prediction_and_gold(
        ham_to_mhn(correct_num_memories_best_mhn),
        train_data,
        should_show=True,
        should_save=False,
        golden_mhn=golden_mhn,
    )

    incorrect_num_memories_best_mhn = train(
        DEFAULT_SEED, train_jax, incorrect_num_memories
    )
    plot_prediction_and_gold(
        ham_to_mhn(incorrect_num_memories_best_mhn),
        train_data,
        should_show=True,
        should_save=False,
        golden_mhn=golden_mhn,
    )


if __name__ == "__main__":
    main()
