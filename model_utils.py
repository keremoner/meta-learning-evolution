from typing import Callable, Sequence, Any
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util
import pickle
from torch.utils.data import Dataset
from torch.utils.data import Subset

import os

import numpy as np

import flax
import flax.linen as nn

import optax
import jaxopt
import netket as nk

import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from functions import Fourier, Mixture, Slope, Polynomial, WhiteNoise, Shift
from networks import MixtureNeuralProcess, MLP, MeanAggregator, SequenceAggregator, NonLinearMVN, ResBlock
#from dataloader import MixtureDataset

from jax.tree_util import tree_map
from torch.utils import data

def create_model(rng):
    embedding_xs = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)
    embedding_ys = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)
    embedding_both = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)

    projection_posterior = NonLinearMVN(MLP([128, 64], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True))
    output_model = nn.Sequential([
        ResBlock(
            MLP([128, 128], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True),
        ),
        ResBlock(
            MLP([128, 128], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True),
        ),
        nn.Dense(2)
    ])
    # output_model = MLP([64, 64, 2], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True)
    projection_outputs = NonLinearMVN(output_model)

    posterior_aggregator = MeanAggregator(projection_posterior)
    # posterior_aggregator = SequenceAggregator(projection_posterior)

    model = MixtureNeuralProcess(
        embedding_xs, embedding_ys, embedding_both, 
        posterior_aggregator, 
        projection_outputs
    )

    rng, data_init_rng = jax.random.split(rng)

    xs = jax.random.uniform(data_init_rng, (128,)) * 2 - 1

    rng, key = jax.random.split(rng)
    params = model.init({'params': key, 'default': key}, xs[:, None], xs[:, None], xs[:3, None])

    return model, params

def save_model_params(params, path, name):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name+'_best_end_params.pkl'), 'wb') as f:
        pickle.dump(params, f)


def load_model_params(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
