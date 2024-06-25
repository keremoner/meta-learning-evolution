from typing import Callable, Sequence, Any
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util
from torch.utils.data import Dataset
from torch.utils.data import Subset


import flax
import flax.linen as nn

import netket as nk
import numpy as np


from torch.utils.data import  Dataset

from functions import Fourier, Mixture, Slope, Polynomial, WhiteNoise, Shift
from networks import MixtureNeuralProcess, MLP, MeanAggregator, SequenceAggregator, NonLinearMVN, ResBlock
#from dataloader import MixtureDataset

from jax.tree_util import tree_map
from torch.utils import data



# f1 = Fourier(n=4, amplitude=.5, period=1.0)
# f2 = Fourier(n=2, amplitude=.5, period=1.0)
# f3 = Fourier(n=6, amplitude=.5, period=2.0)
# f4 = Fourier(n=3, amplitude=1.0, period=2.0)


# # Used for training and intra train evaluation.

# f5 = Shift(Fourier(n=2, amplitude=1, period=1, period_range=0.5), x_shift=0.0, x_shift_range=1.5, y_shift=0.0, y_shift_range=3.0)
# f6 = Fourier(n=2, amplitude= 1.5, period= 1.0, period_range= 0.2)

# m = Mixture([Shift(f1, y_shift=-2), Shift(f2, y_shift=0.0), Shift(f3, y_shift=2)])
# nm = Mixture([WhiteNoise(m.branches[0], 0.05), WhiteNoise(m.branches[1], 0.2), WhiteNoise(m.branches[2], 0.1)])

rng = jax.random.key(0)

def f(
    key: flax.typing.PRNGKey, 
    x: jax.Array, 
    noise_scale: float = 0.2, 
    mixture_prob: float = 0.5, 
    corrupt: bool = True
):
    key_noise, key_mixture = jax.random.split(key)
    
    noise = jax.random.normal(key, x.shape) * noise_scale

    # return choice * (jnp.sin(2 * jnp.pi * x / 2)) + (1 - choice) * (jnp.cos(2 * jnp.pi * 2 * x)) + corrupt * noise
    return(-2-jnp.cos(2 * jnp.pi * x)) + corrupt * noise




## Joint and uniform samplers


def joint(
    module: nn.Module, 
    data_sampler: Callable[
        [nn.Module, flax.typing.VariableDict, flax.typing.PRNGKey], 
        tuple[jax.Array, jax.Array]
    ],
    key: flax.typing.PRNGKey, 
    return_params: bool = False
) -> tuple[jax.Array, jax.Array]:
    # Samples from p(Z, X, Y)
    key_param, key_rng, key_data = jax.random.split(key, 3)
    
    params = module.init({'param': key_param, 'default': key_rng}, jnp.zeros(()))
    xs, ys = data_sampler(module, params, key_data)

    if return_params:
        return xs, ys, params
    return xs, ys


def uniform(
    module: nn.Module, 
    params: flax.typing.VariableDict, 
    key: flax.typing.PRNGKey, 
    n: int,
    bounds: tuple[float, float]
) -> tuple[jax.Array, jax.Array]:
    
    # Samples from p(X, Y | Z) = p(Y | Z, X)p(X)
    key_xs, key_ys = jax.random.split(key)
    xs = jax.random.uniform(key_xs, (n,)) * (bounds[1] - bounds[0]) + bounds[0]

    ys = jax.vmap(module.apply, in_axes=(None, 0))(params, xs, rngs={'default': jax.random.split(key_ys, n)})

    return xs, ys



class RegressionDataset(Dataset):

    """ Dataset extension from pytorch dataset for regression data 
        It receives a set of x and y values representing several functions
    """
    def __init__(self, dataset):
        self.x , self.y = dataset

    def __len__(self):
        return  self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def _get_data(self):

        return self.x, self.y

@partial(jax.jit, static_argnums=(1))
def gen_sampler_datapoint(key,  sampler):
    
    """ Generate a single regression dataset for a function from a function sampler sampler"""    
    x, y = sampler(key)
    x, y = x[..., None], y[..., None]

    return x,y 


def generate_dataset(rng , dataset_size, sampler, chunk_size):
    
    """ Generate a dataset_size number of training datapoints using chunked vmap and function sampler"""

    # rng_old , key = jax.random.split(rng)
    # keys = jax.random.split(rng, dataset_size)
    # # Apply the function in chunks using netket.jax.vmap_chunked
    # batched_generate = nk.jax.vmap_chunked(
    #     partial(gen_sampler_datapoint,  sampler=sampler),
    #     in_axes=0,
    #     chunk_size=chunk_size
    # )
    # x ,y  = batched_generate(keys)
    
    x, y = jax.vmap(sampler)(jax.random.split(rng, dataset_size)) 
    x, y = x[..., None], y[..., None]
    return  x ,y


def generate_noisy_split_trainingdata(samplers , sampler_ratios, dataset_size, chunk_size, rng):
    """ Generate a dataset with a split of different samplers and ratios
    """

    assert len(samplers) == len(sampler_ratios), "The number of samplers and ratios must be the same"
    assert sum(sampler_ratios) == 1.0, "The sum of the ratios must be 1.0"
    keys = jax.random.split(rng, len(samplers))
    datasets = []
    for (sampler, ratio, key) in zip(samplers, sampler_ratios, keys):
        dataset = generate_dataset(key, int(dataset_size*ratio),  sampler, chunk_size)
        datasets.append(np.asarray(dataset))



    x_datasets, y_datasets = zip(*datasets)
    return  np.asarray((jnp.concatenate(x_datasets), jnp.concatenate(y_datasets)))