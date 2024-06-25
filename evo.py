from typing import Callable, Any
from functools import partial

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import jaxopt
import tqdm
from jax.scipy.stats.norm import logpdf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functions import Fourier, Mixture, Slope, Polynomial, WhiteNoise, Shift
from networks import MixtureNeuralProcess, MLP, MeanAggregator, SequenceAggregator, NonLinearMVN, ResBlock
from typing import Callable, List, Tuple
import random
import numpy as np
from typing import List, Dict, Tuple
import netket as nk
from model_utils import save_model_params, load_model_params
from dataset_generation import joint, uniform, RegressionDataset , generate_noisy_split_trainingdata
from torch.utils.data import DataLoader

# Ensure reproducibility
rng = jax.random.PRNGKey(0)
print('CUDA?', jax.devices(), jax.devices()[0].device_kind)


# Test-configuration
test_resolution = 512

# Train-configuration
num_posterior_mc = 1  # number of latents to sample from p(Z | X, Y)
batch_size = 64  # number of functions to sample from p(Z)
kl_penalty = 1e-4  # Magnitude of the KL-divergence in the loss
num_target_samples = 32
num_context_samples = 64

class Fourier_nr(nn.Module):
    """Generate random functions as the sum of randomized sine waves

    i.e. (simplified),
        z = scale * x - shift,
        y = a_0 + sum_(i=1...n) a_i cos(2pi * i * z - phi_i),
    where a_i are amplitudes and phi_i are phase shifts.

    See the Amplitude-Phase form,
     - https://en.wikipedia.org/wiki/Fourier_series
    """
    n: int = 3
    period: float = 1.0
    amplitude: jax.typing.ArrayLike = jnp.array([1.0, 0.5, 0.25])
    phase: jax.typing.ArrayLike = jnp.array([0.0, jnp.pi/4, jnp.pi/2])

    # def setup(self):
    #     # Ensure amplitude and phase are jax arrays
    #     self.amplitude = jnp.array(self.amplitude)
    #     self.phase = jnp.array(self.phase)

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:
        # shift = self.param(
        #     'shift',
        #     nn.initializers.uniform(scale=jnp.pi),
        #     jnp.shape(x), x.dtype
        # )

        # x = x - shift

        # (heuristic) Monotonically scale amplitudes.
        amplitude = self.amplitude * jnp.arange(1, len(self.amplitude) + 1) / (self.n - 1)

        waves = jnp.cos(
            (2 * jnp.pi * jnp.arange(1, self.n) * x - self.phase) / self.period
        )

        return amplitude[0] / 2.0 + jnp.sum(amplitude[1:] * waves, axis=0)
    
def cross_entropy_error(model, params, x_context, y_context, x_target, y_target , rng , k):
    y_means, y_stds = model.apply(params, x_context, y_context, x_target,k=k, rngs={'default': rng})


    # Lets compute the log likelihood of the target points given the means and stds

    # Ensure y_means and y_stds are squeezed correctly
    y_means = jnp.squeeze(y_means, axis=-1) if k > 1 else jnp.squeeze(y_means)
    y_stds = jnp.squeeze(y_stds, axis=-1) if k > 1 else jnp.squeeze(y_stds)
    full_y = jnp.squeeze(y_target, axis=-1) if k > 1 else jnp.squeeze(y_target) 

    log_pdf = logpdf(full_y, y_means,y_stds) 
   


    return -jnp.mean(log_pdf)



def RMSE_means(model, params, x_context, y_context, x_target, y_target, rng, k):
    
    y_means, y_stds = model.apply(params, x_context, y_context, x_target,k=k, rngs={'default': rng}) 
    
    
    return jnp.sqrt(jnp.mean((y_means - y_target)**2))


# Define chromosome structure
Chromosome = Dict[str, np.ndarray]
# Dummy fitness calculation function (to be replaced with actual regressor model loss inversion)
def calculate_fitness(chromosome: Chromosome, key, model, params, num_evaluation_samples: int = 1) -> float:
    # Create a Fourier instance from the chromosome
    fourier_instance = Fourier_nr(
        n=chromosome['n'],
        period=chromosome['period'],
        amplitude=jnp.array(chromosome['amplitudes']),
        phase=jnp.array(chromosome['phases'])
    )
        # Calculate the loss using the provided evaluate_model function
    # losses = []
    # for _ in range(num_evaluation_samples):
    #     key, subkey = jax.random.split(key)
    #     data_sampler_real = partial(joint, fourier_instance, partial(uniform, n=num_target_samples + num_context_samples, bounds=(-1, 1)))
    #     x, y = data_sampler_real(subkey)
    #     x, y = x[..., None], y[..., None]
    #     X, x_test = jnp.split(x, indices_or_sections=(num_context_samples,))
    #     y, y_test = jnp.split(y, indices_or_sections=(num_context_samples,))
    data_sampler_real = partial(joint, fourier_instance, partial(uniform, n=num_target_samples + num_context_samples, bounds=(-1, 1)))
    x, y = jax.vmap(data_sampler_real)(jax.random.split(key, num_evaluation_samples)) 
    x, y = x[..., None], y[..., None]

    #lets split them into the context and target sets
    x_contexts, x_targets = jnp.split(x, indices_or_sections=(num_context_samples, ), axis=1)
    y_contexts, y_targets = jnp.split(y, indices_or_sections=(num_context_samples, ), axis=1)
    key, subkey = jax.random.split(key)
    ece_errors = nk.jax.vmap_chunked(partial(RMSE_means, model, params, k=num_posterior_mc), in_axes=(0,0,0,0,0), chunk_size=64*100)(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(subkey, num_evaluation_samples))

    #losses.append(loss)
    average_loss = jnp.mean(jnp.array(ece_errors))
    return average_loss


# Initialize population with random chromosomes
def initialize_population(pop_size: int, n_range: Tuple[int, int], period_range: Tuple[float, float], amplitude_range: Tuple[float, float], phase_range: Tuple[float, float]) -> List[Chromosome]:
    population = []
    for _ in range(pop_size):
        n = random.randint(n_range[0], n_range[1])
        period = random.uniform(period_range[0], period_range[1])
        amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], size=(n,))
        phases = np.random.uniform(phase_range[0], phase_range[1], size=(n-1,))
        chromosome = {
            'n': n,
            'period': period,
            'amplitudes': amplitudes,
            'phases': phases
        }
        population.append(chromosome)
    return population

# Crossover between two parent chromosomes
def crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    child = {}
    
    # Choose n and period from parents
    n1, n2 = parent1['n'], parent2['n']
    n = random.choice([n1, n2])
    period = random.uniform(parent1['period'], parent2['period'])

    # Interpolate amplitudes and phases
    amplitudes1 = np.copy(parent1['amplitudes'])
    amplitudes2 = np.copy(parent2['amplitudes'])
    phases1 = np.copy(parent1['phases'])
    phases2 = np.copy(parent2['phases'])
    if n1 < n2:
        amplitudes1 = np.pad(parent1['amplitudes'], (0, n2 - n1), mode='constant', constant_values=0)
        phases1 = np.pad(parent1['phases'], (0, n2 - n1), mode='constant', constant_values=0)
    else:
        amplitudes2 = np.pad(parent2['amplitudes'], (0, n1 - n2), mode='constant', constant_values=0)
        phases2 = np.pad(parent2['phases'], (0, n1 - n2), mode='constant', constant_values=0)
    
    amplitudes = np.array([random.choice([a, b]) for a, b in zip(amplitudes1, amplitudes2)])
    phases = np.array([random.choice([a, b]) for a, b in zip(phases1, phases2)])
    
    # Select correct number of amplitudes and phases for the child
    child['n'] = n
    child['period'] = period
    child['amplitudes'] = amplitudes[:n]
    child['phases'] = phases[:n-1]

    return child

# Mutation function to introduce variations
def mutate(chromosome: Chromosome, n_range: Tuple[int, int], period_range: Tuple[float, float], amplitude_range: Tuple[float, float], phase_range: Tuple[float, float], mutation_rate: float) -> Chromosome:
    if random.uniform(0, 1) < mutation_rate:
        new_n = int(np.round(np.random.normal(chromosome['n'], (n_range[1] - n_range[0])) / 8))
        # # Ensure the new n is within the range
        new_n = max(n_range[0], min(n_range[1], new_n))
        original_n = chromosome['n']
        chromosome['n'] = new_n
        
        if chromosome['n'] != original_n:
            if chromosome['n'] > original_n:
                # Increase the size of the arrays
                additional_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], size=(chromosome['n'] - original_n,))
                additional_phases = np.random.uniform(phase_range[0], phase_range[1], size=(chromosome['n'] - original_n,))
                chromosome['amplitudes'] = np.concatenate((chromosome['amplitudes'], additional_amplitudes))
                chromosome['phases'] = np.concatenate((chromosome['phases'], additional_phases))
            else:
                # Decrease the size of the arrays
                chromosome['amplitudes'] = chromosome['amplitudes'][:chromosome['n']]
                chromosome['phases'] = chromosome['phases'][:chromosome['n'] - 1]


    if random.uniform(0, 1) < mutation_rate:
        new_period = np.random.normal(chromosome['period'], (period_range[1] - period_range[0]) / 8)
        new_period = max(period_range[0], min(period_range[1], new_period))
        chromosome['period'] = new_period
    if random.uniform(0, 1) < mutation_rate:
        for idx, amplitude in enumerate(chromosome['amplitudes']):
            new_amplitude = np.random.normal(amplitude, (amplitude_range[1] - amplitude_range[0]) / 8)
            new_amplitude = max(amplitude_range[0], min(amplitude_range[1], amplitude))
            chromosome['amplitudes'][idx] = new_amplitude
        
        #chromosome['amplitudes'] = np.random.uniform(amplitude_range[0], amplitude_range[1], size=(chromosome['n'],))
    if random.uniform(0, 1) < mutation_rate:
        for idx, phase in enumerate(chromosome['phases']):
            new_phase = np.random.normal(phase, (phase_range[1] - phase_range[0]) / 8)
            new_phase = max(phase_range[0], min(phase_range[1], new_phase))
            chromosome['phases'][idx] = new_phase
        
        #chromosome['phases'] = np.random.uniform(phase_range[0], phase_range[1], size=(chromosome['n'] - 1,))
    return chromosome

# Evolutionary loop
def evolutionary_algorithm(pop_size: int, generations: int, n_range: Tuple[int, int], period_range: Tuple[float, float], amplitude_range: Tuple[float, float], phase_range: Tuple[float, float], mutation_rate: float, top_k: int, retain_rate, rng, model, params):
    population = initialize_population(pop_size, n_range, period_range, amplitude_range, phase_range)
    best_fitness_log = []
    print("Starting evolutionary algorithm")
    for generation in range(generations):
        # Evaluate fitness
        #rng, key = jax.random.split(rng)
        fitness_scores = [calculate_fitness(individual, key, model, params, num_evaluation_samples = 1) for individual in population]
        
        # Log best fitness for this generation
        best_fitness_log.append(max(fitness_scores))
        print(f"Generation {generation}: Best fitness - {max(fitness_scores)}")
        #average fitness
        print(f"Generation {generation}: Average fitness - {jnp.mean(jnp.array(fitness_scores))}")

        # Select parents based on fitness
        #normalize fitness scores to 0, 1
        fitness_scores = np.array(fitness_scores)
        #fitness_scores = (fitness_scores - fitness_scores.min()) / (fitness_scores.max() - fitness_scores.min())
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Retain a certain percentage of the top performers
        num_retained = int(pop_size * retain_rate)
        retained_population = sorted_population[:num_retained]
        
        # Breed the rest
        num_breed = pop_size - num_retained
        parents = random.choices(sorted_population, weights=sorted_fitness_scores, k=num_breed)
        next_generation = [crossover(random.choice(parents), random.choice(parents)) for _ in range(num_breed)]
        
        # # Create next generation through crossover
        # next_generation = [crossover(random.choice(parents), random.choice(parents)) for _ in range(pop_size)]
        
        # Apply mutation
        next_generation = [mutate(individual, n_range, period_range, amplitude_range, phase_range, mutation_rate) for individual in next_generation]
        
        population = retained_population + next_generation
    
    # Evaluate fitness for the final population
    fitness_scores = [calculate_fitness(individual, key, model, params, num_evaluation_samples=1) for individual in population]
    
    # Get the top_k best individuals
    best_indices = np.argsort(fitness_scores)[-top_k:]
    best_individuals = [population[i] for i in best_indices]
    best_fitnesses = [fitness_scores[i] for i in best_indices]
    
    return best_individuals, best_fitness_log, best_fitnesses

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


class Fourier_range(nn.Module):
    n: int
    period_range: Tuple[float, float]
    amplitude_range: Tuple[float, float]
    phase_range: Tuple[float, float]

    def setup(self):
        
        self.period = self.param(
            'period', 
            nn.initializers.uniform(scale=self.period_range[1] - self.period_range[0]), 
            ()
        ) + self.period_range[0]

        self.amplitude = self.param(
            'amplitude', 
            nn.initializers.uniform(scale=self.amplitude_range[1] - self.amplitude_range[0]), 
            (self.n,)
        ) + self.amplitude_range[0]
        
        self.phase = self.param(
            'phase', 
            nn.initializers.uniform(scale=self.phase_range[1] - self.phase_range[0]), 
            (self.n - 1,)
        ) + self.phase_range[0]


    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:
        shift = self.param(
            'shift',
            nn.initializers.uniform(scale=jnp.pi),
            jnp.shape(x), x.dtype
        )
        x = x - shift
        amplitude = self.amplitude * jnp.arange(1, len(self.amplitude) + 1) / (self.n - 1)
        waves = jnp.cos(
            (2 * jnp.pi * jnp.arange(1, self.n) * x - self.phase) / self.period
        )

        return amplitude[0] / 2.0 + jnp.sum(amplitude[1:] * waves, axis=0)
    

def posterior_loss(
    params: flax.typing.VariableDict,
    batch,
    key: flax.typing.PRNGKey,
):
    key_data, key_model = jax.random.split(key)
    


    X = batch[0]
    y = batch[1]
    x_test = batch[2]
    y_test = batch[3]
    # Compute ELBO over batch of datasets
    elbos = jax.vmap(
    partial(
            model.apply,
            params,  
            beta=kl_penalty,
            k=num_posterior_mc,
            method=model.elbo
    ) 
    )(
        X, y, x_test, y_test, rngs={'default': jax.random.split(key_model, X.shape[0])}
    )
    
    return -elbos.mean()

@jax.jit
def step(
    theta: flax.typing.VariableDict, 
    opt_state: optax.OptState,
    current_batch,
    random_key: flax.typing.PRNGKey,
) -> tuple[flax.typing.VariableDict, optax.OptState, jax.Array]:
    # Implements a generic SGD Step
    
    # value, grad = jax.value_and_grad(posterior_loss_filtered, argnums=0)(theta, random_key)
    value, grad = jax.value_and_grad(posterior_loss, argnums=0)(theta, current_batch, random_key )
    
    updates, opt_state = optimizer.update(grad, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    
    return theta, opt_state, value


def body_batch(carry, batch):
    params, opt_state, key = carry
    key_carry, key_step = jax.random.split(key)
    # Unpack the batch
    X_full, y_full = batch

    # Shuffle the data while preserving (X, y) pairs
    num_samples = X_full.shape[1]
    indices = jax.random.permutation(key_step, num_samples)
    X_full_shuffled = jnp.take(X_full, indices, axis=1)
    y_full_shuffled = jnp.take(y_full, indices, axis=1)

    # Split the shuffled data into context and test sets
    X, x_test = jnp.split(X_full_shuffled, indices_or_sections=(num_context_samples,), axis=1)
    y, y_test = jnp.split(y_full_shuffled, indices_or_sections=(num_context_samples,), axis=1)

    params, opt_state, value = step(params, opt_state, (X,y, x_test,y_test ), key_step )

    return (params, opt_state, key_carry ), value

@jax.jit
def scan_train(params, opt_state, key,  batches):
    
    last, out = jax.lax.scan(body_batch, (params, opt_state, key ), batches)

    params, opt_state, _ = last
    
    return params, opt_state, out


def create_evol_dataset(best_solutions, best_fitnesses, dataset_size, evol_rate, rng):
    samplers = []
    # convert best_fitnesses to probabilities proportional to fitnesses
    total_fitness = sum(best_fitnesses)
    ratios = [(fitness / total_fitness) * evol_rate for fitness in best_fitnesses]
    

    for chromosome in best_solutions:
        fourier_instance = Fourier_nr(
        n=chromosome['n'],
        period=chromosome['period'],
        amplitude=jnp.array(chromosome['amplitudes']),
        phase=jnp.array(chromosome['phases'])
    )
        samplers.append(partial(joint, fourier_instance, partial(uniform, n=num_target_samples + num_context_samples, bounds=(-1, 1))))
    samplers += samplers_def
    ratios_def = [(1 - evol_rate) / (len(samplers_def)) for _ in samplers_def[:len(samplers_def)]]
    ratios += ratios_def
    ratios[-1] = 1 - sum(ratios[0: len(ratios) - 1])
    rng, dataset_key = jax.random.split(rng)
    chunk_size = 64*100
    dataset = RegressionDataset(generate_noisy_split_trainingdata(samplers, ratios, dataset_size, chunk_size , dataset_key))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def recreate_dataset(rng, model, params):
    best_solutions, best_fitness_log, best_fitnesses = evolutionary_algorithm(
        pop_size, generations, (3,6), (0.05, 2.0), (0.1, 10.0), (0, jnp.pi),
        mutation_rate, top_k, retain_rate, rng, model, params
    )
    dataloader = create_evol_dataset(best_solutions, best_fitnesses, 10000, 0.3, rng)
    return dataloader


period_range=[0.05, 2.0]
amplitude_range=[0.1, 10.0]
phase_range=[0.0, jnp.pi]
n_range = [3, 6]

fouriers = [Fourier_range(n=i, period_range=period_range, amplitude_range=amplitude_range, phase_range=phase_range) for i in range(n_range[0], n_range[1] + 1)]
samplers_def = [partial(joint, fourier, partial(uniform, n=num_target_samples + num_context_samples, bounds=(-1, 1))) for fourier in fouriers]
#equal ratios
ratios_def = [1.0 / (len(fouriers)) for _ in fouriers[:len(fouriers)]]

def perform_evaluation(params_new, eval_epoch_key, samplers=samplers_def, ratios=ratios_def, eval_dataset_size=1000, chunk_size=64*100):
    eval_epoch_key, eval_inkey_data, eval_outkey_data, eval_model_key = jax.random.split(eval_epoch_key, 4)
    intask_x_eval, intask_y_eval = generate_noisy_split_trainingdata(samplers, ratios, eval_dataset_size, chunk_size, eval_inkey_data)
    intask_x_eval, intask_y_eval = intask_x_eval[..., None], intask_y_eval[..., None]

    x_contexts, x_targets = jnp.split(intask_x_eval, indices_or_sections=(num_context_samples,), axis=1)
    y_contexts, y_targets = jnp.split(intask_y_eval, indices_or_sections=(num_context_samples,), axis=1)

    ece_errors = nk.jax.vmap_chunked(partial(cross_entropy_error, model, params_new, k=num_posterior_mc), in_axes=(0, 0, 0, 0, 0), chunk_size=chunk_size)(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(eval_model_key, eval_dataset_size))
    rmse_errors = nk.jax.vmap_chunked(partial(RMSE_means, model, params_new, k=num_posterior_mc), in_axes=(0, 0, 0, 0, 0), chunk_size=chunk_size)(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(eval_model_key, eval_dataset_size))

    in_task_errors['ece'].append((ece_errors.mean(), ece_errors.std() / np.sqrt(ece_errors.shape[0])))
    in_task_errors['rmse'].append((rmse_errors.mean(), rmse_errors.std() / np.sqrt(rmse_errors.shape[0])))
    eval_params["eval_point_model_params"].append(params_new)

from tqdm import tqdm

save_path = "./evo_deneme10/"
model_name = "base_0_1_"
os.makedirs(save_path, exist_ok=True)
eval_intervals = 512
eval_dataset_size = 1000
evolution_interval = 4096

# Example usage
n_range = (3, 6)
period_range = (0.05, 2.0)
amplitude_range = (0.1, 10.0)
phase_range = (0.0, np.pi)
mutation_rate = 0.20
pop_size = 75
generations = 10
top_k = 75
retain_rate = 0.35
batch_size = 64
key = rng = jax.random.key(0)


model, params = create_model(key)
optimizer = optax.chain(
    optax.clip(.1),
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=1e-3, weight_decay=1e-6),
)
opt_state = optimizer.init(params)

best, best_params = jnp.inf, params

training_steps = 0
losses = []
best = float('inf')
best_params = None
eval_params = {"eval_point_model_params": []}
in_task_errors = {'ece': [], 'rmse': []}
out_task_errors = []
total_training_samples = 64000

import pickle
#import save_model_params



pbar = tqdm(total=total_training_samples, desc='Training Progress')
train_till_eval = 0
while training_steps < total_training_samples:
    rng, key = jax.random.split(rng)
    _, eval_epoch_key = jax.random.split(rng)
    dataloader = recreate_dataset(rng, model, params)
    for batch in dataloader:
        if training_steps >= total_training_samples:
            break
    
        batcha = jnp.asarray(jax.tree_util.tree_map(lambda tensor: tensor.numpy(), [batch]))

        if (train_till_eval >= eval_intervals) and (training_steps != 0):
            perform_evaluation(params, eval_epoch_key)
            train_till_eval %= eval_intervals
        
        params, opt_state, loss_arr = scan_train(params, opt_state, key, batcha)
        training_steps += len(batch[0])
        train_till_eval += len(batch[0])
        losses.extend(loss_arr)

        if loss_arr.min() < best:
            best = loss_arr.min()
            best_params = params

        if jnp.isnan(loss_arr).any():
            break

        pbar.update(len(batch[0]))

        pbar.set_description(f'Optimizing params. Loss: {loss_arr.min():.4f}')
        if (training_steps % evolution_interval == 0) and (training_steps != 0):
            print("girdi: ", len(losses))
            break

    save_model_params(best_params, save_path, model_name)
    with open(os.path.join(save_path, model_name + '_eval_point_params.pkl'), 'wb') as f:
        pickle.dump(eval_params, f)
    with open(os.path.join(save_path, model_name + '_training_metrics.pkl'), 'wb') as f:
        pickle.dump({"training_loss": losses, "training_intask_errors": in_task_errors, "training_outtask_errors": out_task_errors}, f)

pbar.close()
