import multiprocessing as mp
import os
from itertools import product

NUM_WORKERS = 8
NUM_GPUS = 1

datasets = ["SC12", "SC20", "SC35"] * 3
models = ["S", "L"]

def make_command(i, dataset, model):
    batch_size = 64
    acc_grad = 2

    gpu_device = i % NUM_GPUS
    logdir = f"{dataset}_Audiomer-{model}"
    command = f'''sleep {i}; CUDA_VISIBLE_DEVICES={gpu_device} python3 experiments.py --dataset {dataset} --model {model} --batch_size {batch_size} --val_check_interval 0.33 --gpus 1 --num_workers {NUM_WORKERS} --pin_memory=True --logdir lightning_logs/{logdir}/{i}/ --accumulate_grad_batches {acc_grad}'''
    return command

def run(command):
    os.system(command)

commands = [make_command(i, ds, model) for i, (ds, model) in enumerate(product(datasets, models))]

# Main Table results
with mp.Pool(2 * NUM_GPUS) as p:
    p.map(run, commands)

####################### ABLATIONS ###########################
surgery_level_to_args = {
    1: " --no_se ",
    2: " --no_se  --unequal_strides ",
    3: " --no_se  --unequal_strides  --no_residual ",
    4: " --no_se  --unequal_strides  --no_residual  --no_attention ",    
    5: " --no_se  --no_attention ",    
}

def make_command_ablation(i, surgery_level):
    model = "S"
    batch_size = 64
    acc_grad = 2

    gpu_device = i % NUM_GPUS
    logdir = f"surgery{surgery_level}"
    command = f'''sleep {i}; CUDA_VISIBLE_DEVICES={gpu_device} python3 experiments.py --dataset SC12 --model {model} --batch_size {batch_size} --val_check_interval 0.33 --gpus 1 --num_workers {NUM_WORKERS} --pin_memory=True --logdir lightning_logs/{logdir}/{i}/ --accumulate_grad_batches {acc_grad}'''
    command = f"{command} {surgery_level_to_args[surgery_level]}"
    print(command)
    return command

surgery_levels = [1, 2, 3, 4, 5] * 3
commands = [make_command_ablation(i, l) for i, l in enumerate(surgery_levels)]

with mp.Pool(2 * NUM_GPUS) as p:
    p.map(run, commands)
