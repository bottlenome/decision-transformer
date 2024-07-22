import logging
# make deterministic
from .mingpt.utils import set_seed
import numpy as np
import torch
from torch.utils.data import Dataset
from .mingpt.model_atari import GPT, GPTConfig
from .mingpt.trainer_atari import Trainer, TrainerConfig
from .mingpt.utils import sample
import torch
import argparse
from .create_dataset import create_dataset, StateActionReturnDataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--context_length', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--model_name', type=str, default='gpt')
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
    args = parser.parse_args()

    set_seed(args.seed)


    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

    # set up logging
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )

    train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=6, n_head=8, n_embd=128,
                    model_type=args.model_type, model_name=args.model_name, max_timestep=max(timesteps))
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    epochs = args.epochs
    tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                        num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()