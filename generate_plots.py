# %%
import pickle
from argparse import Namespace
from copy import deepcopy
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tianshou
import torch
from causal_ccm.causal_ccm import ccm
from pettingzoo.mpe import simple_tag_v2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.data.experimental import ExperimentFromDev
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    PPOPolicy,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from utils import *
from viz_config import *

TAU = 1  # Sensible default
E = 2  # To vary
ENV = simple_tag_v2
ENV_INSTANCE = get_env(simple_tag_v2)
NAMES = ENV_INSTANCE.agents
COLORS = [
    "tab:orange",
    "tab:purple",
    "tab:green",
    "tab:red",
    "tab:cyan",
    "tab:brown",
    "tab:pink",
]
STEPS_PER_EPOCH = 1000
# EPISODE_PER_UPDATE = 10  # translates to 1000 steps per update
N_EPOCHS = 1000
N_EPOCHS_BETWEEN_CPS = 100
SLICE_SIZE = STEPS_PER_EPOCH * N_EPOCHS_BETWEEN_CPS
NUM_SLICES = (N_EPOCHS // N_EPOCHS_BETWEEN_CPS) + 1
N_EP_EVAL = 10


def get_rewards(archs: list, seeds: list, save_loc=None):
    """Get rewards stored inside TFEvents in Tensorboard

    Args:
        archs (list): List of architectures to retrieve
        seeds (list): List of seeds to retrieve
        save_loc (str, optional): save location of the resulting dict. Defaults to None.

    Returns:
        dict: dictionary holding rewards for differents architectures and seeds
    """
    rewards = {}

    for arch in archs:
        rewards[arch] = {}
        for seed in seeds:
            # Get TFEvents store with Tensorboard
            rewards[arch][seed] = []
            path = f"log/tag/ppo/{arch}/{seed}"
            event_acc = EventAccumulator(path)
            event_acc.Reload()

            # Iterate through them, select ones that match with SLICE_SIZE
            # to be coherent with number of CCM values.
            for event in event_acc.Scalars("test/reward"):
                # Event steps increment by the number of agents each time
                # So we need need to take the first slice every time to get
                # a step that lands on an evaluation step in the training pipeline.
                # e.g. 4 agents means every single loop of the agents is 4 steps and
                # not 1 step, although only one frame of the game is played
                if (event.step / len(NAMES)) % SLICE_SIZE == 0:
                    rewards[arch][seed].append(event.value)

    # Save rewards if needed
    if save_loc is not None:
        with open(save_loc, "wb") as f:
            pickle.dump(rewards, f)
    return rewards


def get_ccms(archs: list, seeds: list, save_loc=None):
    """Compute Convergent Cross-Mapping (CCM) for given architectures and seeds. Requires pretrained models.

    Args:
        archs (list): List of architectures to retrieve
        seeds (list): List of seeds to retrieve
        save_loc (str, optional): save location of the resulting dict. Defaults to None.

    Returns:
        dict: Dictionary containing CCMs for the given architectures and seeds
    """
    ccms = {}
    possible_random_agents = [0, 1, 2]

    for arch in archs:
        ccms[arch] = {}

        for seed in seeds:
            # Retrieve arguments used for the targeted run
            path = f"log/tag/ppo/{arch}/{seed}"
            event_acc = EventAccumulator(path)
            event_acc.Reload()
            args = eval(
                event_acc.Tensors("args/text_summary")[0].tensor_proto.string_val[0]
            )
            # To load on device without GPU
            args.device = "cpu"

            for random_agent in possible_random_agents:
                assert (
                    random_agent < 4
                )  # Constraint for simple_tag_v2 environment. 4th agent is prey.

                # Get remaning agents that we can evaluate on CCM
                remaining_adv = deepcopy(possible_random_agents)
                remaining_adv.remove(random_agent)

                # Retrieve available checkpoints in directory
                checkpoints_path = f"log/tag/ppo/cp/{arch}/{seed}/"
                avail_checkpoints = sorted(
                    os.listdir(checkpoints_path), key=lambda x: int(x.split("=")[1])
                )
                avail_checkpoints_paths = [
                    checkpoints_path + checkpoint_name
                    for checkpoint_name in avail_checkpoints
                ]

                # Iterate through checkpoints, loading agents and setting one as random policy
                # then, collect episodes to compute CCM
                for current_checkpoint_path in avail_checkpoints_paths:
                    args.resume_path = current_checkpoint_path

                    # Load agent, set one as random
                    policy, optim, agents, _ = get_agents(
                        ENV, args, override_agent=[random_agent]
                    )

                    # Setup data collection with Tianshou
                    test_env = DummyVectorEnv([lambda: get_env(ENV) for i in range(10)])
                    replay_buffer = VectorReplayBuffer(20_000, len(test_env))
                    collector = Collector(policy, test_env, replay_buffer)

                    ep = collector.collect(n_episode=N_EP_EVAL)  #

                    data = replay_buffer.sample(0)[0]
                    actions = {}

                    # Attribute actions to proper agent so CCM can construct proper manifolds
                    for name in NAMES:
                        agent_indices = data["obs"]["agent_id"] == name
                        actions[name] = data["act"][agent_indices]

                    L = len(actions[NAMES[0]])

                    # Iterate through remaining adversaries that aren't random
                    # compute CCM for each of them for the current random agent.
                    for adv in remaining_adv:
                        adv_name = f"adversary_{adv}"
                        save_name = f"adversary_random_{random_agent}_on_{adv_name}"

                        # Give basic structure to dict
                        if save_name not in ccms[arch].keys():
                            ccms[arch][save_name] = {}
                            for s in seeds:
                                ccms[arch][save_name][s] = {}
                                ccms[arch][save_name][s]["correl"] = []
                                ccms[arch][save_name][s]["p-value"] = []

                        # Compute and store ccm
                        ccm1 = ccm(
                            actions[f"adversary_{random_agent}"],
                            actions[adv_name],
                            TAU,
                            E,
                            L,
                        )
                        correl, p = ccm1.causality()
                        ccms[arch][save_name][seed]["correl"].append(correl)
                        ccms[arch][save_name][seed]["p-value"].append(p)

    # Save ccms if needed
    if save_loc is not None:
        with open(save_loc, "wb") as f:
            pickle.dump(ccms, f)

    return ccms


# def retrieve_key_data_from_dict(dic: dict, target_key: str):
#     """Recursively walk throught dict and return target_key's values across the dict
#     Args:
#         dic (dict): dict to walk
#         target_key (str): name of the key where value of interest is stored

#     Returns:
#         list: list of values of interest
#     """
#     res = []
#     current_keys = dic.keys()
#     if target_key in current_keys:
#         return dic[target_key]

#     for key in current_keys:
#         res.append(retrieve_key_data_from_dict(dic[key], target_key))

#     pprint(res)
#     return res


def plot_ccms(ccms):
    fig, ax = plt.subplots(1, 1)
    x_axis = [f"{SLIC_SIZE * i:.2e}" for i in range(1, NUM_SLICES)]
    for i, arch in enumerate(ccms.keys()):
        correls = []
        for save_name in ccms[arch].keys():
            for seed in ccms[arch][save_name].keys():
                correls.append(ccms[arch][save_name][seed]["correl"])

        arch_label = arch.replace("_", "x")

        # Compute values to show
        correl_mean = np.nanmean(correls, axis=0)
        correl_std = np.nanstd(correls, axis=0)

        # Plot
        ax.fill_between(
            x_axis,
            correl_mean - correl_std,
            correl_mean + correl_std,
            alpha=0.3,
            color=COLORS[i],
        )
        ax.plot(x_axis, correl_mean, label=arch_label, color=COLORS[i])
        ax.set_xlabel("Number of environment interactions")
        ax.set_ylabel("Convergent X-mapping")
        ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.legend()
    fig.savefig("test_ccms.png")


def plot_rewards(rewards_dict):
    fig, ax = plt.subplots(1, 1)
    x_axis = [f"{SLICE_SIZE * i:.2e}" for i in range(1, NUM_SLICES + 1)]

    for i, arch in enumerate(rewards_dict.keys()):
        rewards_list = []

        for seed in rewards_dict[arch].keys():
            rewards_list.append(rewards_dict[arch][seed])

        arch_label = arch.replace("_", "x")

        # Compute values to show
        reward_mean = np.nanmean(rewards_list, axis=0)
        reward_std = np.nanstd(rewards_list, axis=0)

        # Plot
        ax.fill_between(
            x_axis,
            reward_mean - reward_std,
            reward_mean + reward_std,
            alpha=0.3,
            color=COLORS[i],
        )
        ax.plot(x_axis, reward_mean, label=arch_label, color=COLORS[i])
        ax.set_xlabel("Number of environment interactions")
        ax.set_ylabel("Obtained reward")
        ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.legend()
    fig.savefig("test_rewards.png")


if __name__ == "__main__":
    # Variables
    archs = ["64_64_64_64"]
    # seeds = [1, 2, 3, 4, 5]
    seeds = [1, 2, 3, 4, 5]
    # ccms = get_ccms(archs, seeds, save_loc="test_ccms.pkl")

    # with open("test_ccms.pkl", "rb") as f:
    #     ccms = pickle.load(f)
    # plot_ccms(ccms)

    rewards = get_rewards(archs, seeds)
    plot_rewards(rewards)
    # retrieve_key_data_from_dict(ccms, "correl")
