import argparse
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from supersuit import pad_observations_v0
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

# from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter


class TagNet(Net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        # Because Tianshou gives this bad solution to a problem...
        # obs = obs.to_torch(device=self.device)
        obs = obs.obs
        return Net.forward(self, obs, state, info)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])

    parser.add_argument(
        "--training-num", type=int, default=16, help="Number of training envs"
    )
    parser.add_argument(
        "--test-num", type=int, default=100, help="Number of testing envs"
    )
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument("--repeat-per-collect", type=int, default=2)

    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )

    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--step-per-epoch", type=int, default=150000)

    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--episode-per-collect", type=int, default=16)

    # DQN
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)

    # ppo special
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--rew-norm", type=int, default=1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    args = parser.parse_known_args()[0]

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents(
    env,
    args: argparse.Namespace = get_args(),
    agents: Optional[Tuple[BasePolicy]] = None,
    optims: Optional[Tuple[Optimizer]] = None,
    override_agent: Optional[Tuple[int]] = None,
) -> Tuple[BasePolicy, Optimizer, list]:
    env = get_env(env)
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    if agents is None:
        agents = []
        optims = []

        for _ in range(env.num_agents):
            net = TagNet(
                args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device
            ).to(args.device)

            actor = Actor(
                net,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            critic = Critic(
                deepcopy(net), hidden_sizes=args.hidden_sizes, device=args.device
            ).to(args.device)

            actor_critic = ActorCritic(actor, critic)

            # orthogonal initialization
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            dist_fn = torch.distributions.Categorical

            optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

            agent = PPOPolicy(
                actor,
                critic,
                optim,
                dist_fn,
                discount_factor=args.gamma,
                max_grad_norm=args.max_grad_norm,
                eps_clip=args.eps_clip,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                reward_normalization=args.rew_norm,
                advantage_normalization=args.norm_adv,
                recompute_advantage=args.recompute_adv,
                dual_clip=args.dual_clip,
                value_clip=args.value_clip,
                gae_lambda=args.gae_lambda,
                action_space=env.action_space,
            )

            agents.append(agent)
            optims.append(optim)

    if args.resume_path:
        # Only works because save names are indeed ordered properly
        for i, name in enumerate(env.agents):
            agents[i].load_state_dict(
                torch.load(args.resume_path + f"/{name}.pth", map_location=args.device)
            )

    if override_agent is not None:
        for i in override_agent:
            agents[i] = RandomPolicy()

    policy = MultiAgentPolicyManager(agents, env)
    n_params_net = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return policy, optims, env.agents, n_params_net


def get_env(env, render_mode=None):
    return PettingZooEnv(
        pad_observations_v0(
            env.env(max_cycles=100, num_obstacles=0, render_mode=render_mode)
        )
    )
