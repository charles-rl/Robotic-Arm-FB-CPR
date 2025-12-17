# Based on CleanRL PPO+RND: https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/
import os
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import gymnasium as gym
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import pickle

# Import your environment
from environment import RobotArmEnv

DEBUG = False

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True  # Enable WandB by default
    wandb_project_name: str = "so101-reach-fb"
    wandb_entity: str = None
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "SO101-Reach-v0"
    total_timesteps: int = 2000000  # 1M steps
    learning_rate: float = 3e-4
    num_envs: int = 20
    num_steps: int = 50  # Steps per env per update
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0  # Entropy is handled by std dev in continuous PPO
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # RND arguments
    update_proportion: float = 0.25
    int_coef: float = 1.0  # Intrinsic reward weight (Curiosity)
    ext_coef: float = 2.0  # Extrinsic reward weight (Distance/Success)
    int_gamma: float = 0.99
    num_iterations_obs_norm_init: int = 50

    # Runtime computed
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# --- UTILS ---
class RunningMeanStd:
    # Ported from OpenAI Baselines to pure PyTorch/Numpy
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    return new_mean, new_var, tot_count


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# --- MODELS ---

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Continuous Action Space Agent
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.array(envs.single_action_space.shape).prod()

        # Critic (Shared backbone for Intrinsic and Extrinsic)
        self.critic_body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 448)),
            nn.Tanh(),
            layer_init(nn.Linear(448, 448)),
            nn.Tanh(),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=1.0)
        self.critic_int = layer_init(nn.Linear(448, 1), std=1.0)

        # Actor
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 448)),
            nn.Tanh(),
            layer_init(nn.Linear(448, 448)),
            nn.Tanh(),
            layer_init(nn.Linear(448, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        hidden = self.critic_body(x)
        return self.critic_ext(hidden), self.critic_int(hidden)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        hidden = self.critic_body(x)
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic_ext(hidden),
            self.critic_int(hidden),
        )


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # RND for Continuous States (MLP instead of CNN)

        # Predictor Network (Trainable)
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(input_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),  # Feature output
        )

        # Target Network (Fixed, Randomly Initialized)
        self.target = nn.Sequential(
            layer_init(nn.Linear(input_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
        )

        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature


# --- MAIN LOOP ---

if __name__ == "__main__":
    args = tyro.cli(Args)
    if DEBUG:
        args.num_envs = 8
        args.num_steps = 2048
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    # Env Setup: Using Standard Gymnasium SyncVectorEnv
    def make_env():
        def thunk():
            # Create your specific robot environment
            env = RobotArmEnv(render_mode=None, reward_type="sparse", task="reach")
            # Wrappers
            env = gym.wrappers.ClipAction(env)
            return env

        return thunk


    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(args.num_envs)])

    # Initialize Models
    agent = Agent(envs).to(device)
    # RND Input size = Observation dimension
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    rnd_model = RNDModel(obs_dim, 256).to(device)

    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(combined_parameters, lr=args.learning_rate, eps=1e-5)

    # RND Normalization utils
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, obs_dim))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start Game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Init Observation Normalization (Collect random steps to get mean/std)
    print("Initializing observation normalization...")
    for step in range(args.num_iterations_obs_norm_init):
        # Random actions
        acs = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        s, _, _, _, _ = envs.step(acs)
        obs_rms.update(s)
    print("Initialization done.")

    # --- TRAINING LOOP ---
    for update in range(1, args.num_iterations + 1):
        # Anneal LR
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                # Get Values and Action
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            # Step Env
            real_next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            real_next_obs = torch.Tensor(real_next_obs).to(device)
            next_obs = real_next_obs
            next_done = torch.Tensor(done).to(device)

            # --- RND CALCULATION ---
            # 1. Normalize observation for RND
            rnd_next_obs = (
                    (real_next_obs - torch.from_numpy(obs_rms.mean).to(device).float())
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device).float() + 1e-8)
            ).clip(-5, 5).float()

            # 2. Compute Intrinsic Reward (Error between Target and Predictor)
            target_feat = rnd_model.target(rnd_next_obs)
            predict_feat = rnd_model.predictor(rnd_next_obs)
            # Intrinsic reward = MSE / 2
            curiosity_rewards[step] = ((target_feat - predict_feat).pow(2).sum(1) / 2).data

            # Logging
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        # We don't have wrappers extracting 'r', using Monitor wrapper recommended usually
                        # But for now let's just log extrinsic reward raw sum
                        pass

        # Update Observation Normalization
        obs_rms.update(obs.cpu().numpy().reshape(-1, obs_dim))

        # Normalize Intrinsic Rewards (Standard RND trick)
        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = np.mean(curiosity_reward_per_env), np.std(curiosity_reward_per_env), len(
            curiosity_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # Divide by std dev to keep scale consistent
        curiosity_rewards /= np.sqrt(reward_rms.var + 1e-8)

        # Bootstrap and GAE
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards).to(device)
            int_advantages = torch.zeros_like(curiosity_rewards).to(device)

            ext_lastgaelam = 0
            int_lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0  # Intrinsic reward doesn't care about episodic done usually
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]

                # Extrinsic GAE
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                            ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam)

                # Intrinsic GAE
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                int_advantages[t] = int_lastgaelam = (
                            int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam)

            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # Flatten Batches
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, envs.single_action_space.shape[0]))
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)

        # Combine Advantages
        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # Optimizing
        b_inds = np.arange(args.batch_size)

        # Recalculate RND normalization for batch
        rnd_next_obs_batch = ((b_obs - torch.from_numpy(obs_rms.mean).to(device).float()) / torch.sqrt(
            torch.from_numpy(obs_rms.var).to(device).float() + 1e-8)).clip(-5, 5).float()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # RND Forward Loss
                predict_feat, target_feat = rnd_model(rnd_next_obs_batch[mb_inds])
                forward_loss = F.mse_loss(predict_feat, target_feat.detach(), reduction="none").mean(-1)

                # Masking (Optional logic from paper, kept simple here: update on 25% of data)
                mask = torch.rand(len(forward_loss)).to(device)
                mask = (mask < args.update_proportion).float()
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1.0]).to(device))

                # PPO Update
                new_action, new_logprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)

                # Extrinsic Value Loss
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, (ext_v_clipped - b_ext_returns[mb_inds]) ** 2)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                # Intrinsic Value Loss
                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()

                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Log
        if update % 10 == 0:
            torch.save(agent.state_dict(), f"runs/{run_name}/latest_model.pt")

            # 2. Overwrite Normalization Stats
            with open(f"runs/{run_name}/latest_obs_rms.pkl", "wb") as f:
                pickle.dump(obs_rms, f)

            print(f"Updated latest checkpoint at update {update}")
        print(
            f"Update {update}/{args.num_iterations}, Global Step: {global_step}, FPS: {int(global_step / (time.time() - start_time))}")
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
        writer.add_scalar("charts/curiosity_reward", curiosity_rewards.mean().item(), global_step)
        writer.add_scalar("charts/extrinsic_reward", rewards.mean().item(), global_step)
        writer.add_scalar("results/reward", rewards.mean().item(), global_step)

    torch.save(agent.state_dict(), f"runs/{run_name}/final_model.pt")
    with open(f"runs/{run_name}/final_obs_rms.pkl", "wb") as f:
        pickle.dump(obs_rms, f)

    envs.close()
    writer.close()
