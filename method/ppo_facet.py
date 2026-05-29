"""Small-scale PPO implementation for FACET-style impedance tracking.

The full paper trains legged policies in Isaac Lab. This module keeps the
algorithmic pieces that matter for the reduced benchmark: a policy is optimized
with PPO to track temporally smoothed impedance-reference targets, first with a
privileged feature encoder and then with a deployable estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from data.facet_planar_env import PlanarForceEnv
from method.facet_reference import ImpedanceCommand, SmoothedImpedanceReference


@dataclass
class PPOConfig:
    n_envs: int = 128
    rollout_steps: int = 96
    teacher_updates: int = 60
    student_updates: int = 30
    minibatches: int = 4
    epochs: int = 3
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    estimator_coef: float = 0.2
    max_grad_norm: float = 0.7
    lr: float = 3.0e-4
    seed: int = 7
    dt: float = 0.02


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        layers.append(activation())
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class FACETPolicy(nn.Module):
    """Asymmetric actor-critic with privileged teacher and student estimator."""

    def __init__(self, obs_dim: int, priv_dim: int, action_dim: int):
        super().__init__()
        latent_dim = 64
        self.obs_encoder = mlp(obs_dim, (128,), latent_dim)
        self.priv_encoder = mlp(priv_dim, (128,), latent_dim)
        self.estimator = mlp(obs_dim, (128,), latent_dim)
        self.actor = mlp(2 * latent_dim, (256, 256), action_dim)
        self.critic = mlp(obs_dim + priv_dim, (256, 256), 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.6))

    def latent(self, obs: torch.Tensor, priv: torch.Tensor, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
        obs_latent = self.obs_encoder(obs)
        priv_latent = self.priv_encoder(priv)
        if stage == "teacher":
            actor_latent = priv_latent
        elif stage == "student":
            actor_latent = self.estimator(obs)
        else:
            raise ValueError(f"unknown stage: {stage}")
        return torch.cat([obs_latent, actor_latent], dim=-1), priv_latent

    def distribution(self, obs: torch.Tensor, priv: torch.Tensor, stage: str) -> Normal:
        actor_input, _ = self.latent(obs, priv, stage)
        mean = self.actor(actor_input)
        return Normal(mean, self.log_std.exp())

    def value(self, obs: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([obs, priv], dim=-1)).squeeze(-1)

    def act(self, obs: torch.Tensor, priv: torch.Tensor, stage: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs, priv, stage)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        value = self.value(obs, priv)
        return action, logp, value

    def deterministic(self, obs: torch.Tensor, priv: torch.Tensor, stage: str = "student") -> torch.Tensor:
        actor_input, _ = self.latent(obs, priv, stage)
        return self.actor(actor_input)


class FACETTrainingEnv:
    """Vectorized planar task with FACET reward targets."""

    def __init__(self, n_envs: int, seed: int, device: torch.device):
        self.n_envs = n_envs
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.env = PlanarForceEnv(n_envs, mode="impulse", seed=seed)
        self.ref = SmoothedImpedanceReference(n_envs=n_envs, dt=self.env.cfg.dt)
        self.last_actions = np.zeros((n_envs, 3, 2), dtype=np.float32)
        self.command = self._sample_command(n_envs)
        self.modes = np.zeros(n_envs, dtype=np.int32)
        self.reset()

    def _sample_command(self, count: int) -> ImpedanceCommand:
        kp = self.rng.uniform(1.5, 8.0, size=count).astype(np.float32)
        target_velocity = np.zeros((count, 2), dtype=np.float32)
        target_velocity[:, 0] = self.rng.uniform(1.2, 1.8, size=count).astype(np.float32)
        return ImpedanceCommand(
            kp=kp,
            kd=2.0 * np.sqrt(kp),
            virtual_mass=np.full(count, self.env.cfg.mass, dtype=np.float32),
            target_velocity=target_velocity,
        )

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.env.reset()
        self.modes = self.rng.integers(0, 2, size=self.n_envs, dtype=np.int32)
        self.last_actions.fill(0.0)
        self.command = self._sample_command(self.n_envs)
        self.ref.reset(self.env.pos, self.env.vel)
        return self.observations(np.zeros((self.n_envs, 2), dtype=np.float32))

    def reset_indices(self, done: np.ndarray) -> None:
        idx = np.flatnonzero(done)
        if len(idx) == 0:
            return
        self.env.t[idx] = 0.0
        self.env.step_count[idx] = 0
        self.env.pos[idx] = 0.0
        self.env.vel[idx] = 0.0
        self.env.tilt[idx] = 0.0
        self.env.failed[idx] = False
        self.env.done[idx] = False
        self.env.push_peak[idx] = self.rng.uniform(320.0, 430.0, size=len(idx)).astype(np.float32)
        self.env.push_duration[idx] = self.rng.uniform(0.45, 0.55, size=len(idx)).astype(np.float32)
        self.modes[idx] = self.rng.integers(0, 2, size=len(idx), dtype=np.int32)
        command = self._sample_command(len(idx))
        self.command.kp[idx] = command.kp
        self.command.kd[idx] = command.kd
        self.command.virtual_mass[idx] = command.virtual_mass
        self.command.target_velocity[idx] = command.target_velocity
        self.last_actions[idx] = 0.0
        for pos, vel, force in self.ref.history:
            pos[idx] = 0.0
            vel[idx] = 0.0
            force[idx] = 0.0

    def external_forces(self) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros((self.n_envs, 2), dtype=np.float32)
        wall_forces = np.zeros((self.n_envs, 2), dtype=np.float32)
        impulse_mask = self.modes == 0
        if impulse_mask.any():
            tau = self.env.t[impulse_mask] - self.env.cfg.push_start
            duration = self.env.push_duration[impulse_mask]
            half = 0.5 * duration
            rising = tau / np.maximum(half, 1e-6)
            falling = (duration - tau) / np.maximum(half, 1e-6)
            scale = np.where(tau <= half, rising, falling)
            active = (tau >= 0.0) & (tau <= duration)
            forces[impulse_mask, 1] = self.env.push_peak[impulse_mask] * np.maximum(0.0, scale) * active

        wall_mask = self.modes == 1
        if wall_mask.any():
            penetration = self.env.pos[wall_mask, 0] - self.env.cfg.wall_x
            active = penetration > 0.0
            local_wall = np.zeros_like(penetration, dtype=np.float32)
            local_wall[active] = (
                -self.env.cfg.wall_kp * penetration[active]
                - self.env.cfg.wall_kd * np.maximum(self.env.vel[wall_mask, 0][active], 0.0)
            )
            wall_forces[wall_mask, 0] = local_wall
        return forces + wall_forces, wall_forces

    def observations(self, force_ext: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.env.t[:, None]
        phase = np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)
        command = np.column_stack(
            [
                self.command.target_velocity[:, 0],
                self.command.target_velocity[:, 1],
                self.command.kp / 40.0,
                self.command.kd / 20.0,
                self.command.virtual_mass / 20.0,
            ]
        ).astype(np.float32)
        deploy = np.concatenate(
            [
                self.env.vel,
                self.env.tilt[:, None],
                phase,
                command,
                self.last_actions.reshape(self.n_envs, -1),
            ],
            axis=1,
        )
        pos_targets, vel_targets, acc_targets = self.ref.targets(self.env.pos, force_ext, self.command)
        ref_pos = pos_targets.mean(axis=0)
        ref_vel = vel_targets.mean(axis=0)
        ref_acc = acc_targets.mean(axis=0)
        priv = np.concatenate(
            [
                force_ext / 400.0,
                ref_pos - self.env.pos,
                ref_vel - self.env.vel,
                ref_acc / 40.0,
            ],
            axis=1,
        ).astype(np.float32)
        return (
            torch.as_tensor(deploy, dtype=torch.float32, device=self.device),
            torch.as_tensor(priv, dtype=torch.float32, device=self.device),
        )

    def step(self, action_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        action = action_t.detach().cpu().numpy().astype(np.float32)
        action = np.clip(action, -self.env.cfg.action_limit, self.env.cfg.action_limit)
        force_ext, _ = self.external_forces()
        self.env.mode = "impulse"
        # Inline the environment step so mixed impulse/wall modes can share one
        # vectorized state array while preserving PlanarForceEnv dynamics.
        acc = action + force_ext / self.env.cfg.mass - self.env.cfg.damping * self.env.vel
        self.env.vel += acc * self.env.cfg.dt
        self.env.pos += self.env.vel * self.env.cfg.dt
        lateral_reaction = np.abs(action[:, 1])
        self.env.tilt = 0.93 * self.env.tilt + 0.03 * lateral_reaction + 0.0002 * np.abs(force_ext[:, 1])
        new_fail = (lateral_reaction > self.env.cfg.fall_reaction_threshold) & (np.abs(force_ext[:, 1]) > 80.0)
        new_fail |= self.env.tilt > self.env.cfg.fall_tilt_threshold
        self.env.failed |= new_fail
        self.env.step_count += 1
        self.env.t += self.env.cfg.dt
        done_np = (self.env.step_count >= self.env.cfg.episode_steps) | self.env.failed

        next_force, _ = self.external_forces()
        pos_targets, vel_targets, _ = self.ref.targets(self.env.pos, next_force, self.command)
        ref_pos = pos_targets.mean(axis=0)
        ref_vel = vel_targets.mean(axis=0)
        pos_err = np.sum((self.env.pos - ref_pos) ** 2, axis=1)
        vel_err = np.sum((self.env.vel - ref_vel) ** 2, axis=1)
        forward_err = (self.env.vel[:, 0] - self.env.cfg.target_vx) ** 2
        reward_np = (
            1.2 * np.exp(-pos_err / 0.25)
            + 1.8 * np.exp(-vel_err / 0.25)
            - 0.03 * np.sum(action * action, axis=1)
            - 0.08 * forward_err
            - 2.0 * self.env.failed.astype(np.float32)
        ).astype(np.float32)

        self.ref.push_state(self.env.pos, self.env.vel, next_force)
        self.last_actions = np.roll(self.last_actions, shift=-1, axis=1)
        self.last_actions[:, -1, :] = action
        obs, priv = self.observations(next_force)
        done = torch.as_tensor(done_np.astype(np.float32), device=self.device)
        reward = torch.as_tensor(reward_np, device=self.device)
        reset_count = int(done_np.sum())
        if reset_count:
            self.reset_indices(done_np)
            obs, priv = self.observations(np.zeros((self.n_envs, 2), dtype=np.float32))
        info = {
            "mean_reward": float(reward_np.mean()),
            "failure_rate": float(self.env.failed.mean()),
            "reset_count": float(reset_count),
        }
        return obs, priv, reward, done, info


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(steps)):
        next_nonterminal = 1.0 - dones[t]
        next_values = next_value if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train_policy(output_dir: Path, config: PPOConfig) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = FACETTrainingEnv(config.n_envs, config.seed, device)
    obs, priv = runner.reset()
    policy = FACETPolicy(obs.shape[-1], priv.shape[-1], 2).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training FACET PPO on {device} with {config.n_envs} envs, "
        f"{config.teacher_updates} teacher updates and {config.student_updates} student updates",
        flush=True,
    )

    def run_stage(stage: str, updates: int) -> None:
        nonlocal obs, priv
        batch_size = config.rollout_steps * config.n_envs
        mini_size = batch_size // config.minibatches
        for update in range(updates):
            obs_buf = []
            priv_buf = []
            action_buf = []
            logp_buf = []
            reward_buf = []
            done_buf = []
            value_buf = []
            for _ in range(config.rollout_steps):
                with torch.no_grad():
                    action, logp, value = policy.act(obs, priv, stage)
                next_obs, next_priv, reward, done, _ = runner.step(action)
                obs_buf.append(obs)
                priv_buf.append(priv)
                action_buf.append(action)
                logp_buf.append(logp)
                reward_buf.append(reward)
                done_buf.append(done)
                value_buf.append(value)
                obs, priv = next_obs, next_priv
            with torch.no_grad():
                next_value = policy.value(obs, priv)
            obs_b = torch.stack(obs_buf)
            priv_b = torch.stack(priv_buf)
            actions_b = torch.stack(action_buf)
            old_logp_b = torch.stack(logp_buf)
            rewards_b = torch.stack(reward_buf)
            dones_b = torch.stack(done_buf)
            values_b = torch.stack(value_buf)
            adv_b, ret_b = compute_gae(
                rewards_b, dones_b, values_b, next_value, config.gamma, config.gae_lambda
            )
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            flat_obs = obs_b.reshape(batch_size, -1)
            flat_priv = priv_b.reshape(batch_size, -1)
            flat_actions = actions_b.reshape(batch_size, -1)
            flat_old_logp = old_logp_b.reshape(batch_size)
            flat_adv = adv_b.reshape(batch_size)
            flat_ret = ret_b.reshape(batch_size)
            indices = torch.randperm(batch_size, device=device)
            for _ in range(config.epochs):
                for start in range(0, batch_size, mini_size):
                    mb = indices[start : start + mini_size]
                    dist = policy.distribution(flat_obs[mb], flat_priv[mb], stage)
                    new_logp = dist.log_prob(flat_actions[mb]).sum(dim=-1)
                    ratio = (new_logp - flat_old_logp[mb]).exp()
                    unclipped = ratio * flat_adv[mb]
                    clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * flat_adv[mb]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(policy.value(flat_obs[mb], flat_priv[mb]), flat_ret[mb])
                    entropy = dist.entropy().sum(dim=-1).mean()
                    _, teacher_latent = policy.latent(flat_obs[mb], flat_priv[mb], "teacher")
                    estimator_loss = F.mse_loss(policy.estimator(flat_obs[mb]), teacher_latent.detach())
                    # The estimator is trained during the teacher stage, then
                    # PPO fine-tunes the student with the estimated feature.
                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy
                        + config.estimator_coef * estimator_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                    optimizer.step()
            if (update + 1) % 10 == 0 or update == updates - 1:
                print(
                    f"{stage} update {update + 1}/{updates}: "
                    f"reward={float(rewards_b.mean()):.3f} done={float(dones_b.mean()):.3f}",
                    flush=True,
                )
                torch.save(
                    {
                        "model": policy.state_dict(),
                        "config": config.__dict__,
                        "obs_dim": flat_obs.shape[-1],
                        "priv_dim": flat_priv.shape[-1],
                    },
                    output_dir / "facet_student.pt",
                )

    run_stage("teacher", config.teacher_updates)
    run_stage("student", config.student_updates)
    checkpoint = output_dir / "facet_student.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "config": config.__dict__,
            "obs_dim": obs.shape[-1],
            "priv_dim": priv.shape[-1],
        },
        checkpoint,
    )
    return checkpoint


def load_policy(checkpoint: Path, device: torch.device) -> FACETPolicy:
    payload = torch.load(checkpoint, map_location=device)
    policy = FACETPolicy(payload["obs_dim"], payload["priv_dim"], 2).to(device)
    policy.load_state_dict(payload["model"])
    policy.eval()
    return policy


def policy_rollout(mode: str, episodes: int, seed: int, kp: float, name: str, checkpoint: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PlanarForceEnv(episodes, mode=mode, seed=seed)
    ref = SmoothedImpedanceReference(n_envs=episodes, dt=env.cfg.dt)
    command = ImpedanceCommand(
        kp=kp,
        kd=2.0 * np.sqrt(kp),
        virtual_mass=env.cfg.mass,
        target_velocity=np.array([env.cfg.target_vx, 0.0], dtype=np.float32),
    )
    policy = load_policy(checkpoint, device)
    env.reset()
    ref.reset(env.pos, env.vel)
    last_actions = np.zeros((episodes, 3, 2), dtype=np.float32)
    records = []
    wall_impulse = np.zeros(episodes, dtype=np.float32)

    def make_obs(force_ext: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        t = env.t[:, None]
        phase = np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)
        command_arr = np.tile(
            np.array(
                [command.target_velocity[0], command.target_velocity[1], kp / 40.0, command.kd / 20.0, env.cfg.mass / 20.0],
                dtype=np.float32,
            ),
            (episodes, 1),
        )
        deploy = np.concatenate(
            [env.vel, env.tilt[:, None], phase, command_arr, last_actions.reshape(episodes, -1)],
            axis=1,
        )
        pos_targets, vel_targets, acc_targets = ref.targets(env.pos, force_ext, command)
        priv = np.concatenate(
            [
                np.zeros_like(force_ext),
                pos_targets.mean(axis=0) - env.pos,
                vel_targets.mean(axis=0) - env.vel,
                acc_targets.mean(axis=0) / 40.0,
            ],
            axis=1,
        ).astype(np.float32)
        return torch.as_tensor(deploy, dtype=torch.float32, device=device), torch.as_tensor(
            priv, dtype=torch.float32, device=device
        )

    for _ in range(env.cfg.episode_steps):
        force_ext, _ = env.external_forces()
        obs_t, priv_t = make_obs(force_ext)
        with torch.no_grad():
            action = policy.deterministic(obs_t, priv_t, "student").cpu().numpy().astype(np.float32)
        _, _, done, info = env.step(action)
        wall_impulse += np.abs(info["wall_force"][:, 0]) * env.cfg.dt
        ref.push_state(env.pos, env.vel, info["force_ext"])
        last_actions = np.roll(last_actions, shift=-1, axis=1)
        last_actions[:, -1, :] = np.clip(action, -env.cfg.action_limit, env.cfg.action_limit)
        records.append(
            {
                "t": info["time"].tolist(),
                "pos": env.pos.tolist(),
                "vel": env.vel.tolist(),
                "action": action.tolist(),
                "force_ext": info["force_ext"].tolist(),
                "wall_force": info["wall_force"].tolist(),
                "failed": info["failed"].astype(int).tolist(),
            }
        )
        if done.all():
            break
    return {
        "method": name,
        "mode": mode,
        "episodes": episodes,
        "kp": kp,
        "success_rate": float(1.0 - env.failed.mean()),
        "mean_wall_impulse": float(wall_impulse.mean()),
        "mean_abs_lateral_displacement": float(np.abs(env.pos[:, 1]).mean()),
        "mean_final_forward_speed": float(env.vel[:, 0].mean()),
        "records": records,
    }
