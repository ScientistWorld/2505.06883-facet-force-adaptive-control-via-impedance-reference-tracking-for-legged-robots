"""Reduced-scale FACET training (paper Sec. 4.3 and Appendix 10.1).

A policy is trained with PPO to imitate a virtual mass-spring-damper:
rewards follow Eq. (7), averaging position/velocity exponentials over targets
integrated from the robot's states 8/16/32 control steps in the past, plus the
quadratic velocity cost and survival term of Sec. 10.1.3. Training is two-stage
(Fig. 3): a teacher acts on a privileged-feature encoder while a GRU estimator
learns to predict that feature from deployable observation history; the
student actor is initialised from the teacher and fine-tuned with PPO using
the estimator. The critic is asymmetric (sees privileged information) and all
inputs use running empirical normalisation. The deployed student is exported
as a TorchScript policy matching the benchmark interface in data/.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from data.facet_planar_env import (
    ACTION_LIMIT,
    DT,
    NOMINAL_MASS,
    PlanarForceSim,
    Scenario,
    blank_scenario,
    randomize_body,
    set_stiffness,
)
from method.facet_reference import ImpedanceCommand, ReferenceTargets

EPISODE_STEPS = 350
# Virtual mass is part of the command (paper Sec. 10.1.4 samples {1, 2, 4} kg on a
# ~15 kg robot); on this benchmark's actuation scale it is 0.5x/1x/2x the body.
VIRTUAL_MASS_CHOICES = (6.0, 12.0, 24.0)


@dataclass
class PPOConfig:
    n_envs: int = 256
    rollout_steps: int = 96
    # 600 + 150 updates of 256 envs x 96 steps = 14.7M + 3.7M env steps (paper: 200M + 50M).
    teacher_updates: int = 600
    student_updates: int = 150
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    estimator_lr: float = 1.0e-3
    estimator_epochs: int = 2
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    seed: int = 7
    estimator: str = "gru"          # "gru" (paper) or "mlp" (current observation only)
    target_mode: str = "smoothed"   # "smoothed" (paper, lags 8/16/32) or "open_loop"
    lags: tuple[int, ...] = (8, 16, 32)
    latent_dim: int = 32
    estimator_hidden: int = 128
    encoder_hidden: int = 128
    actor_hidden: tuple[int, ...] = (128, 128, 128)
    critic_hidden: tuple[int, ...] = (256, 128, 128)
    action_scale: float = 4.0
    init_log_std: float = -1.0
    threads: int = 1


# ---------------------------------------------------------------------------
# Training distribution (paper Sec. 10.1.4, on the benchmark's scale).
# ---------------------------------------------------------------------------
def sample_training_scenarios(rng: np.random.Generator, n: int) -> Scenario:
    sc = blank_scenario(n)
    randomize_body(rng, sc)
    set_stiffness(sc, np.exp(rng.uniform(math.log(0.5), math.log(32.0), size=n)))
    sc.virtual_mass[:] = rng.choice(np.asarray(VIRTUAL_MASS_CHOICES), size=n)
    sc.target_velocity[:, 0] = rng.uniform(-0.5, 2.0, size=n)
    sc.target_velocity[:, 1] = rng.uniform(-0.5, 0.5, size=n)
    sc.init_vel[:, 0] = rng.uniform(-0.5, 2.0, size=n)
    sc.init_vel[:, 1] = rng.uniform(-0.5, 0.5, size=n)
    has_const = rng.random(n) < 0.5
    sc.const_force[:] = rng.uniform(-40.0, 40.0, size=(n, 2)) * has_const[:, None]
    sc.const_start[:] = rng.uniform(0.0, 5.0, size=n)
    sc.const_duration[:] = rng.uniform(1.0, 4.0, size=n)
    has_push = rng.random(n) < 0.8
    sc.impulse_peak[:, 0] = rng.uniform(0.0, 200.0, size=n) * rng.choice([-1.0, 1.0], size=n) * has_push
    # Lateral peaks cover the benchmark's whole push sweep (log-uniform 80-1200 N).
    sc.impulse_peak[:, 1] = np.exp(rng.uniform(math.log(80.0), math.log(1200.0), size=n)) * rng.choice([-1.0, 1.0], size=n) * has_push
    sc.impulse_start[:] = rng.uniform(0.3, 6.0, size=n)
    sc.impulse_duration[:] = rng.uniform(0.4, 0.6, size=n)
    has_wall = rng.random(n) < 0.3
    sc.wall_x[:] = np.where(has_wall, sc.init_pos[:, 0] + rng.uniform(0.5, 3.0, size=n), np.inf)
    return sc


class TrainingTask:
    """Vectorised training environments with FACET rewards and privileged observations."""

    def __init__(self, config: PPOConfig):
        n = config.n_envs
        self.n = n
        self.rng = np.random.default_rng(config.seed)
        self.scenario = sample_training_scenarios(self.rng, n)
        self.sim = PlanarForceSim(self.scenario)
        self.ref = ReferenceTargets(n, DT, config.lags, config.target_mode)
        self.ref.reset(np.arange(n), self.sim.pos, self.sim.vel)
        # Stagger the first episodes so resets are not synchronised.
        self.limit = self.rng.integers(EPISODE_STEPS // 4, EPISODE_STEPS + 1, size=n)
        self.prev_action = np.zeros((n, 2))

    def command(self) -> ImpedanceCommand:
        sc = self.scenario
        return ImpedanceCommand(sc.kp, sc.kd, sc.virtual_mass, sc.target_velocity)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        sim, sc = self.sim, self.scenario
        obs = sim.observation()
        force_now, _ = sim.external_forces()
        xs, vs = self.ref.predicted_targets(force_now, self.command())
        priv = np.concatenate(
            [
                force_now / 200.0,
                ((sc.body_mass - NOMINAL_MASS) / 2.0)[:, None],
                ((sc.actuator_gain - 1.0) / 0.15)[:, None],
                ((sc.actuator_tau - 0.05) / 0.03)[:, None],
                sim.acc_eff / ACTION_LIMIT,
                (xs - sim.pos[None]).transpose(1, 0, 2).reshape(self.n, -1),
                (vs - sim.vel[None]).transpose(1, 0, 2).reshape(self.n, -1),
            ],
            axis=1,
        )
        return obs, priv.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        sim = self.sim
        prev_pos = sim.pos.copy()
        info = sim.step(action)
        command = self.command()
        self.ref.push(prev_pos, info["force"], sim.pos, sim.vel, command)
        xs, vs = self.ref.current_targets(command)
        pos_err = np.sum((sim.pos[None] - xs) ** 2, axis=-1)
        vel_err = np.sum((sim.vel[None] - vs) ** 2, axis=-1)
        # Eq. (7): mean over targets of per-target rewards (weights of Sec. 10.1.3).
        tracking = (
            0.75 * np.exp(-pos_err / 0.25) + 1.5 * (np.exp(-vel_err / 0.25) - 0.5 * np.minimum(vel_err, 16.0))
        ).mean(axis=0)
        clipped = np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)
        action_rate = np.sum((clipped - self.prev_action) ** 2, axis=-1)
        newly_failed = info["newly_failed"]
        reward = tracking + 1.0 - 0.002 * action_rate - 2.0 * newly_failed
        self.prev_action = clipped
        done = newly_failed | (sim.k >= self.limit)
        stats = {
            "tracking": float(tracking.mean()),
            "vel_err": float(np.sqrt(vel_err.mean())),
            "falls": float(newly_failed.mean()),
        }
        idx = np.flatnonzero(done)
        if idx.size:
            self.scenario.assign(idx, sample_training_scenarios(self.rng, idx.size))
            sim.reset(idx)
            self.ref.reset(idx, sim.pos, sim.vel)
            self.prev_action[idx] = 0.0
            self.limit[idx] = EPISODE_STEPS
        return reward.astype(np.float32), done, stats


# ---------------------------------------------------------------------------
# Networks.
# ---------------------------------------------------------------------------
class RunningNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / total
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -5.0, 5.0)


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers += [nn.Linear(prev, width), nn.ELU()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class FACETAgent(nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config
        self.obs_norm = RunningNorm(obs_dim)
        self.priv_norm = RunningNorm(priv_dim)
        self.priv_encoder = mlp(obs_dim + priv_dim, (config.encoder_hidden,), config.latent_dim)
        if config.estimator == "gru":
            self.gru = nn.GRUCell(obs_dim, config.estimator_hidden)
            self.est_head = nn.Linear(config.estimator_hidden, config.latent_dim)
        elif config.estimator == "mlp":
            self.est_mlp = mlp(obs_dim, (config.estimator_hidden,), config.latent_dim)
        else:
            raise ValueError(f"unknown estimator {config.estimator}")
        self.actor = mlp(obs_dim + config.latent_dim, config.actor_hidden, 2)
        self.critic = mlp(obs_dim + priv_dim, config.critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), config.init_log_std))

    def hidden_size(self) -> int:
        return self.config.estimator_hidden if self.config.estimator == "gru" else 1

    def estimator_parameters(self) -> list[nn.Parameter]:
        if self.config.estimator == "gru":
            return [*self.gru.parameters(), *self.est_head.parameters()]
        return list(self.est_mlp.parameters())

    def teacher_latent(self, on: torch.Tensor, pn: torch.Tensor) -> torch.Tensor:
        return self.priv_encoder(torch.cat([on, pn], dim=-1))

    def estimate(self, on: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.estimator == "gru":
            h = self.gru(on, h)
            return self.est_head(h), h
        return self.est_mlp(on), h

    def distribution(self, on: torch.Tensor, latent: torch.Tensor) -> Normal:
        mean = self.actor(torch.cat([on, latent], dim=-1))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, on: torch.Tensor, pn: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([on, pn], dim=-1)).squeeze(-1)


class GRUStudentPolicy(nn.Module):
    """Deployable student: GRU estimator over observation history + actor."""

    def __init__(self, agent: FACETAgent):
        super().__init__()
        self.register_buffer("obs_mean", agent.obs_norm.mean.detach().clone())
        self.register_buffer("obs_std", torch.sqrt(agent.obs_norm.var.detach() + 1e-8).clone())
        self.gru = copy.deepcopy(agent.gru)
        self.head = copy.deepcopy(agent.est_head)
        self.actor = copy.deepcopy(agent.actor)
        self.action_scale = float(agent.config.action_scale)
        self.hidden_size = int(agent.config.estimator_hidden)
        self.register_buffer("hidden", torch.zeros(1, self.hidden_size))

    @torch.jit.export
    def reset(self, n: int) -> None:
        self.hidden = torch.zeros(n, self.hidden_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        on = torch.clamp((obs - self.obs_mean) / self.obs_std, -5.0, 5.0)
        if self.hidden.shape[0] != obs.shape[0]:
            self.hidden = torch.zeros(obs.shape[0], self.hidden_size)
        self.hidden = self.gru(on, self.hidden)
        latent = self.head(self.hidden)
        action = self.action_scale * self.actor(torch.cat([on, latent], dim=-1))
        return torch.clamp(action, -8.0, 8.0)


class MLPStudentPolicy(nn.Module):
    """Deployable student whose estimator only sees the current observation."""

    def __init__(self, agent: FACETAgent):
        super().__init__()
        self.register_buffer("obs_mean", agent.obs_norm.mean.detach().clone())
        self.register_buffer("obs_std", torch.sqrt(agent.obs_norm.var.detach() + 1e-8).clone())
        self.est = copy.deepcopy(agent.est_mlp)
        self.actor = copy.deepcopy(agent.actor)
        self.action_scale = float(agent.config.action_scale)

    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        on = torch.clamp((obs - self.obs_mean) / self.obs_std, -5.0, 5.0)
        latent = self.est(on)
        action = self.action_scale * self.actor(torch.cat([on, latent], dim=-1))
        return torch.clamp(action, -8.0, 8.0)


def export_student(agent: FACETAgent, path: Path) -> None:
    agent = copy.deepcopy(agent).cpu().eval()
    module = GRUStudentPolicy(agent) if agent.config.estimator == "gru" else MLPStudentPolicy(agent)
    scripted = torch.jit.script(module.eval())
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(path))


# ---------------------------------------------------------------------------
# PPO.
# ---------------------------------------------------------------------------
def compute_gae(rewards, dones, values, next_value, gamma, lam):
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(steps)):
        nonterminal = 1.0 - dones[t]
        nxt = next_value if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        advantages[t] = last
    return advantages, advantages + values


def train(output_dir: Path, config: PPOConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, config.threads))
    output_dir.mkdir(parents=True, exist_ok=True)
    task = TrainingTask(config)
    obs_np, priv_np = task.observe()
    agent = FACETAgent(obs_np.shape[1], priv_np.shape[1], config)
    actor_params = [*agent.priv_encoder.parameters(), *agent.actor.parameters(), agent.log_std]
    critic_params = list(agent.critic.parameters())
    est_params = agent.estimator_parameters()
    actor_opt = torch.optim.Adam(actor_params, lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic_params, lr=config.critic_lr)
    est_opt = torch.optim.Adam(est_params, lr=config.estimator_lr)
    n, T = config.n_envs, config.rollout_steps
    batch = n * T
    mini = batch // config.minibatches
    h = torch.zeros(n, agent.hidden_size())
    history: list[dict] = []
    started = time.time()
    elapsed_before = 0.0
    start_stage, start_update = 0, 0
    resume_path = output_dir / "resume.pt"
    if resume_path.exists():
        # Exact resume after preemption: networks, optimisers, RNGs and the
        # full environment state are restored from the last saved update.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(state["agent"])
        actor_opt.load_state_dict(state["actor_opt"])
        critic_opt.load_state_dict(state["critic_opt"])
        est_opt.load_state_dict(state["est_opt"])
        task, obs_np, priv_np, h = state["task"], state["obs"], state["priv"], state["h"]
        history = state["history"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["np_rng"])
        start_stage, start_update = state["stage_index"], state["next_update"]
        elapsed_before = state["elapsed"]
        print(f"[facet] resumed from {resume_path} at stage {start_stage} update {start_update}", flush=True)

    def save_resume(stage_index: int, next_update: int) -> None:
        state = {
            "agent": agent.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "est_opt": est_opt.state_dict(),
            "task": task,
            "obs": obs_np,
            "priv": priv_np,
            "h": h,
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
            "stage_index": stage_index,
            "next_update": next_update,
            "elapsed": elapsed_before + time.time() - started,
        }
        tmp = resume_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(resume_path)

    print(
        f"[facet] seed={config.seed} estimator={config.estimator} targets={config.target_mode} "
        f"envs={n} steps/update={batch} teacher={config.teacher_updates} student={config.student_updates}",
        flush=True,
    )

    stages = (("teacher", config.teacher_updates), ("student", config.student_updates))
    for stage_index, (stage, updates) in enumerate(stages):
        if stage_index < start_stage:
            continue
        for update in range(start_update if stage_index == start_stage else 0, updates):
            on_buf, pn_buf, lat_buf, u_buf, logp_buf, val_buf, rew_buf, done_buf = ([] for _ in range(8))
            h0 = h.clone()
            stat_acc = {"tracking": 0.0, "vel_err": 0.0, "falls": 0.0}
            for _ in range(T):
                obs_t = torch.from_numpy(obs_np)
                priv_t = torch.from_numpy(priv_np)
                agent.obs_norm.update(obs_t)
                agent.priv_norm.update(priv_t)
                with torch.no_grad():
                    on = agent.obs_norm(obs_t)
                    pn = agent.priv_norm(priv_t)
                    est, h = agent.estimate(on, h)
                    latent = agent.teacher_latent(on, pn) if stage == "teacher" else est
                    dist = agent.distribution(on, latent)
                    u = dist.sample()
                    logp = dist.log_prob(u).sum(-1)
                    value = agent.value(on, pn)
                reward, done, stats = task.step(config.action_scale * u.numpy().astype(np.float64))
                for key in stat_acc:
                    stat_acc[key] += stats[key] / T
                done_t = torch.from_numpy(done.astype(np.float32))
                on_buf.append(on)
                pn_buf.append(pn)
                lat_buf.append(latent)
                u_buf.append(u)
                logp_buf.append(logp)
                val_buf.append(value)
                rew_buf.append(torch.from_numpy(reward))
                done_buf.append(done_t)
                h = h * (1.0 - done_t)[:, None]
                obs_np, priv_np = task.observe()
            with torch.no_grad():
                next_value = agent.value(agent.obs_norm(torch.from_numpy(obs_np)), agent.priv_norm(torch.from_numpy(priv_np)))
            on_seq = torch.stack(on_buf)
            pn_seq = torch.stack(pn_buf)
            done_seq = torch.stack(done_buf)
            rewards = torch.stack(rew_buf)
            values = torch.stack(val_buf)
            adv, ret = compute_gae(rewards, done_seq, values, next_value, config.gamma, config.gae_lambda)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            flat_on = on_seq.reshape(batch, -1)
            flat_pn = pn_seq.reshape(batch, -1)
            flat_lat = torch.stack(lat_buf).reshape(batch, -1)
            flat_u = torch.stack(u_buf).reshape(batch, -1)
            flat_logp = torch.stack(logp_buf).reshape(batch)
            flat_adv = adv.reshape(batch)
            flat_ret = ret.reshape(batch)
            for _ in range(config.epochs):
                perm = torch.randperm(batch)
                for start in range(0, batch, mini):
                    mb = perm[start : start + mini]
                    if stage == "teacher":
                        latent_mb = agent.teacher_latent(flat_on[mb], flat_pn[mb])
                    else:
                        latent_mb = flat_lat[mb]
                    dist = agent.distribution(flat_on[mb], latent_mb)
                    new_logp = dist.log_prob(flat_u[mb]).sum(-1)
                    ratio = (new_logp - flat_logp[mb]).exp()
                    surrogate = torch.min(
                        ratio * flat_adv[mb],
                        torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * flat_adv[mb],
                    )
                    actor_loss = -surrogate.mean() - config.entropy_coef * dist.entropy().sum(-1).mean()
                    value_loss = F.mse_loss(agent.value(flat_on[mb], flat_pn[mb]), flat_ret[mb])
                    actor_opt.zero_grad(set_to_none=True)
                    critic_opt.zero_grad(set_to_none=True)
                    (actor_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(actor_params, config.max_grad_norm)
                    nn.utils.clip_grad_norm_(critic_params, config.max_grad_norm)
                    actor_opt.step()
                    critic_opt.step()

            # Estimator regression onto the privileged feature (both stages).
            with torch.no_grad():
                target = agent.teacher_latent(on_seq, pn_seq)
            est_loss_value = 0.0
            for _ in range(config.estimator_epochs):
                hh = h0
                preds = []
                for t in range(T):
                    est, hh = agent.estimate(on_seq[t], hh)
                    preds.append(est)
                    hh = hh * (1.0 - done_seq[t])[:, None]
                est_loss = F.mse_loss(torch.stack(preds), target)
                est_opt.zero_grad(set_to_none=True)
                est_loss.backward()
                nn.utils.clip_grad_norm_(est_params, config.max_grad_norm)
                est_opt.step()
                est_loss_value = float(est_loss.detach())

            record = {
                "stage": stage,
                "update": update + 1,
                "reward": float(rewards.mean()),
                "done": float(done_seq.mean()),
                "estimator_loss": est_loss_value,
                "std": float(agent.log_std.detach().exp().mean()),
                "elapsed_s": round(elapsed_before + time.time() - started, 1),
                **{key: round(val, 5) for key, val in stat_acc.items()},
            }
            history.append(record)
            if (update + 1) % 10 == 0 or update + 1 == updates:
                print(f"[facet] {json.dumps(record)}", flush=True)
                save_resume(stage_index, update + 1)

    export_student(agent, output_dir / "policy.pt")
    torch.save({"agent": agent.state_dict(), "config": asdict(config)}, output_dir / "agent_state.pt")
    summary = {"config": asdict(config), "history": history, "train_seconds": round(elapsed_before + time.time() - started, 1)}
    (output_dir / "train_log.json").write_text(json.dumps(summary, indent=1))
    resume_path.unlink(missing_ok=True)
    return summary
