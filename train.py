"""Train a policy on the WarehouseSort task at a chosen difficulty, on the TRAIN
distribution (narrow randomisation ranges, visible seeds). Saves a checkpoint to the Hydra
run output dir.

  python train.py difficulty=easy total_steps=200_000
  python train.py difficulty=hard num_envs=256 total_steps=5_000_000

This is a plain PPO baseline so the loop closes end-to-end; the competition is in replacing
the reward (and policy). See README.
"""

import os

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from warehouse_sort.policy import Agent
from warehouse_sort.utils import log_run_header, make_env


def to_device(obs, device):
    if isinstance(obs, dict):
        return {k: v.to(device) for k, v in obs.items()}
    return obs.to(device)


def clone_obs(obs):
    if isinstance(obs, dict):
        return {k: v.clone() for k, v in obs.items()}
    return obs.clone()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    log_run_header(cfg, "train")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    p = cfg.ppo
    n_envs = cfg.num_envs
    n_steps = p.num_steps
    batch = n_envs * n_steps
    minibatch = batch // p.num_minibatches
    num_updates = max(1, cfg.total_steps // batch)

    # train distribution = the difficulty's own (narrow) randomisation ranges.
    env, is_rgb = make_env(cfg, cfg.obs_mode, cfg.randomization, num_envs=n_envs)
    action_dim = env.single_action_space.shape[0]

    obs, _ = env.reset(seed=cfg.seed)
    obs = to_device(obs, device)
    agent = Agent(obs, action_dim).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=p.lr, eps=1e-5)

    # rollout buffers
    def alloc(o):
        if isinstance(o, dict):
            return {k: torch.zeros((n_steps, *v.shape), dtype=v.dtype, device=device) for k, v in o.items()}
        return torch.zeros((n_steps, *o.shape), dtype=o.dtype, device=device)

    obs_buf = alloc(obs)
    act_buf = torch.zeros((n_steps, n_envs, action_dim), device=device)
    logp_buf = torch.zeros((n_steps, n_envs), device=device)
    rew_buf = torch.zeros((n_steps, n_envs), device=device)
    done_buf = torch.zeros((n_steps, n_envs), device=device)
    val_buf = torch.zeros((n_steps, n_envs), device=device)

    global_step = 0
    next_done = torch.zeros(n_envs, device=device)
    for update in range(num_updates):
        for step in range(n_steps):
            global_step += n_envs
            if isinstance(obs, dict):
                for k in obs:
                    obs_buf[k][step] = obs[k]
            else:
                obs_buf[step] = obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value.flatten()

            obs, reward, term, trunc, info = env.step(action)
            obs = to_device(obs, device)
            rew_buf[step] = reward.to(device).view(-1)
            next_done = torch.logical_or(term, trunc).float().to(device).view(-1)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(obs).flatten()
            adv = torch.zeros_like(rew_buf)
            lastgae = 0
            for t in reversed(range(n_steps)):
                nonterminal = 1.0 - (next_done if t == n_steps - 1 else done_buf[t + 1])
                nextval = next_value if t == n_steps - 1 else val_buf[t + 1]
                delta = rew_buf[t] + p.gamma * nextval * nonterminal - val_buf[t]
                adv[t] = lastgae = delta + p.gamma * p.gae_lambda * nonterminal * lastgae
            returns = adv + val_buf

        # flatten
        if isinstance(obs, dict):
            b_obs = {k: v.reshape((batch, *v.shape[2:])) for k, v in obs_buf.items()}
        else:
            b_obs = obs_buf.reshape((batch, *obs_buf.shape[2:]))
        b_act = act_buf.reshape(batch, action_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)

        idx = np.arange(batch)
        for _ in range(p.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, batch, minibatch):
                mb = idx[s:s + minibatch]
                mb_obs = {k: v[mb] for k, v in b_obs.items()} if isinstance(b_obs, dict) else b_obs[mb]
                _, newlogp, ent, newval = agent.get_action_and_value(mb_obs, b_act[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - p.clip_coef, 1 + p.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newval.flatten() - b_ret[mb]) ** 2).mean()
                loss = pg - p.ent_coef * ent.mean() + p.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
                opt.step()

        if update % max(1, num_updates // 10) == 0 or update == num_updates - 1:
            sps = int(global_step / (update + 1) / max(1e-9, 1))
            print(f"update {update+1}/{num_updates}  step {global_step}  "
                  f"ret {b_ret.mean().item():.3f}  v_loss {v_loss.item():.4f}", flush=True)

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt = os.path.join(out_dir, "ckpt.pt")
    torch.save({
        "model": agent.state_dict(),
        "obs_mode": cfg.obs_mode,
        "action_dim": action_dim,
        "difficulty": cfg.difficulty.name,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, ckpt)
    env.close()
    print(f"[train] saved checkpoint -> {ckpt}", flush=True)
    return ckpt


if __name__ == "__main__":
    main()
