"""RL trainer for the WarehouseSort task — our own, faithful adaptation of the ManiSkill PPO
baseline (examples/baselines/ppo/ppo.py), the exact algorithm used to train STATE policies on
PickCube-v1 / StackCube-v1.

  python train.py difficulty=easy reward=example_dense
  python train.py difficulty=easy num_envs=1024 total_steps=10_000_000 reward=example_dense
  python train.py difficulty=hard num_envs=256 total_steps=25_000_000
  python train.py difficulty=easy evaluate=true checkpoint=outputs/.../ckpt.pt   # eval only

This is intentionally kept as close to the standalone ppo.py as possible (same loop, same
bootstrapping, same diagnostics, same in-loop eval + video + checkpointing) so it reproduces
the baseline's performance and is easy to fiddle with. The only adaptations are:

  * config comes from Hydra (conf/config.yaml ppo.*) instead of tyro CLI args — every PPO knob
    is still exposed and overridable on the CLI;
  * the env is built through warehouse_sort.utils.make_env (our difficulty/reward/obs config);
  * the actor-critic is warehouse_sort.policy.Agent, which also handles rgb dict observations
    (state mode is byte-for-byte the baseline's 3x256 Tanh MLP).

What it ports from the baseline, in full:
  * partial reset (each env resets the instant it succeeds) + value bootstrap on truncation AND
    termination via the env's `final_observation`;
  * GAE (with optional finite-horizon variant), advantage normalisation, clipped policy loss,
    optional clipped value loss, entropy bonus, gradient clipping, optional LR annealing, and
    target-KL early stopping;
  * action clipping to the action-space bounds;
  * an in-loop evaluation every `eval_freq` updates on a separate eval env (deterministic,
    full-horizon), which prints metrics, writes them to TensorBoard, records an mp4 of the
    rollout, and saves a checkpoint — plus an `evaluate=true` mode that only does this.

The competition is in replacing the reward (and policy), not this loop. An imitation-learning
trainer can mirror this script's structure.
"""

import os
import random
import time
from collections import defaultdict

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from warehouse_sort.policy import Agent
from warehouse_sort.utils import index_obs, log_run_header, make_env


def to_device(obs, device):
    if isinstance(obs, dict):
        return {k: v.to(device) for k, v in obs.items()}
    return obs.to(device)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    log_run_header(cfg, "train")
    p = cfg.ppo
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # seeding (matches the baseline)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.get("torch_deterministic", True)

    n_envs = cfg.num_envs
    n_steps = p.num_steps
    batch = n_envs * n_steps
    minibatch = batch // p.num_minibatches
    num_updates = max(1, cfg.total_steps // batch)
    target_kl = p.get("target_kl", None)
    evaluate_only = bool(cfg.get("evaluate", False))

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    eval_video_dir = os.path.join(out_dir, "eval_videos")
    writer = SummaryWriter(out_dir)
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(
        f"|{k}|{v}|" for k, v in OmegaConf.to_container(cfg, resolve=True).items()))

    # --- environments ---
    # train env: partial reset (ignore_terminations=False) so each env resets on success.
    # eval env: full-horizon episodes (ignore_terminations=True) + RecordEpisode video.
    eval_env, _ = make_env(cfg, cfg.obs_mode, cfg.randomization, num_envs=cfg.num_eval_envs,
                           ignore_terminations=True,
                           video_dir=(eval_video_dir if cfg.capture_video else None))
    num_parcels = getattr(eval_env.unwrapped, "num_parcels", 1)
    if not evaluate_only:
        env, _ = make_env(cfg, cfg.obs_mode, cfg.randomization, num_envs=n_envs,
                          ignore_terminations=False)
        action_dim = env.single_action_space.shape[0]
        a_low = torch.as_tensor(env.single_action_space.low, device=device)
        a_high = torch.as_tensor(env.single_action_space.high, device=device)
        obs, _ = env.reset(seed=cfg.seed)
        obs = to_device(obs, device)
        sample_obs = obs
    else:
        action_dim = eval_env.single_action_space.shape[0]
        sample_obs = to_device(eval_env.reset(seed=cfg.seed)[0], device)

    def clip_action(a):
        return torch.clamp(a.detach(), a_low, a_high)

    agent = Agent(sample_obs, action_dim).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=p.lr, eps=1e-5)
    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[train] loaded checkpoint {cfg.checkpoint}", flush=True)

    def save_ckpt():
        """Write/overwrite ckpt.pt in the format load_agent() (test.py/eval.py) expects."""
        torch.save({
            "model": agent.state_dict(),
            "obs_mode": cfg.obs_mode,
            "action_dim": action_dim,
            "difficulty": cfg.difficulty.name,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, ckpt_path)

    @torch.no_grad()
    def run_eval(global_step):
        """Deterministic, full-horizon eval on eval_env. Prints + logs the §9.1 metrics and
        (via RecordEpisode) saves an mp4 of the rollout. Mirrors ppo.py's eval block."""
        agent.eval()
        eo, _ = eval_env.reset()
        eo = to_device(eo, device)
        metrics = defaultdict(list)
        sorted_, mis_ = [], []
        for _ in range(cfg.num_eval_steps):
            eo, _, _, _, infos = eval_env.step(agent.act(eo, deterministic=True))
            eo = to_device(eo, device)
            if "final_info" in infos:
                mask = infos["_final_info"]
                fi = infos["final_info"]
                for k, v in fi["episode"].items():
                    metrics[k].append(v[mask].float().mean())
                if "success_count" in fi:
                    sorted_.append(fi["success_count"][mask].float().mean())
                if "mis_sort_count" in fi:
                    mis_.append(fi["mis_sort_count"][mask].float().mean())
        agent.train()
        ms = float(torch.stack(sorted_).mean()) if sorted_ else float("nan")
        misr = float(torch.stack(mis_).mean()) if mis_ else float("nan")
        for k, v in metrics.items():
            writer.add_scalar(f"eval/{k}", float(torch.stack(v).float().mean()), global_step)
        writer.add_scalar("eval/sorted_per_episode", ms, global_step)
        ae = metrics.get("success_at_end") or metrics.get("success_once")
        ae = float(torch.stack(ae).float().mean()) if ae else float("nan")
        print(f"[eval] step {global_step}  sort_accuracy {ms / max(num_parcels,1):.3f}  "
              f"sorted {ms:.2f}/{num_parcels}  all_placed {ae:.3f}  mis_sort {misr:.2f}  "
              f"video -> {eval_video_dir}", flush=True)

    if evaluate_only:
        run_eval(0)
        eval_env.close()
        writer.close()
        return ckpt_path

    # --- rollout buffers ---
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
    start_time = time.time()
    next_done = torch.zeros(n_envs, device=device)
    # train-side episode metrics, accumulated across the updates between prints
    log_succ, log_sorted = [], []

    for update in range(1, num_updates + 1):
        # in-loop eval + checkpoint (baseline does this at iteration % eval_freq == 1)
        if (update - 1) % cfg.eval_freq == 0:
            run_eval(global_step)
            save_ckpt()

        if p.get("anneal_lr", False):
            opt.param_groups[0]["lr"] = (1.0 - (update - 1.0) / num_updates) * p.lr

        final_values = torch.zeros((n_steps, n_envs), device=device)

        # ---- rollout ----
        rollout_t = time.time()
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

            obs, reward, term, trunc, info = env.step(clip_action(action))
            obs = to_device(obs, device)
            rew_buf[step] = reward.to(device).view(-1) * p.get("reward_scale", 1.0)
            next_done = torch.logical_or(term, trunc).float().to(device).view(-1)

            if "final_info" in info:
                done_mask = info["_final_info"]
                final_obs = to_device(info["final_observation"], device)
                with torch.no_grad():
                    final_values[step, done_mask] = agent.get_value(
                        index_obs(final_obs, done_mask)).view(-1)
                fi = info["final_info"]
                ep = fi.get("episode", {})
                if "success_once" in ep:
                    log_succ.append(ep["success_once"][done_mask].float().mean().item())
                if "success_count" in fi:
                    log_sorted.append(fi["success_count"][done_mask].float().mean().item())
        rollout_t = time.time() - rollout_t

        # ---- GAE (bootstrap on truncation & termination) ----
        with torch.no_grad():
            next_value = agent.get_value(obs).reshape(1, -1)
            adv = torch.zeros_like(rew_buf)
            lastgae = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                if p.get("finite_horizon_gae", False):
                    if t == n_steps - 1:
                        lam_sum = 0.0
                        rew_term = 0.0
                        val_term = 0.0
                    lam_sum = lam_sum * next_not_done
                    rew_term = rew_term * next_not_done
                    val_term = val_term * next_not_done
                    lam_sum = 1 + p.gae_lambda * lam_sum
                    rew_term = p.gae_lambda * p.gamma * rew_term + lam_sum * rew_buf[t]
                    val_term = p.gae_lambda * p.gamma * val_term + p.gamma * real_next_values
                    adv[t] = (rew_term + val_term) / lam_sum - val_buf[t]
                else:
                    delta = rew_buf[t] + p.gamma * real_next_values - val_buf[t]
                    adv[t] = lastgae = delta + p.gamma * p.gae_lambda * next_not_done * lastgae
            returns = adv + val_buf

        # ---- flatten ----
        if isinstance(obs, dict):
            b_obs = {k: v.reshape((batch, *v.shape[2:])) for k, v in obs_buf.items()}
        else:
            b_obs = obs_buf.reshape((batch, *obs_buf.shape[2:]))
        b_act = act_buf.reshape(batch, action_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        # ---- optimise ----
        idx = np.arange(batch)
        clipfracs = []
        approx_kl = old_approx_kl = None
        update_t = time.time()
        for _ in range(p.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, batch, minibatch):
                mb = idx[s:s + minibatch]
                mb_obs = {k: v[mb] for k, v in b_obs.items()} if isinstance(b_obs, dict) else b_obs[mb]
                _, newlogp, ent, newval = agent.get_action_and_value(mb_obs, b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > p.clip_coef).float().mean().item())
                if target_kl is not None and approx_kl > target_kl:
                    break

                mb_adv = b_adv[mb]
                if p.get("norm_adv", True):
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - p.clip_coef, 1 + p.clip_coef),
                ).mean()

                newval = newval.view(-1)
                if p.get("clip_vloss", False):
                    v_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(newval - b_val[mb], -p.clip_coef, p.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                entropy_loss = ent.mean()
                loss = pg - p.ent_coef * entropy_loss + p.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
                opt.step()
            if target_kl is not None and approx_kl is not None and approx_kl > target_kl:
                break
        update_t = time.time() - update_t

        # ---- diagnostics ----
        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", float(approx_kl), global_step)
        writer.add_scalar("losses/old_approx_kl", float(old_approx_kl), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout_time", rollout_t, global_step)
        writer.add_scalar("time/update_time", update_t, global_step)

        if update % max(1, num_updates // 20) == 0 or update == num_updates:
            sr = float(np.mean(log_succ)) if log_succ else float("nan")
            sc = float(np.mean(log_sorted)) if log_sorted else float("nan")
            print(f"update {update}/{num_updates}  step {global_step}  SPS {sps}  "
                  f"ret {b_ret.mean().item():.3f}  v_loss {v_loss.item():.4f}  "
                  f"approx_kl {float(approx_kl):.4f}  expl_var {explained_var:.2f}  "
                  f"train_all_placed {sr:.3f}  train_sorted {sc:.2f}/{num_parcels}", flush=True)
            log_succ.clear()
            log_sorted.clear()

    run_eval(global_step)
    save_ckpt()
    env.close()
    eval_env.close()
    writer.close()
    print(f"[train] saved checkpoint -> {ckpt_path}", flush=True)
    return ckpt_path


if __name__ == "__main__":
    main()
