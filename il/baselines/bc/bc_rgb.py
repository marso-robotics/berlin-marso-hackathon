"""RGB (wrist-cam) behaviour cloning for WarehouseSort-v1.

This is an **RGB-only** adaptation of the vendored ManiSkill ``bc_rgbd.py`` baseline: same
PlainConv encoder + MLP head and the same train/eval loop, but it consumes only the wrist
camera's 3-channel RGB image (no depth) plus proprioceptive state. Depth adds little on this
task and keeps the obs identical to what a real wrist camera streams.

Differences from ``bc_rgbd.py`` (all mechanical):
  * env ``obs_mode="rgb"`` and ``FlattenRGBDObservationWrapper(rgb=True, depth=False)``  ->
    obs dict is ``{"rgb": (N,H,W,3) uint8, "state": (N,S) float32}``.
  * the dataset loads only ``obs/sensor_data/<cam>/rgb`` (normalised /255).
  * ``PlainConv`` input channels = ``3 * camera_count``.

Run from this directory (so ``behavior_cloning`` resolves):
  pixi run python bc_rgb.py --env-id WarehouseSort-v1 \
      --demo-path ../../demos/easy/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5 \
      --control-mode pd_ee_delta_pos --sim-backend gpu \
      --max-episode-steps 150 --total-iters 30000 --batch-size 256 \
      --num-eval-envs 8 --num-eval-episodes 32 --exp-name warehouse_rgb_bc
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from behavior_cloning.evaluate import evaluate
from behavior_cloning.make_env import make_eval_envs


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    env_id: str = "WarehouseSort-v1"
    demo_path: str = "../../demos/easy/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5"
    num_demos: Optional[int] = None
    total_iters: int = 30_000
    batch_size: int = 256

    lr: float = 3e-4

    max_episode_steps: Optional[int] = None
    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 32
    num_eval_envs: int = 8
    sim_backend: str = "gpu"
    num_dataload_workers: int = 0
    control_mode: str = "pd_ee_delta_pos"
    demo_type: Optional[str] = None


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


def flatten_state_dict_with_space(state_dict: dict):
    states = []
    for key in state_dict.keys():
        value = state_dict[key]
        if isinstance(value, dict):
            state = flatten_state_dict_with_space(value)
        elif isinstance(value, np.ndarray):
            state = value if value.ndim > 1 else value.reshape(-1, 1)
        else:
            state = np.array(value).reshape(-1, 1)
        states.append(state)
    return np.hstack(states)


class ManiSkillRGBDataset(Dataset):
    """Loads (wrist rgb, proprio state, action) frames from a replayed rgb trajectory .h5."""

    def __init__(self, dataset_file: str, device, load_count=None):
        self.data = h5py.File(dataset_file, "r")
        self.json_data = load_json(dataset_file.replace(".h5", ".json"))
        self.episodes = self.json_data["episodes"]
        self.device = device

        self.camera_data = defaultdict(list)
        self.actions, self.states = [], []
        if load_count is None:
            load_count = len(self.episodes)
        print(f"Loading {load_count} episodes (rgb)")
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            traj = load_h5_data(self.data[f"traj_{eps['episode_id']}"])
            agent = traj["obs"]["agent"]
            extra = traj["obs"]["extra"]
            state = np.hstack([
                flatten_state_dict_with_space(agent),
                flatten_state_dict_with_space(extra),
            ])
            self.states.append(state[:-1])  # drop terminal obs (no action)
            for cam, cam_data in traj["obs"]["sensor_data"].items():
                self.camera_data[cam + "_rgb"].append(cam_data["rgb"][:-1])
            self.actions.append(traj["actions"])
        for k in self.camera_data:
            self.camera_data[k] = np.vstack(self.camera_data[k]) / 255.0
        self.states = np.vstack(self.states)
        self.actions = np.vstack(self.actions)
        for k in self.camera_data:
            assert self.camera_data[k].shape[0] == self.actions.shape[0]

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        out = {
            "action": torch.from_numpy(self.actions[idx]).float().to(self.device),
            "state": torch.from_numpy(self.states[idx]).float().to(self.device),
        }
        rgb = [torch.from_numpy(self.camera_data[k][idx]).float().to(self.device)
               for k in sorted(self.camera_data.keys())]
        out["rgb"] = torch.cat(rgb, dim=-1)  # (H, W, 3*camera_count)
        return out


class IterationBasedBatchSampler(BatchSampler):
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class PlainConv(nn.Module):
    """Same small conv stack as bc_rgbd.PlainConv (no torchvision dependency)."""

    def __init__(self, in_channels=3, out_dim=256, max_pooling=False, inactivated_output=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )
        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)
        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, camera_count=1):
        super().__init__()
        self.encoder = PlainConv(in_channels=3 * camera_count, out_dim=256,
                                 max_pooling=False, inactivated_output=False)
        self.final_mlp = make_mlp(256 + state_dim, [512, 256, action_dim], last_act=False)
        self.get_eval_action = self.get_action = self.forward

    def forward(self, rgb, state):
        img = rgb.permute(0, 3, 1, 2)  # (B, C, H, W)
        feature = self.encoder(img)
        return self.final_mlp(torch.cat([feature, state], dim=1))


def save_ckpt(run_name, tag, actor):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save({"actor": actor.state_dict()}, f"runs/{run_name}/checkpoints/{tag}.pt")


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = args.exp_name or f"{args.env_id}__bc_rgb__{args.seed}__{int(time.time())}"

    # control-mode sanity check against the demo json
    import json
    with open(args.demo_path[:-2] + "json") as f:
        demo_info = json.load(f)
    cm = demo_info["env_info"]["env_kwargs"].get("control_mode")
    assert cm is None or cm == args.control_mode, \
        f"control mode mismatch: dataset={cm}, args={args.control_mode}"

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse",
                      obs_mode="rgb", render_mode="all")
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    # rgb-only flatten: obs dict -> {"rgb": (N,H,W,3) uint8, "state": (N,S)}
    rgb_wrapper = lambda e: FlattenRGBDObservationWrapper(e, rgb=True, depth=False, state=True)
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs,
                          video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
                          wrappers=[rgb_wrapper])

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" %
                    ("\n".join(f"|{k}|{v}|" for k, v in vars(args).items())))

    ds = ManiSkillRGBDataset(args.demo_path, device=device, load_count=args.num_demos)
    _vobs, _ = envs.reset(seed=args.seed)
    # --- VERIFY (every run): the RGB policy sees ONLY images + robot proprioception, never the
    # privileged state (parcel poses / tags / bin positions). The easy privileged state vector is
    # 54-d; this proprio-only 'state' is 26-d (qpos9+qvel9+tcp_pose7+is_grasped1). ---
    print("[bc_rgb] observation passed to policy:",
          {k: tuple(v.shape) for k, v in _vobs.items()}, flush=True)
    assert set(_vobs.keys()) == {"rgb", "state"}, f"unexpected obs keys: {list(_vobs.keys())}"
    assert _vobs["state"].shape[1] == ds.states.shape[1], "eval/demo state dim mismatch"
    assert _vobs["state"].shape[1] < 54, \
        f"state dim {_vobs['state'].shape[1]} looks privileged (>=54); RGB must be proprio-only"
    print(f"[bc_rgb] OK: rgb {tuple(_vobs['rgb'].shape[1:])} + proprio state "
          f"({_vobs['state'].shape[1]}-d) only; no privileged parcel/bin/tag info.", flush=True)
    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    data_loader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=args.num_dataload_workers)

    camera_count = len(ds.camera_data.keys())  # rgb-only -> one key per camera
    actor = Actor(ds.states.shape[1], envs.single_action_space.shape[0], camera_count).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    best_eval_metrics = defaultdict(float)

    for iteration, batch in enumerate(data_loader):
        optimizer.zero_grad()
        preds = actor(batch["rgb"], batch["state"])
        loss = F.mse_loss(preds, batch["action"])
        loss.backward()
        optimizer.step()

        if iteration % args.log_freq == 0:
            print(f"Iteration {iteration}, loss: {loss.item():.5f}", flush=True)
            writer.add_scalar("losses/total_loss", loss.item(), iteration)

        if iteration % args.eval_freq == 0:
            actor.eval()

            def sample_fn(obs):
                if isinstance(obs["rgb"], np.ndarray):
                    obs = {k: torch.from_numpy(v).to(device) for k, v in obs.items()}
                rgb = obs["rgb"].float() / 255.0
                action = actor(rgb, obs["state"].float())
                if args.sim_backend == "cpu":
                    action = action.cpu().numpy()
                return action

            with torch.no_grad():
                eval_metrics = evaluate(args.num_eval_episodes, sample_fn, envs)
            actor.train()
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes", flush=True)
            for k in eval_metrics:
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"  {k}: {eval_metrics[k]:.4f}", flush=True)
            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}", actor)
                    print(f"  New best {k}: {eval_metrics[k]:.4f} -> saved ckpt", flush=True)

        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration), actor)
    envs.close()
