"""Record ONE clean DP-state rollout video (render + scene cam) from a trained checkpoint.

Builds the eval env directly (rather than via make_eval_envs) so we can set
``max_steps_per_video`` to the FULL horizon — make_eval_envs derives it from the *registered*
max_episode_steps (100), which would clip the video before the return-home and split it in two.
"""
import sys, glob, os
from types import SimpleNamespace
import torch
import gymnasium as gym
import warehouse_sort  # noqa
from train import Agent  # vendored DP Agent (diffusion U-Net)
from mani_skill.utils import common
from mani_skill.utils.wrappers import FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/warehouse_state_dp_v2/checkpoints/best_eval_success_at_end.pt"
VIDDIR = sys.argv[2] if len(sys.argv) > 2 else "/home/david/code/marso_hackathon/il/videos/state_dp"
HORIZON = 200   # full two-parcel sort + clean return-home in ONE clip
args = SimpleNamespace(obs_horizon=2, act_horizon=8, pred_horizon=16,
                       diffusion_step_embed_dim=64, unet_dims=[64, 128, 256], n_groups=8)
device = "cuda"

base = gym.make("WarehouseSort-v1", num_envs=1, obs_mode="state",
                control_mode="pd_ee_delta_pos", sim_backend="gpu", render_mode="all",
                max_episode_steps=HORIZON, difficulty="easy", num_parcels=2, fixed_poses=True,
                human_render_camera_configs=dict(shader_pack="default"))
base = FrameStack(base, num_stack=args.obs_horizon)
base = RecordEpisode(base, output_dir=VIDDIR, save_trajectory=False, save_video=True,
                     video_fps=20, max_steps_per_video=HORIZON)
envs = ManiSkillVectorEnv(base, ignore_terminations=True, record_metrics=True)

agent = Agent(envs, args).to(device)
ck = torch.load(CKPT, map_location=device, weights_only=False)
agent.load_state_dict(ck["ema_agent"]); agent.eval()

obs, _ = envs.reset(seed=5000)
steps = 0
while steps < HORIZON:
    obs = common.to_tensor(obs, device)
    aseq = agent.get_action(obs)
    stop = False
    for i in range(aseq.shape[1]):
        obs, r, te, tr, info = envs.step(aseq[:, i]); steps += 1
        if tr.any() or steps >= HORIZON:
            stop = True; break
    if stop:
        break
sc = info["final_info"]["episode"]["success_at_end"].float().mean().item() if "final_info" in info else "n/a"
print("steps:", steps, "final success_at_end:", sc)
envs.close()
for extra in sorted(glob.glob(os.path.join(VIDDIR, "*.mp4")))[1:]:
    os.remove(extra)
print("video ->", sorted(glob.glob(os.path.join(VIDDIR, "*.mp4"))))
