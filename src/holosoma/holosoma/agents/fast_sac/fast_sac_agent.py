from __future__ import annotations

import copy
import itertools
import math
import os
import statistics
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Sequence

import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.fast_sac.fast_sac import Actor, CNNActor, CNNCritic, Critic, PCActor, PCCritic
from holosoma.agents.fast_sac.fast_sac_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    save_params,
)
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import FastSACConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.safe_torch_import import (
    F,
    GradScaler,
    TensorDict,
    autocast,
    nn,
    optim,
    torch,
)

from torch.utils.tensorboard import SummaryWriter
from collections import deque

torch.set_float32_matmul_precision("high")


class FastSACEnv:
    def __init__(
        self,
        env: BaseTask,
        actor_obs_keys: Sequence[str],
        critic_obs_keys: Sequence[str],
    ):
        self._env = env
        self._actor_obs_keys = actor_obs_keys
        self._critic_obs_keys = critic_obs_keys

        # Initialize per-joint action boundaries for proper tanh scaling
        self._action_boundaries = self._compute_action_boundaries()

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self._env, name)

    def reset(self) -> torch.Tensor:
        obs_dict = self._env.reset_all()
        return torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict = self._env.reset_all()
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        return actor_obs, critic_obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Actions are now already scaled by the actor, so pass them directly to the environment
        obs_dict, rew_buf, reset_buf, info_dict = self._env.step({"actions": actions})  # type: ignore[attr-defined]
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        if "final_observations" in info_dict:
            # Use true final observations when available
            final_actor_obs = torch.cat([info_dict["final_observations"][k] for k in self._actor_obs_keys], dim=1)
            final_critic_obs = torch.cat([info_dict["final_observations"][k] for k in self._critic_obs_keys], dim=1)
        else:
            final_actor_obs = actor_obs
            final_critic_obs = critic_obs
        extras = {
            "time_outs": info_dict["time_outs"],
            "observations": {
                "critic": critic_obs,
                "final": {
                    "actor_obs": final_actor_obs,
                    "critic_obs": final_critic_obs,
                },
            },
            "episode": info_dict["episode"],
            "episode_all": info_dict["episode_all"],
            "raw_episode": info_dict.get("raw_episode", {}),
            "raw_episode_all": info_dict.get("raw_episode_all", {}),
            "to_log": info_dict["to_log"],
            "ep_success": info_dict["ep_success"],
            "ep_counted": info_dict.get("ep_counted", 0),
        }
        return actor_obs, rew_buf, reset_buf, extras

    def _compute_action_boundaries(self) -> torch.Tensor:
        """
        Compute per-joint action scaling factors based on robot configuration.
        Returns tensor of shape (num_dof,) containing the scaling factor for each joint.

        The scaling factor is the maximum difference between default and joint limits,
        ensuring that action=0 corresponds to default position and action=±1 reaches
        the furthest limit from default.
        """
        robot_config = self._env.robot_config

        # Get joint limits and default positions
        dof_pos_lower_limits = torch.tensor(robot_config.dof_pos_lower_limit_list, device=self._env.device)
        dof_pos_upper_limits = torch.tensor(robot_config.dof_pos_upper_limit_list, device=self._env.device)

        # Get default joint angles
        default_joint_angles = torch.zeros(len(robot_config.dof_names), device=self._env.device)
        for i, joint_name in enumerate(robot_config.dof_names):
            if joint_name in robot_config.init_state.default_joint_angles:
                default_joint_angles[i] = robot_config.init_state.default_joint_angles[joint_name]

        # Get action scale from robot config
        action_scale = robot_config.control.action_scale

        # Compute maximum range from default to either limit for each joint
        # This ensures symmetric scaling where action=0 -> default position
        range_to_lower = torch.abs(dof_pos_lower_limits - default_joint_angles)
        range_to_upper = torch.abs(dof_pos_upper_limits - default_joint_angles)
        max_range = torch.maximum(range_to_lower, range_to_upper)

        # Account for action_scale: the environment applies actions_scaled = actions * action_scale
        # So our scaling factor should be: max_range / action_scale
        action_scaling_factors = max_range / action_scale

        logger.info(f"Computed action scaling factors for {len(robot_config.dof_names)} DOFs")
        logger.info(f"Action scale: {action_scale}")
        logger.info(f"Scaling: {action_scaling_factors}")

        return action_scaling_factors


# Minimum completed episodes before an episode-derived metric is logged. Below this the values
# quantise hard: at low env counts a logging window can hold a single episode, so success_rate
# reads 0.00 or 1.00 and the episodic means swing with one sample.
MIN_EPISODES_TO_LOG = 100


class FastSACAgent(BaseAlgo):
    """
    FastSAC is an efficient variant of Soft Actor-Critic (SAC) tuned for
    large-scale training with massively parallel simulation.
    See https://arxiv.org/abs/2505.22642 for more details about FastTD3.
    Detailed technical report for FastSAC will be available soon.
    """

    config: FastSACConfig
    env: FastSACEnv  # type: ignore[assignment]
    actor: Actor
    qnet: Critic

    def __init__(
        self, env: BaseTask, config: FastSACConfig, device: str, log_dir: str, multi_gpu_cfg: dict | None = None, expert_policy=None, expert_critic=None, lambda_bc_policy=0.0, lambda_bc_critic=0.0, use_cpu_rb=False
    ):
        wrapped_env = FastSACEnv(env, config.actor_obs_keys, config.critic_obs_keys)

        super().__init__(wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.log_dir = log_dir
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=log_dir)

        self.training_metrics = TensorAverageMeterDict()
        self.eval_callbacks: list[RLEvalCallback] = []
        self.expert_policy = expert_policy
        self.expert_critic = expert_critic
        self.lambda_bc_policy = lambda_bc_policy
        self.lambda_bc_critic = lambda_bc_critic
        self.expert_ratio: float = 0.5
        self.expert_ratio_anneal_steps: int = 0  # 0 = no annealing
        self.rb_device = "cpu" if use_cpu_rb else self.device

    def enable_sgft(
        self, shape_rewards: bool = True, h_step_backup: bool = False, ckpt_path: str | None = None
    ) -> None:
        """Freeze a source value function for reward shaping.

        SGFT replaces each stored reward with

            r_hat = r + gamma * Phi(s') - Phi(s),   Phi(s) = mean_i Q_i(s, mu(s))

        where mu is the frozen source actor's deterministic action. This is potential-based
        shaping, so the optimal policy is unchanged -- it only redistributes credit toward
        states the source policy already valued.

        If `ckpt_path` is given, the source actor/critic/normalizers are loaded from that
        checkpoint (same format as save()/load()), so the shaping source can differ from the
        weights being trained. Otherwise it snapshots whatever weights are live at the time,
        so it MUST be called after load(). The observation normalizers are frozen too, because
        the live ones keep updating during training and a frozen network behind a live
        normalizer would not be a fixed function of s.
        """
        self._sgft_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self._sgft_qnet = copy.deepcopy(self.qnet).eval().requires_grad_(False)
        self._sgft_obs_norm = copy.deepcopy(self.obs_normalizer).eval().requires_grad_(False)
        self._sgft_critic_obs_norm = copy.deepcopy(self.critic_obs_normalizer).eval().requires_grad_(False)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self._sgft_actor.load_state_dict(ckpt["actor_state_dict"])
            self._sgft_qnet.load_state_dict(ckpt["qnet_state_dict"])
            self._sgft_obs_norm.load_state_dict(ckpt["obs_normalizer_state"])
            self._sgft_critic_obs_norm.load_state_dict(ckpt["critic_obs_normalizer_state"])
        self.sgft_enabled = True
        self.sgft_shaping = shape_rewards
        self.h_step_backup = h_step_backup
        source = f"checkpoint {ckpt_path}" if ckpt_path is not None else "the live (loaded) weights"
        logger.info(
            f"SGFT source frozen from {source} | reward shaping={shape_rewards} | "
            f"h-step backup={h_step_backup}"
        )
        if shape_rewards and h_step_backup:
            logger.warning(
                "SGFT: reward shaping AND h-step backup are both on. The shaped n-step return "
                "already telescopes to (raw + gamma^n * V(s_n) - V(s_0)), and the h-step target "
                "adds gamma^n * V(s_n) again -- V_source is counted twice. They are normally "
                "alternatives; enable only one unless you specifically want the doubled term."
            )
        if shape_rewards and self.expert_rb is not None:
            self._sgft_shape_expert_buffer()

    def _renorm_to_source(self, x_norm: torch.Tensor, live: nn.Module, src: nn.Module) -> torch.Tensor:
        """Re-express an obs normalized by the LIVE normalizer in the FROZEN source's frame.

        Batches reaching the update are already normalized with the live running statistics, which
        keep moving. Feeding them straight to the frozen source would silently drift V_source as
        training progresses -- exactly what freezing was meant to prevent. Undo the live transform
        and reapply the snapshotted one.
        """
        if not self.obs_normalization:
            return x_norm
        raw = x_norm * (live._std + live.eps) + live._mean
        return (raw - src._mean) / (src._std + src.eps)

    @torch.no_grad()
    def sgft_value(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Phi(s) = mean_i Q_i(s, mu(s)) under the frozen source networks. Shape [batch]."""
        a_in = self._sgft_obs_norm(actor_obs, update=False) if self.obs_normalization else actor_obs
        c_in = self._sgft_critic_obs_norm(critic_obs, update=False) if self.obs_normalization else critic_obs
        action = self._sgft_actor(a_in)[0]  # forward returns (action, mean, log_std); [0] is deterministic
        q_logits = self._sgft_qnet(c_in, action)  # [num_critics, batch, num_atoms]
        q = self._sgft_qnet.get_value(F.softmax(q_logits, dim=-1))  # [num_critics, batch]
        return q.mean(dim=0)

    @torch.no_grad()
    def sgft_shaped_rewards(
        self, obs, critic_obs, next_obs, next_critic_obs, rewards, dones, truncations, gamma
    ) -> torch.Tensor:
        """r_hat = r + gamma * Phi(s') - Phi(s), with Phi(s') forced to 0 on true terminations.

        Terminations zero the potential for two independent reasons: it is the standard
        potential-based convention (policy invariance only holds when an absorbing state has
        potential 0), and `next_obs` is the POST-RESET observation on terminations -- only
        truncations carry the true final obs -- so Phi(next_obs) would be read off an unrelated
        state and inject a spurious bonus at every episode boundary.
        """
        phi_s = self.sgft_value(obs, critic_obs)
        phi_next = self.sgft_value(next_obs, next_critic_obs)
        terminated = dones.bool() & ~truncations.bool()
        phi_next = torch.where(terminated, torch.zeros_like(phi_next), phi_next)
        return rewards + gamma * phi_next - phi_s

    @torch.no_grad()
    def _sgft_shape_expert_buffer(self) -> None:
        """Rewrite the expert buffer's rewards with the same shaping, once, in place.

        Without this a mixed batch would carry two different reward definitions and the critic
        would regress to an inconsistent target. Chunked over the capacity axis so a multi-GB
        buffer never has to be resident on the compute device all at once.
        """
        rb = self.expert_rb
        n_env, cap = rb.observations.shape[0], rb.observations.shape[1]
        n_obs, n_cobs = rb.observations.shape[-1], rb.critic_observations.shape[-1]
        gamma = self.config.gamma
        chunk = max(1, 2 ** 20 // max(n_env * max(n_obs, 1), 1))
        for s in range(0, cap, chunk):
            e = min(s + chunk, cap)
            dev = self.device
            r_hat = self.sgft_shaped_rewards(
                rb.observations[:, s:e].reshape(-1, n_obs).to(dev),
                rb.critic_observations[:, s:e].reshape(-1, n_cobs).to(dev),
                rb.next_observations[:, s:e].reshape(-1, n_obs).to(dev),
                rb.next_critic_observations[:, s:e].reshape(-1, n_cobs).to(dev),
                rb.rewards[:, s:e].reshape(-1).to(dev),
                rb.dones[:, s:e].reshape(-1).to(dev),
                rb.truncations[:, s:e].reshape(-1).to(dev),
                gamma,
            )
            rb.rewards[:, s:e] = r_hat.view(n_env, e - s).to(rb.rewards.device)
        logger.info(f"SGFT: shaped {n_env * cap} expert transitions in place (chunk={chunk})")

    @property
    def _current_expert_ratio(self) -> float:
        if self.expert_ratio_anneal_steps <= 0:
            return self.expert_ratio
        return self.expert_ratio * max(0.0, 1.0 - self.global_step / self.expert_ratio_anneal_steps)

    def setup(self) -> None:
        logger.info("Setting up FastSAC")

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

        args = self.config
        device = self.device
        env = self.env

        algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        algo_history_length_dict: Dict[str, int] = {}

        for group_cfg in self.env.observation_manager.cfg.groups.values():
            history_len = getattr(group_cfg, "history_length", 1)
            for term_name in group_cfg.terms:
                algo_history_length_dict[term_name] = history_len

        actor_obs_keys = self.config.actor_obs_keys
        critic_obs_keys = self.config.critic_obs_keys

        n_act = self.env.robot_config.actions_dim

        # Compute actor observation dimensions and store indices
        actor_obs_dim = 0
        self.actor_obs_indices = {}
        for obs_key in actor_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len

            # Store start and end indices for this observation key
            self.actor_obs_indices[obs_key] = {
                "start": actor_obs_dim,
                "end": actor_obs_dim + obs_size,
                "size": obs_size,
            }
            actor_obs_dim += obs_size

        self.actor_obs_dim = actor_obs_dim

        # Compute critic observation dimensions and store indices
        critic_obs_dim = 0
        self.critic_obs_indices = {}
        for obs_key in critic_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len

            # Store start and end indices for this observation key
            self.critic_obs_indices[obs_key] = {
                "start": critic_obs_dim,
                "end": critic_obs_dim + obs_size,
                "size": obs_size,
            }
            critic_obs_dim += obs_size

        self.scaler = GradScaler(enabled=args.amp)

        self.obs_normalization = args.obs_normalization
        if args.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Get action scaling parameters from the environment
        action_scale = env._action_boundaries if args.use_tanh else torch.ones(n_act, device=device)
        action_bias = torch.zeros(n_act, device=device)  # Assuming zero bias for now

        # Handle CNN actor/critic
        if args.use_cnn_encoder or args.use_pc_encoder:
            # We assume that MLP doesn't take raw encoder observations
            actor_mlp_obs_keys = [k for k in actor_obs_keys if k != args.encoder_obs_key]
            critic_mlp_obs_keys = [k for k in critic_obs_keys if k != args.encoder_obs_key]

            if args.use_cnn_encoder:
                actor_cls, critic_cls = (CNNActor, CNNCritic)
            else:
                actor_cls, critic_cls = (PCActor, PCCritic)
        else:
            actor_mlp_obs_keys = list(actor_obs_keys)
            critic_mlp_obs_keys = list(critic_obs_keys)
            actor_cls, critic_cls = (Actor, Critic)

        self.actor = actor_cls(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_mlp_obs_keys,
            n_act=n_act,
            num_envs=env.num_envs,
            device=device,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )

        print(self.actor)
        print(self.qnet)

        self.log_alpha = torch.tensor([math.log(args.alpha_init)], requires_grad=True, device=device)
        self.policy = self.actor.explore

        self.qnet_target = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        self.target_entropy = -n_act * args.target_entropy_ratio
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=args.alpha_learning_rate, fused=True, betas=(0.9, 0.95))

        logger.info(f"actor_obs_dim: {actor_obs_dim}, critic_obs_dim: {critic_obs_dim}")

        # Only the first num_collect_envs envs feed the buffer. The rest still step and still count
        # toward the logged episode statistics -- that is the whole point: at 1 collecting env a
        # logging window holds ~0.6 episodes, so success_rate is a coin flip, while 128 stepping envs
        # give ~80 episodes per window for ~17% throughput. Sizing the buffer by env.num_envs instead
        # would also multiply its memory by the eval envs (128 x 1e6 transitions is ~448 GB).
        self.num_collect_envs = args.num_collect_envs or env.num_envs
        if not 1 <= self.num_collect_envs <= env.num_envs:
            raise ValueError(
                f"num_collect_envs={self.num_collect_envs} must be in [1, num_envs={env.num_envs}]"
            )
        if self.num_collect_envs != env.num_envs:
            logger.info(
                f"Collecting from {self.num_collect_envs}/{env.num_envs} envs; the other "
                f"{env.num_envs - self.num_collect_envs} step for episode statistics only."
            )
        self.rb = SimpleReplayBuffer(
            n_env=self.num_collect_envs,
            buffer_size=args.buffer_size,
            n_obs=actor_obs_dim,
            n_act=n_act,
            n_critic_obs=critic_obs_dim,
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=self.rb_device,
        )
        self.expert_rb: SimpleReplayBuffer | None = None

        # getattr: this agent is fed by two config lineages (holosoma FastSACConfig and
        # isaaclab_rl RslRlOffPolicyRunnerCfg); tolerate one that predates this field.
        if getattr(args, "load_replay_buffer_path", ""):
            self.load_replay_buffer(args.load_replay_buffer_path)

        # SGFT: frozen source value function used for potential-based reward shaping.
        # Populated by enable_sgft(); never trained, never updated.
        self.sgft_enabled: bool = False      # source frozen and available
        self.sgft_shaping: bool = False      # rewrite stored rewards as r + gamma*V(s') - V(s)
        self.h_step_backup: bool = False     # bootstrap the n-step target off V_source, not Q_target
        self._sgft_actor: nn.Module | None = None
        self._sgft_qnet: nn.Module | None = None
        self._sgft_obs_norm: nn.Module | None = None
        self._sgft_critic_obs_norm: nn.Module | None = None

        if args.use_symmetry:
            # using env._env is not really ideal..
            self.symmetry_utils = SymmetryUtils(env._env)

        # Synchronize model parameters across GPUs for consistent initialization
        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self):
        """Synchronize actor, qnet, and log_alpha parameters across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast qnet weights from rank 0 to all other ranks
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast log_alpha parameter from rank 0 to all other ranks
        torch.distributed.broadcast(self.log_alpha.data, src=0)

        # Load qnet_target weights from synced qnet
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Batches and all-reduces gradients across GPUs to reduce NCCL call count.

        This flattens all existing parameter gradients into a single contiguous
        tensor, performs one all_reduce, averages by world size, and then
        scatters the reduced values back into the original gradient tensors.
        """
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    def _update_main(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()

            # `dones` is terminated | truncated (see the vec-env wrapper), so the default masks the
            # bootstrap on timeouts too, treating them as true terminals. With handle_truncations
            # the target bootstraps through a truncation and only real terminations zero it; the
            # rollout already stores the pre-reset final observation for those steps, so the
            # bootstrapped value is taken at the correct state.
            if args.handle_truncations:
                bootstrap = (truncations | ~dones).float()
            else:
                bootstrap = (~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                discount = args.gamma ** data["next"]["effective_n_steps"]

                if self.h_step_backup:
                    # h-step backup: the n-step return bootstraps off the FROZEN source critic
                    # instead of the learned target critic. The source is distributional, so its
                    # atom distribution goes through the same categorical Bellman projection --
                    # collapsing it to a scalar point mass would discard the return distribution
                    # the C51 cross-entropy is regressing against.
                    #
                    # The entropy bonus is deliberately dropped here: V_source is a hard value
                    # function carrying no entropy term, so subtracting alpha*log_pi from a target
                    # built on it would mix two different value definitions.
                    src_next_obs = self._renorm_to_source(
                        next_observations, self.obs_normalizer, self._sgft_obs_norm
                    )
                    src_next_critic_obs = self._renorm_to_source(
                        next_critic_observations, self.critic_obs_normalizer, self._sgft_critic_obs_norm
                    )
                    src_actions = self._sgft_actor(src_next_obs)[0]  # deterministic source action
                    target_net = self._sgft_qnet
                    target_distributions = target_net.projection(
                        src_next_critic_obs, src_actions, rewards, bootstrap, discount,
                    )
                else:
                    target_net = qnet_target
                    target_distributions = target_net.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                        bootstrap,
                        discount,
                    )
                target_values = target_net.get_value(target_distributions)
                if args.min_q_target:
                    # Clipped double-Q target: per-sample, take whichever critic predicts the
                    # LOWER value and share its full categorical distribution as the target for
                    # every critic head's loss (reduces overestimation bias).
                    batch_size = target_values.shape[1]
                    min_idx = target_values.argmin(dim=0)  # [batch]
                    batch_idx = torch.arange(batch_size, device=self.device)
                    target_distributions = target_distributions[min_idx, batch_idx].unsqueeze(0).expand(
                        target_distributions.shape[0], -1, -1
                    )
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            q_values = qnet.get_value(F.softmax(q_outputs, dim=-1))
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

            bc_critic_loss = torch.tensor(0.0, device=self.device)
            if self.expert_critic is not None: #  and self.lambda_bc_critic > 0:
                with torch.no_grad():
                    expert_v_next = self.expert_critic(unnormed_next_critic_observations)  # [batch]
                    expert_target = rewards + discount * bootstrap * expert_v_next  # [batch]
                    # Project expert_target scalar onto atom support as a point distribution
                    batch_size = expert_target.shape[0]
                    num_atoms = qnet.num_atoms
                    delta_z = (qnet.v_max - qnet.v_min) / (num_atoms - 1)
                    t = expert_target.clamp(qnet.v_min, qnet.v_max)
                    b = (t - qnet.v_min) / delta_z  # [batch]
                    lower = torch.floor(b).long()
                    upper = torch.ceil(b).long()
                    # Edge case: b is exactly an integer — ensure lower != upper
                    is_integer = upper == lower
                    lower_mask = is_integer & (lower > 0)
                    upper_mask = is_integer & (lower == 0)
                    lower = torch.where(lower_mask, lower - 1, lower)
                    upper = torch.where(upper_mask, upper + 1, upper)
                    # Build point distribution [batch, num_atoms]
                    point_dist = torch.zeros(batch_size, num_atoms, device=self.device)
                    batch_idx = torch.arange(batch_size, device=self.device)
                    point_dist[batch_idx, lower] = upper.float() - b
                    point_dist[batch_idx, upper] = b - lower.float()
                # Cross-entropy against point distribution; q_outputs: [num_critics, batch, num_atoms]
                log_probs = F.log_softmax(q_outputs, dim=-1)  # [num_critics, batch, num_atoms]
                bc_critic_loss = -(point_dist.unsqueeze(0) * log_probs).sum(-1).mean() * self.lambda_bc_critic
                qf_loss += bc_critic_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

            scaler.unscale_(alpha_optimizer)

            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            q_values.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
            bc_critic_loss.detach(),
        )

    def _update_pol(self, data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            _, mean, log_std = actor(data["observations"])
            std = log_std.exp()
            with torch.no_grad():
                action_std = std.mean()
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            # Clipped double-Q: take the min across the critic ensemble instead of the mean,
            # matching the standard SAC actor-loss convention (reduces overestimation bias).
            qf_value = q_values.min(dim=0).values if args.min_q_target else q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

            bc_policy_loss = torch.tensor(0.0, device=self.device)
            if self.expert_policy is not None:
                with torch.no_grad():
                    expert_mean, expert_std = self.expert_policy(data["unnormed_observations"])
                expert_dist = torch.distributions.Normal(expert_mean, expert_std)
                current_dist = torch.distributions.Normal(mean, std)
                bc_policy_loss = torch.distributions.kl_divergence(expert_dist, current_dist).sum(-1).mean() * self.lambda_bc_policy
                actor_loss += bc_policy_loss

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
            bc_policy_loss.detach(),
        )

    def _sample_and_prepare_batches(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * num_updates

        if self.expert_rb is not None:
            # Mix expert/online per update using self.expert_ratio (linearly annealed if configured).
            # Expert rb may have a different n_env from the online env, so we
            # compute a per-env count that lands at ~expert_ratio of total samples.
            main_per_env = max(round(batch_size * (1.0 - self._current_expert_ratio)), 1)
            target_expert_total_per_update = (batch_size - main_per_env) * self.rb.n_env
            expert_per_env = max(target_expert_total_per_update // self.expert_rb.n_env, 1)

            per_update_batches = []
            for _ in range(num_updates):
                main_data = self.rb.sample(main_per_env)
                expert_data = self.expert_rb.sample(expert_per_env)
                per_update_batches.append(torch.cat([main_data, expert_data], dim=0))
            large_data = torch.cat(per_update_batches, dim=0)
            samples_per_update = large_data["actions"].shape[0] // num_updates
        else:
            large_data = self.rb.sample(large_batch_size)
            samples_per_update = batch_size * self.rb.n_env

        if self.config.use_symmetry:
            samples_per_update *= 2

            augmented_large_data: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {"next": {}}

            augmented_large_data["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["actions"] = self.symmetry_utils.augment_actions(actions=large_data["actions"])
            assert isinstance(augmented_large_data["next"], dict)
            augmented_large_data["next"]["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )
            augmented_large_data["next"]["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )

            # Calculate augmentation factor and repeat non-augmented data
            observations_tensor = augmented_large_data["observations"]
            assert isinstance(observations_tensor, torch.Tensor), (
                "observations should be a Tensor after data augmentation"
            )
            num_aug = int(observations_tensor.shape[0] / large_data["next"]["rewards"].shape[0])
            augmented_large_data["next"]["rewards"] = large_data["next"]["rewards"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["dones"] = large_data["next"]["dones"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["truncations"] = large_data["next"]["truncations"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["effective_n_steps"] = large_data["next"]["effective_n_steps"].repeat(num_aug)  # type: ignore[index]

            # Override large_data
            large_data = augmented_large_data

        # Normalize all data once
        large_data = large_data.to(self.device)
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        large_data["critic_observations"] = normalize_critic_obs(large_data["critic_observations"])
        large_data["next"]["critic_observations"] = normalize_critic_obs(large_data["next"]["critic_observations"])

        # Split into smaller batches
        prepared_batches = []

        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                    "critic_observations": large_data["critic_observations"][start_idx:end_idx],
                },
                batch_size=samples_per_update,
            )
            batch_data["next"]["critic_observations"] = large_data["next"]["critic_observations"][start_idx:end_idx]

            prepared_batches.append(batch_data)

        return prepared_batches

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        # Load checkpoint if specified
        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Handle DDP-wrapped models
        actor_state_dict = torch_checkpoint["actor_state_dict"]
        qnet_state_dict = torch_checkpoint["qnet_state_dict"]

        self.actor.load_state_dict(actor_state_dict)
        self.qnet.load_state_dict(qnet_state_dict)

        self.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        self.critic_obs_normalizer.load_state_dict(torch_checkpoint["critic_obs_normalizer_state"])
        self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        self.log_alpha.data.copy_(torch_checkpoint["log_alpha"].to(self.device))
        if self.config.reset_optimizers:
            # Skip the optimizer restore entirely: Adam exp_avg/exp_avg_sq start at zero, so the
            # effective step size is not inherited from the source run's gradient scale. Networks and
            # normalizers are still loaded above.
            logger.info(
                "reset_optimizers=True: optimizer + grad-scaler state NOT restored from checkpoint "
                "(Adam moments start from zero)"
            )
        else:
            self.actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
            self.q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
            self.alpha_optimizer.load_state_dict(torch_checkpoint["alpha_optimizer_state_dict"])
            self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])
        # Re-apply optimizer hyperparameters from config. Optimizer.load_state_dict replaces
        # param_groups wholesale, and lr/weight_decay live there -- so without this every resumed run
        # silently inherits the CHECKPOINT's learning rates and ignores agent.*_learning_rate.
        # weight_decay is only re-applied to the actor/critic optimizers, which are the ones
        # constructed with args.weight_decay (alpha_optimizer uses the AdamW default).
        for opt, lr, wd in (
            (self.actor_optimizer, self.config.actor_learning_rate, self.config.weight_decay),
            (self.q_optimizer, self.config.critic_learning_rate, self.config.weight_decay),
            (self.alpha_optimizer, self.config.alpha_learning_rate, None),
        ):
            for g in opt.param_groups:
                g["lr"] = lr
                if wd is not None and "weight_decay" in g:
                    g["weight_decay"] = wd
        logger.info(
            f"Optimizer hyperparameters re-applied from config: "
            f"critic_lr={self.config.critic_learning_rate}, actor_lr={self.config.actor_learning_rate}, "
            f"alpha_lr={self.config.alpha_learning_rate}, weight_decay={self.config.weight_decay}"
        )
        self.global_step = torch_checkpoint["global_step"]
        self._restore_env_state(torch_checkpoint.get("env_state"))

    def load_expert_replay_buffer(self, path: str | None) -> None:
        """Load an expert replay buffer saved by the UWLab play.py recorder.

        Expected payload layout (see ``record_transitions_to_replay_buffer``):
            {"buffer_tensors": {observations, actions, rewards, dones, truncations,
                                next_observations, critic_observations,
                                next_critic_observations, ptr},
             "metadata": {n_env, buffer_size, n_obs, n_act, n_critic_obs, ...}}
        """
        if not path:
            return

        payload = torch.load(path, map_location=self.rb_device, weights_only=False)
        tensors = payload["buffer_tensors"]
        meta = payload["metadata"]

        # Shape check against main rb dims to catch obvious mismatches early.
        if meta["n_obs"] != self.rb.n_obs or meta["n_act"] != self.rb.n_act:
            raise ValueError(
                f"Expert buffer dims (n_obs={meta['n_obs']}, n_act={meta['n_act']}) "
                f"do not match agent dims (n_obs={self.rb.n_obs}, n_act={self.rb.n_act})."
            )
        if meta["n_critic_obs"] != self.rb.n_critic_obs:
            raise ValueError(
                f"Expert buffer n_critic_obs={meta['n_critic_obs']} does not match "
                f"agent n_critic_obs={self.rb.n_critic_obs}."
            )

        expert_rb = SimpleReplayBuffer(
            n_env=meta["n_env"],
            buffer_size=meta["buffer_size"],
            n_obs=meta["n_obs"],
            n_act=meta["n_act"],
            n_critic_obs=meta["n_critic_obs"],
            n_steps=self.config.num_steps,
            gamma=self.config.gamma,
            device=self.rb_device,
        )
        expert_rb.observations.copy_(tensors["observations"].to(self.rb_device))
        expert_rb.actions.copy_(tensors["actions"].to(self.rb_device))
        expert_rb.rewards.copy_(tensors["rewards"].to(self.rb_device))
        expert_rb.dones.copy_(tensors["dones"].to(self.rb_device))
        expert_rb.truncations.copy_(tensors["truncations"].to(self.rb_device))
        expert_rb.next_observations.copy_(tensors["next_observations"].to(self.rb_device))
        expert_rb.critic_observations.copy_(tensors["critic_observations"].to(self.rb_device))
        expert_rb.next_critic_observations.copy_(tensors["next_critic_observations"].to(self.rb_device))
        expert_rb.ptr = int(tensors["ptr"])

        self.expert_rb = expert_rb
        logger.info(
            f"Loaded expert replay buffer from {path}: "
            f"n_env={meta['n_env']}, buffer_size={meta['buffer_size']}, ptr={expert_rb.ptr}"
        )

    def save_replay_buffer(self, path: str) -> None:
        """Save the online replay buffer to disk for later resumption."""
        rb = self.rb
        payload = {
            "observations": rb.observations.cpu(),
            "actions": rb.actions.cpu(),
            "rewards": rb.rewards.cpu(),
            "dones": rb.dones.cpu(),
            "truncations": rb.truncations.cpu(),
            "next_observations": rb.next_observations.cpu(),
            "critic_observations": rb.critic_observations.cpu(),
            "next_critic_observations": rb.next_critic_observations.cpu(),
            "ptr": rb.ptr,
            "n_env": rb.n_env,
            "buffer_size": rb.buffer_size,
            "global_step": self.global_step,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(payload, path)
        logger.info(f"Saved replay buffer ({rb.n_env * rb.buffer_size} transitions) to {path}")

    def load_replay_buffer(self, path: str) -> None:
        """Restore a previously saved online replay buffer.

        The saved buffer length need not match the live one: the first ``min(src, dst)`` slots are
        filled and the remainder is left zeroed for online data. ``ptr`` is set to the number of
        valid loaded entries, so sampling only draws from those and the next write appends after
        them. The env count must still match exactly (see scripts/.../slice_rb_envs.py).
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)
        rb = self.rb
        if int(payload["n_env"]) != rb.n_env:
            raise ValueError(
                f"replay buffer env count mismatch: file has n_env={int(payload['n_env'])}, "
                f"run has n_env={rb.n_env}. Slice the file with slice_rb_envs.py first."
            )
        n = min(int(payload["buffer_size"]), rb.buffer_size)
        for key in (
            "observations",
            "actions",
            "rewards",
            "dones",
            "truncations",
            "next_observations",
            "critic_observations",
            "next_critic_observations",
        ):
            getattr(rb, key)[:, :n].copy_(payload[key][:, :n].to(self.device))
        rb.ptr = min(int(payload["ptr"]), n)
        logger.info(
            f"Loaded replay buffer from {path}: "
            f"n_env={payload['n_env']}, buffer_size={payload['buffer_size']} -> {rb.buffer_size} "
            f"(filled first {n} slots), ptr={rb.ptr}, "
            f"saved at global_step={payload.get('global_step', 'unknown')}"
        )

    def learn(self) -> None:
        args = self.config
        device = self.device
        # if args.compile:
        #     update_main = torch.compile(self._update_main)
        #     update_pol = torch.compile(self._update_pol)
        #     policy = torch.compile(self.policy)
        #     normalize_obs = torch.compile(self.obs_normalizer.forward)
        #     normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        # else:
        update_main = self._update_main
        update_pol = self._update_pol
        policy = self.policy
        normalize_obs = self.obs_normalizer.forward
        normalize_critic_obs = self.critic_obs_normalizer.forward
        qnet = self.qnet
        qnet_target = self.qnet_target
        env = self.env
        rb = self.rb
        start_time = time.time()

        # no_learning: hold actor, critic, target critic and alpha fixed for the whole run.
        # Enforced HERE rather than in setup() on purpose -- load() calls
        # optimizer.load_state_dict(), which restores param_groups including `lr` from the
        # checkpoint and therefore silently overwrites any learning rate configured earlier. That
        # is why passing agent.*_learning_rate=0.0 on a resumed run did not actually freeze
        # anything. tau is applied directly in the target EMA below, so it is overridden too.
        # Forward passes still run, so Q/loss diagnostics keep logging; only updates are suppressed.
        tau_eff = args.tau
        if args.no_learning:
            for _opt in (self.actor_optimizer, self.q_optimizer, self.alpha_optimizer):
                for _g in _opt.param_groups:
                    _g["lr"] = 0.0
            tau_eff = 0.0
            logger.info(
                "no_learning=True: all optimizer learning rates forced to 0 and tau forced to 0 "
                "after checkpoint load; networks and alpha are frozen for this run."
            )

        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)

        dones = None
        # Initialize metrics that might not be updated every step
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        bc_policy_loss = torch.tensor(0.0, device=device)
        pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=self.global_step)
    
        # Resolve the update schedule once: `num_updates` gradient steps every `update_every` env
        # steps. These are independent knobs -- num_updates alone could only express "N per step"
        # or "1 per N steps", never "N per M steps".
        _interval = int(getattr(args, "update_interval", 1) or 1)
        if args.num_updates < 1:
            # Legacy encoding: a fractional count means one update every int(1/num_updates) steps.
            if _interval != 1:
                raise ValueError(
                    f"num_updates={args.num_updates} (<1) already encodes an interval of "
                    f"{int(1 / args.num_updates)} steps; combining it with update_interval="
                    f"{_interval} is ambiguous. Use num_updates>=1 with update_interval instead."
                )
            num_updates = 1
            update_every = int(1 / args.num_updates)
        else:
            num_updates = int(args.num_updates)
            update_every = max(_interval, 1)
        critic_updates = 0  # drives the delayed policy update; see the loop below
        logger.info(
            f"update schedule: {num_updates} update(s) every {update_every} env step(s) "
            f"-> replay ratio {num_updates / update_every:g} updates/step; "
            f"actor every {args.policy_frequency} critic updates"
        )


        # Logging stuff
        ep_return = torch.zeros(env.num_envs, device=device)
        ep_length = torch.zeros(env.num_envs, device=device)
        rewbuffer = deque(maxlen=1000)
        lenbuffer = deque(maxlen=1000)
        total_episodes = 0
        num_success_episodes_log = 0
        num_episodes_log = 0
        # Latest value of each "Curriculum/<term>/<key>" the env reported. Only tasks with an
        # active curriculum manager (the Finetune configs) emit these, so this stays empty and
        # logs nothing for every other task. The env only populates the episode log on steps
        # where something reset, hence the carry-forward.
        curriculum_log: dict[str, float] = {}
        # Loss scalars from the most recent update round. Logging fires on its own interval, which
        # includes steps where no update ran (and steps before learning_starts), so these cannot be
        # read straight off loop locals -- they would be undefined on the first logged step.
        last_update_stats: dict[str, float] = {}

        while self.global_step <= args.num_learning_iterations:
            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with torch.no_grad(), self._maybe_amp():
                norm_obs = normalize_obs(obs, update=False)
                actions = policy(obs=norm_obs, dones=dones)

            next_obs, rewards, dones, infos = env.step(actions.float())
            truncations = infos["time_outs"]
            next_critic_obs = infos["observations"]["critic"]

            # Logging stuff
            ep_return += rewards
            ep_length += 1
            rewbuffer.extend(ep_return[dones.bool()].cpu().tolist())
            lenbuffer.extend(ep_length[dones.bool()].cpu().tolist())
            # Denominator comes from ep_counted, not dones: the env drops episodes cut short by
            # first_episode_termination, so numerator and denominator must agree on which episodes
            # count. Using dones here would put those staggering kills back in as failures.
            num_episodes_log += int(infos.get("ep_counted", int(dones.sum())))
            total_episodes += dones.sum()
            num_success_episodes_log += infos["ep_success"].sum().cpu().item()
            # `metrics/` carries MultiResetManager's per-reset-type success rates, so a task with a
            # mixed reset distribution gets one success curve per path instead of a single pooled
            # number. Both prefixes are carried forward because the env only populates the episode
            # log on steps where something actually reset.
            for key, value in (infos.get("episode") or {}).items():
                if key.startswith(("Curriculum/", "metrics/")):
                    curriculum_log[key] = value.item() if torch.is_tensor(value) else float(value)
                elif key.startswith("Episode_Termination/"):
                    # Already a proportion, not a count: TerminationManager latches each env's most
                    # recently completed episode and reports the per-term mean over all envs.
                    term_name = key[len("Episode_Termination/") :]
                    curriculum_log[f"charts/termination_{term_name}"] = (
                        value.item() if torch.is_tensor(value) else float(value)
                    )
            ep_return[dones.bool()] = 0
            ep_length[dones.bool()] = 0

            # Compute 'true' next_obs and next_critic_obs for saving
            true_next_obs = torch.where(
                truncations[:, None] > 0, infos["observations"]["final"]["actor_obs"], next_obs
            )
            true_next_critic_obs = torch.where(
                truncations[:, None] > 0,
                infos["observations"]["final"]["critic_obs"],
                next_critic_obs,
            )
            # SGFT shaping is applied to what gets STORED, not to what gets logged: ep_return
            # above already accumulated the raw env reward, so reported episodic return stays
            # comparable across shaped and unshaped runs.
            store_rewards = rewards
            if self.sgft_enabled and self.sgft_shaping:
                store_rewards = self.sgft_shaped_rewards(
                    obs, critic_obs, true_next_obs, true_next_critic_obs,
                    rewards, dones, truncations, args.gamma,
                )

            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                    "next": {
                        "observations": true_next_obs,
                        "rewards": torch.as_tensor(store_rewards, device=device, dtype=torch.float),
                        "truncations": truncations.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(env.num_envs,),
                device=device,
            )
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs

            obs = next_obs
            critic_obs = next_critic_obs

            # Slice to the collecting envs; the rest contribute statistics only.
            rb.extend(transition if rb.n_env == env.num_envs else transition[: rb.n_env])

            # rb.n_env, not env.num_envs: sample() returns rb.n_env * batch_size rows, so dividing by
            # the stepping env count would shrink the real batch by the eval-env factor.
            batch_size = max(args.batch_size // rb.n_env // self.gpu_world_size, 1)
            # Updates fire on the update schedule; logging and checkpointing must not. Gating those
            # behind the same `continue` meant a point was only written when global_step hit a
            # multiple of BOTH update_every and the interval -- every 800 steps for
            # update_interval=160 / logging_interval=100 -- and nothing at all before learning_starts.
            if rb.ptr > args.learning_starts and not (update_every > 1 and self.global_step % update_every != 0):
                # Use batched sampling: sample once, normalize once, split into updates
                prepared_batches = self._sample_and_prepare_batches(
                    batch_size, num_updates, normalize_obs, normalize_critic_obs
                )
                for i, data in enumerate(prepared_batches):
                    # Data is already normalized, just run the updates
                    (
                        buffer_rewards,
                        critic_grad_norm,
                        qf_loss,
                        q_values,
                        qf_max,
                        qf_min,
                        alpha_loss,
                        bc_critic_loss,
                    ) = update_main(data)
                    if self.expert_critic is not None:
                        self.lambda_bc_critic *= 0.999

                    # Delayed policy update, counted over ALL critic updates rather than special-
                    # cased per schedule. The previous form divided by int(1/num_updates), which is
                    # 0 for any num_updates>=2 -- fine while such runs always took the other branch,
                    # fatal now that a round of N updates can also fire on an interval. Counting
                    # also fixes rounds shorter than policy_frequency, where the old within-round
                    # `i % policy_frequency == 1` test gave the wrong actor:critic ratio.
                    critic_updates += 1
                    if critic_updates % args.policy_frequency == 0:
                        actor_grad_norm, actor_loss, policy_entropy, action_std, bc_policy_loss = update_pol(data)
                        if self.expert_policy is not None:
                            self.lambda_bc_policy *= 0.999

                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - tau_eff)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=tau_eff)

                # Snapshot after the round so the logging block below, which also runs on steps with
                # no update, has something well-defined to write.
                last_update_stats["losses/qf1_values"] = q_values[0].mean().item()
                last_update_stats["losses/qf2_values"] = q_values[1].mean().item()
                last_update_stats["losses/qf_loss"] = qf_loss.item() / 2.0
                last_update_stats["losses/alpha"] = float(self.log_alpha.exp())
                if critic_updates >= args.policy_frequency:
                    last_update_stats["losses/actor_loss"] = actor_loss.item()
                if self.config.use_autotune:
                    last_update_stats["losses/alpha_loss"] = alpha_loss.item()

            if self.global_step % args.logging_interval == 0:
                for _k, _v in last_update_stats.items():
                    self.writer.add_scalar(_k, _v, self.global_step)
                sps = self.global_step / (time.time() - start_time)
                self.writer.add_scalar("charts/SPS", int(sps), self.global_step)
                # Use the realized replay ratio, not args.num_updates: with update_interval>1
                # the round only fires every update_every steps, so args.num_updates alone
                # overstates throughput by exactly that factor.
                samples_per_sec = int(
                    sps * (num_updates / update_every) * args.batch_size * env.num_envs
                )
                self.writer.add_scalar("charts/samples_per_sec", samples_per_sec, self.global_step)
                # Perf/total_fps: transitions/sec (matches old LoggingHelper convention for
                # direct comparison against Speed-Test / Speed-Test-Current runs).
                total_fps = int(sps * env.num_envs)
                self.writer.add_scalar("Perf/total_fps", total_fps, self.global_step)

                # Same floor as success_rate below. These deques (maxlen 1000) start empty on every
                # resume, so without it the first points average over a handful of episodes and swing
                # wildly -- and at low env counts a window may hold a single episode.
                if len(rewbuffer) >= MIN_EPISODES_TO_LOG:
                    self.writer.add_scalar("charts/episodic_return", statistics.mean(rewbuffer), self.global_step)
                    self.writer.add_scalar("charts/episodic_length", statistics.mean(lenbuffer), self.global_step)
                
                # Require a minimum sample before reporting a ratio. Counters deliberately carry over
                # when the threshold is not met, so the value is always over >= this many episodes.
                # Without it, windows holding a handful of episodes quantise hard: the first window
                # cannot contain a timeout (episodes are longer than logging_interval), so it holds
                # only early crashes and reads 0.00, while a 1-episode window reads 0.00 or 1.00.
                if num_episodes_log >= MIN_EPISODES_TO_LOG:
                    self.writer.add_scalar(
                        "charts/success_rate", num_success_episodes_log / num_episodes_log, self.global_step
                    )
                    num_success_episodes_log = 0
                    num_episodes_log = 0

                self.writer.add_scalar("charts/num_episodes", total_episodes, self.global_step)

                for key, value in curriculum_log.items():
                    self.writer.add_scalar(key, value, self.global_step)
            if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                if self.is_main_process:
                    logger.info(f"Saving model at global step {self.global_step}")
                    self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            if args.save_replay_buffer_interval > 0 and self.global_step > 0 and self.global_step % args.save_replay_buffer_interval == 0:
                if self.is_main_process:
                    rb_path = os.path.join(self.log_dir, f"replay_buffer_{self.global_step:07d}.pt")
                    logger.info(f"Saving replay buffer at global step {self.global_step}")
                    self.save_replay_buffer(rb_path)
                    # self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

            # Avoid global_step being incremented beyond args.num_learning_iterations, so that the final checkpoint is
            # saved at exactly args.num_learning_iterations. In the `while` condition, we check for self.global_step <=
            # args.num_learning_iterations, so that we have complete logging data at the final step too (assuming
            # `args.num_learning_iterations` is a multiple of `args.logging_interval`).
            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            # self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

    def save(self, path: str) -> None:  # type: ignore[override]
        env_state = self._collect_env_state()
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.log_alpha,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
            env_state=env_state or None,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
        )

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.unwrapped_env.reset_all()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.config.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.config.critic_obs_keys], dim=1),
        }

    def get_inference_critic(self, device: str | None = None) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        device = device or self.device
        qnet = self.qnet.to(device)
        critic_obs_normalizer = self.critic_obs_normalizer.to(device)
        qnet.eval()
        critic_obs_normalizer.eval()

        def critic_fn(critic_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            if self.obs_normalization:
                normalized_obs = critic_obs_normalizer(critic_obs, update=False)
            else:
                normalized_obs = critic_obs
            q_outputs = qnet(normalized_obs, actions)  # [num_critics, batch, num_atoms]
            q_dist = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_dist)  # [num_critics, batch]
            return q_values.mean(dim=0), q_dist  # [batch]

        return critic_fn

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        # Use the underlying module for inference
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.obs_normalization:
                normalized_obs = obs_normalizer(obs["actor_obs"], update=False)
            else:
                normalized_obs = obs["actor_obs"]
            # Actions are already scaled by the actor
            return policy(normalized_obs)[0]

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        # Use the underlying module for ONNX export
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                # Actions are already scaled by the actor
                return self.actor(normalized_obs)[0]

        return ActorWrapper(actor, obs_normalizer if self.obs_normalization else None)

    def extract_actor_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened actor observation tensor.

        Args:
            obs: Flattened actor observation tensor of shape [batch_size, actor_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'actor_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.actor_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in actor observations. "
                f"Available keys: {list(self.actor_obs_indices.keys())}"
            )

        indices = self.actor_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def extract_critic_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened critic observation tensor.

        Args:
            obs: Flattened critic observation tensor of shape [batch_size, critic_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'critic_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.critic_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in critic observations. "
                f"Available keys: {list(self.critic_obs_indices.keys())}"
            )

        indices = self.critic_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def get_actor_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about actor observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.actor_obs_indices.copy()

    def get_critic_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about critic observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.critic_obs_indices.copy()

    def export(self, onnx_file_path: str) -> None:
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `FastSACAgent.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        # Create dummy all-zero input for ONNX tracing.
        example_input_list = torch.zeros(1, self.actor_obs_dim, device="cpu")

        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                self.actor_onnx_wrapper,
                motion_command,
                onnx_file_path,
                self.device,
            )
        else:
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict={"actor_obs": example_input_list},
            )

        # Extract control gains and velocity limits & attach to onnx as metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.unwrapped_env)
        action_scales = getattr(self.unwrapped_env, "action_scales", None)
        if action_scales is None:
            action_scale_metadata: float | list[float] = float(self.env.robot_config.control.action_scale)
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        # Extract URDF text from the robot config
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "action_scale": action_scale_metadata,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.global_step))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        # Restore original training state
        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()

        obs = self.env.reset()

        for step in itertools.islice(itertools.count(), max_eval_steps):
            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            # Actions are already scaled by the actor
            actions = self.actor(normalized_obs)[0]

            actor_state = {"step": step, "actions": actions, "obs": obs}
            actor_state = self._pre_eval_env_step(actor_state)

            obs, _, _, _ = self.env.step(actor_state["actions"])
            actor_state["obs"] = obs
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb_name in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb_name], training_loop=self))

    def _pre_evaluate_policy(self):
        self.env.set_is_evaluating()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state
