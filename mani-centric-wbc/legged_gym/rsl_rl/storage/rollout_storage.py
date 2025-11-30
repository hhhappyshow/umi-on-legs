# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import copy
from typing import Any, Dict, Optional
import torch

from legged_gym.rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations: torch.Tensor
            self.critic_observations: torch.Tensor
            self.actions: torch.Tensor
            self.rewards: torch.Tensor
            self.dones: torch.Tensor
            self.values: torch.Tensor
            self.actions_log_prob: torch.Tensor
            self.action_mean: torch.Tensor
            self.action_sigma: torch.Tensor
            self.infos: Dict[str, Any]
            self.hidden_states: Optional[torch.Tensor] = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: int,
        action_shape: int,
        privileged_obs_shape: Optional[int] = None,
        device: str = "cpu",
    ):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.action_shape = action_shape

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, obs_shape, device=self.device
        )
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                privileged_obs_shape,
                device=self.device,
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, action_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.deltas = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, action_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, action_shape, device=self.device
        )

        self.infos = []

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.critic_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        # self.infos.append(copy.deepcopy(transition.infos))
        if transition.infos:
        # 只复制标量值，tensor 使用 detach().clone() 或直接引用
            info_copy = {}
            for k, v in transition.infos.items():
                if isinstance(v, torch.Tensor):
                    info_copy[k] = v.detach().clone() if v.requires_grad else v.clone()
                else:
                    info_copy[k] = v
            self.infos.append(info_copy)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0
        self.infos.clear()

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.deltas[step] = delta
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )
        return self.returns

    def mini_batch_generator(
        self, 
        num_mini_batches: int, 
        num_epochs: int,
        sampling_strategy: Optional[str] = None,
        sampling_weights: Optional[torch.Tensor] = None,
        reshuffle_per_epoch: bool = False,
    ):
        """
        生成 minibatch 的生成器
        
        Args:
            num_mini_batches: minibatch 的数量
            num_epochs: epoch 的数量
            sampling_strategy: 采样策略，可选值：
                - None 或 'random': 完全随机采样（默认）
                - 'advantage': 基于 advantages 的加权采样（高 advantage 的样本更可能被选中）
                - 'reward': 基于 rewards 的加权采样（高 reward 的样本更可能被选中）
                - 'delta': 基于 deltas 的加权采样（高 delta 的样本更可能被选中）
            sampling_weights: 自定义采样权重，形状为 [batch_size]，仅在 sampling_strategy='custom' 时使用
            reshuffle_per_epoch: 是否在每个 epoch 开始时重新打乱索引（默认 False，所有 epoch 使用相同顺序）
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        num_samples = num_mini_batches * mini_batch_size

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # 计算采样权重
        if sampling_strategy is None or sampling_strategy == 'random':
            # 随机采样，不需要权重
            weights = None
        elif sampling_strategy == 'advantage':
            # 基于 advantages 的加权采样
            # 将 advantages 转换为正权重（使用 softmax 或归一化）
            advantages_flat = advantages.squeeze(-1) if advantages.dim() > 1 else advantages
            # 使用 softmax 或简单的归一化
            weights = torch.softmax(advantages_flat / (advantages_flat.std() + 1e-8), dim=0)
        elif sampling_strategy == 'reward':
            # 基于 rewards 的加权采样
            rewards_flat = self.rewards.flatten(0, 1).squeeze(-1) if self.rewards.dim() > 1 else self.rewards.flatten(0, 1)
            # 将 rewards 转换为正权重
            rewards_min = rewards_flat.min()
            rewards_shifted = rewards_flat - rewards_min + 1e-8  # 确保为正
            weights = rewards_shifted / rewards_shifted.sum()
        elif sampling_strategy == 'delta':
            # 基于 deltas 的加权采样
            deltas_flat = self.deltas.flatten(0, 1).squeeze(-1) if self.deltas.dim() > 1 else self.deltas.flatten(0, 1)
            # 将 deltas 转换为正权重
            deltas_min = deltas_flat.min()
            deltas_shifted = deltas_flat - deltas_min + 1e-8  # 确保为正
            weights = deltas_shifted / deltas_shifted.sum()
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

        # 生成初始索引（用于向后兼容，当 reshuffle_per_epoch=False 时）
        if weights is not None:
            # ¼ÓÈ¨²ÉÑù£º´ÓÕû¸ö batch_size ÖÐ¸ù¾ÝÈ¨ÖØ²ÉÑù num_samples ¸öË÷Òý£¨ÓÐ·Å»Ø£©
            # initial_indices = torch.multinomial(
            #     weights, 
            #     num_samples=num_samples, 
            #     replacement=False
            # )
            # Ê¹ÓÃÄæÈ¨ÖØ²ÉÑù
            inv_weights = (weights.max() - weights).clamp_min(1e-8)
            inv_weights = inv_weights / inv_weights.sum()
            initial_indices = torch.multinomial(
                inv_weights,
                num_samples=num_samples,
                replacement=False
            )
            
        else:
            # Ëæ»ú²ÉÑù£º´Ó num_samples ¸öË÷ÒýÖÐËæ»úÅÅÁÐ£¨±£³ÖÔ­Âß¼­£©
            initial_indices = torch.randperm(
                num_samples, requires_grad=False, device=self.device
            )

        for epoch in range(num_epochs):
            # ¸ù¾Ý reshuffle_per_epoch ¾ö¶¨ÊÇ·ñÖØÐÂÉú³ÉË÷Òý
            if reshuffle_per_epoch or epoch == 0:
                # ÖØÐÂÉú³ÉË÷Òý
                if weights is not None:
                    # indices = torch.multinomial(
                    #     weights, 
                    #     num_samples=num_samples, 
                    #     replacement=False
                    # )
                    # Ê¹ÓÃÄæÈ¨ÖØ²ÉÑù
                    inv_weights = (weights.max() - weights).clamp_min(1e-8)
                    inv_weights = inv_weights / inv_weights.sum()
                    indices = torch.multinomial(
                        inv_weights,
                        num_samples=num_samples,
                        replacement=False
                    )
                else:
                    indices = torch.randperm(
                        num_samples, requires_grad=False, device=self.device
                    )
            else:
                # 复用初始索引（保持向后兼容）
                indices = initial_indices

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None

    # for RNNs only
    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.privileged_observations, self.dones
            )
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        assert (
            self.saved_hidden_states_a is not None
            and self.saved_hidden_states_c is not None
        )
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = torch.stack(
                    [
                        saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                            first_traj:last_traj
                        ]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_states in self.saved_hidden_states_a
                    ]
                )
                hid_c_batch = torch.stack(
                    [
                        saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                            first_traj:last_traj
                        ]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_states in self.saved_hidden_states_c
                    ]
                )
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_a_batch

                yield obs_batch, critic_obs, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj

    def obs_horizon_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int, horizon: int
    ):
        """
        Samples random obs and critic_obs `horizon` steps into the future
        """
        assert (
            horizon < self.num_transitions_per_env
        ), "Horizon must be smaller than the number of transitions per env"
        # contruct large tensor which concats all valid transitions
        priv_obs_buf: torch.Tensor = self.observations
        if self.privileged_observations is not None:
            priv_obs_buf = self.privileged_observations  # type: ignore

        staggered_indices = (
            torch.arange(horizon, device=self.device)[None, :].repeat(
                (self.num_transitions_per_env - horizon + 1), 1
            )
            + torch.arange(
                (self.num_transitions_per_env - horizon + 1), device=self.device
            )[:, None]
        )
        future_privileged_obs = priv_obs_buf[staggered_indices].permute(0, 2, 1, 3)

        batch_size = self.num_envs * (self.num_transitions_per_env - horizon)
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations[
            : (self.num_transitions_per_env - horizon + 1), ...
        ].flatten(0, 1)
        future_privileged_obs = future_privileged_obs.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield observations[batch_idx], future_privileged_obs[batch_idx]
