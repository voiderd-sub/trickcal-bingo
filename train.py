"""
Custom GPU-Native PPO Training for Bingo.

Stays entirely on GPU - no CPU/GPU transfer overhead.
Implements MaskablePPO with all operations on CUDA tensors.

REFACTORED: Uses num_patterns (1 or 2) instead of allow_store.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import yaml
import os
import time
from collections import deque
from typing import Dict, Tuple, Optional

from bingo_env import BingoEnvGPU
from bingo_policy import BingoCNNExtractor

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MaskablePPOPolicy(nn.Module):
    """
    Actor-Critic policy for MaskablePPO.
    Uses BingoCNNExtractor as feature extractor.
    """
    
    def __init__(
        self,
        observation_space,
        action_dim: int = 49,
        features_dim: int = 256,
        hidden_channels: int = 64,
        num_res_blocks: int = 3,
        kernel_size: int = 3,
        pattern_embed_dim: int = 32,
        num_patterns: int = 1,
        pi_layers: list = [256, 128],
        vf_layers: list = [256, 128],
    ):
        super().__init__()
        
        # Feature extractor
        self.features_extractor = BingoCNNExtractor(
            observation_space,
            features_dim=features_dim,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            kernel_size=kernel_size,
            pattern_embed_dim=pattern_embed_dim,
            num_patterns=num_patterns,
        )

        
        # Policy head
        pi_layers_list = []
        in_dim = features_dim
        for out_dim in pi_layers:
            pi_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim
        pi_layers_list.append(nn.Linear(in_dim, action_dim))
        self.policy_head = nn.Sequential(*pi_layers_list)
        
        # Value head
        vf_layers_list = []
        in_dim = features_dim
        for out_dim in vf_layers:
            vf_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim
        vf_layers_list.append(nn.Linear(in_dim, 1))
        self.value_head = nn.Sequential(*vf_layers_list)
        
        # Initialize orthogonally
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.policy_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            logits: (batch, action_dim) action logits
            values: (batch, 1) state values
        """
        features = self.features_extractor(obs)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values
    
    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation dict
            action_mask: (batch, n_actions) boolean mask of valid actions
            action: Optional action to evaluate (for training)
            deterministic: If True, return argmax action
        
        Returns:
            action: (batch,) sampled or given action
            log_prob: (batch,) log probability of action
            entropy: (batch,) entropy of policy
            value: (batch,) state value
        """
        logits, values = self.forward(obs)
        
        # Mask invalid actions
        masked_logits = logits.clone()
        masked_logits[~action_mask] = float('-inf')
        
        # Create distribution
        probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            if deterministic:
                action = masked_logits.argmax(dim=-1)
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, values.squeeze(-1)


class RolloutBuffer:
    """
    GPU-based rollout buffer for PPO.
    """
    
    def __init__(
        self,
        num_envs: int,
        n_steps: int,
        obs_shapes: Dict[str, tuple],
        n_actions: int,
        device: str = 'cuda',
    ):
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.ptr = 0
        self.full = False
        
        # Allocate buffers
        self.obs = {
            key: torch.zeros((n_steps, num_envs, *shape), device=self.device)
            for key, shape in obs_shapes.items()
        }
        self.actions = torch.zeros((n_steps, num_envs), dtype=torch.int64, device=self.device)
        self.action_masks = torch.zeros((n_steps, num_envs, n_actions), dtype=torch.bool, device=self.device)
        self.rewards = torch.zeros((n_steps, num_envs), device=self.device)
        self.dones = torch.zeros((n_steps, num_envs), dtype=torch.bool, device=self.device)
        self.values = torch.zeros((n_steps, num_envs), device=self.device)
        self.log_probs = torch.zeros((n_steps, num_envs), device=self.device)
        
        # GAE computation results
        self.advantages = torch.zeros((n_steps, num_envs), device=self.device)
        self.returns = torch.zeros((n_steps, num_envs), device=self.device)
    
    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        action_mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        """Add a transition to the buffer."""
        # Only store obs keys that exist in buffer (skip 'action_mask')
        for key in self.obs.keys():
            if key in obs:
                self.obs[key][self.ptr] = obs[key]
        self.actions[self.ptr] = action
        self.action_masks[self.ptr] = action_mask
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.ptr = 0
            self.full = True
    
    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute Generalized Advantage Estimation."""
        last_gae = torch.zeros(self.num_envs, device=self.device)
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.returns = self.advantages + self.values
    
    def get_batches(self, batch_size: int):
        """
        Generate random minibatches for training.
        
        Yields flattened batches of (obs, actions, masks, old_log_probs, advantages, returns)
        """
        total_samples = self.n_steps * self.num_envs
        indices = torch.randperm(total_samples, device=self.device)
        
        # Flatten all data
        flat_obs = {
            key: val.view(total_samples, *val.shape[2:]).float()
            for key, val in self.obs.items()
        }
        flat_actions = self.actions.view(total_samples)
        flat_masks = self.action_masks.view(total_samples, self.n_actions)
        flat_log_probs = self.log_probs.view(total_samples)
        flat_advantages = self.advantages.view(total_samples)
        flat_returns = self.returns.view(total_samples)
        
        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch_obs = {key: val[batch_indices] for key, val in flat_obs.items()}
            
            yield (
                batch_obs,
                flat_actions[batch_indices],
                flat_masks[batch_indices],
                flat_log_probs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices],
            )
    
    def reset(self):
        self.ptr = 0
        self.full = False


class PPOTrainer:
    """Custom GPU-native PPO trainer."""
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        env_cfg = self.cfg['env']
        ppo_cfg = self.cfg['ppo']
        policy_cfg = self.cfg['policy']
        train_cfg = self.cfg['training']
        
        self.device = torch.device(train_cfg['device'])
        self.num_envs = env_cfg['num_envs']
        self.n_steps = ppo_cfg['n_steps']
        self.batch_size = ppo_cfg['batch_size']
        self.n_epochs = ppo_cfg['n_epochs']
        self.gamma = ppo_cfg['gamma']
        self.gae_lambda = ppo_cfg['gae_lambda']
        self.clip_range = ppo_cfg['clip_range']
        self.ent_coef = ppo_cfg['ent_coef']
        self.vf_coef = ppo_cfg['vf_coef']
        self.max_grad_norm = ppo_cfg['max_grad_norm']
        self.total_timesteps = train_cfg['total_timesteps']
        self.num_patterns = env_cfg.get('num_patterns', 1)
        
        # Action dimension based on num_patterns
        self.n_actions = 49 * self.num_patterns
        
        # Create environment
        print(f"Creating GPU environment with {self.num_envs} envs...")
        print(f"  num_patterns: {self.num_patterns}")
        print(f"  action_dim: {self.n_actions}")
        self.env = BingoEnvGPU(
            num_envs=self.num_envs,
            device=str(self.device),
            use_augmentation=env_cfg['use_augmentation'],
            num_patterns=self.num_patterns,
        )
        
        # Create observation space for policy
        from gymnasium import spaces
        obs_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern_indices": spaces.Box(low=-1, high=4, shape=(self.num_patterns,), dtype=np.int64),
        })
        
        # Create policy
        print("Creating policy...")
        self.policy = MaskablePPOPolicy(
            obs_space,
            action_dim=self.n_actions,
            features_dim=policy_cfg['features_dim'],
            hidden_channels=policy_cfg['hidden_channels'],
            num_res_blocks=policy_cfg['num_res_blocks'],
            kernel_size=policy_cfg['kernel_size'],
            pattern_embed_dim=policy_cfg.get('pattern_embed_dim', 32),
            num_patterns=self.num_patterns,
            pi_layers=policy_cfg['pi_layers'],
            vf_layers=policy_cfg['vf_layers'],
        ).to(self.device)
        
        # Optimizer
        self.initial_lr = ppo_cfg['learning_rate']

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.initial_lr,
            weight_decay=1e-4,  # Small weight decay for regularization
        )
        
        # Rollout buffer
        obs_shapes = {
            'board': (7, 7),
            'pattern_indices': (self.num_patterns,),
        }
        self.buffer = RolloutBuffer(
            self.num_envs,
            self.n_steps,
            obs_shapes,
            n_actions=self.n_actions,
            device=str(self.device),
        )
        
        # Curriculum (LINEAR)
        curriculum_cfg = self.cfg.get('curriculum', {'enabled': False})
        self.curriculum_enabled = curriculum_cfg.get('enabled', False)
        self.curriculum_initial_min = curriculum_cfg.get('initial_min', 10)
        self.curriculum_initial_max = curriculum_cfg.get('initial_max', 20)
        self.curriculum_full_ratio = curriculum_cfg.get('full_training_ratio', 0.3)
        self.last_curriculum_min = -1
        self.last_curriculum_max = -1
        
        # Evaluation
        eval_cfg = self.cfg.get('eval', {})
        self.eval_freq = eval_cfg.get('freq', 500_000)
        self.eval_num_episodes = eval_cfg.get('num_episodes', 5000)
        self.eval_num_envs = eval_cfg.get('num_envs', 64)
        self.best_mean_actions = float('inf')
        self.last_eval_timestep = 0
        
        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.ep_rewards_buffer = torch.zeros(self.num_envs, device=self.device)
        self.ep_lengths_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # W&B
        wandb_cfg = self.cfg.get('wandb', {'enabled': False})
        self.use_wandb = wandb_cfg.get('enabled', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_cfg.get('project', 'bingo-rl'),
                entity=wandb_cfg.get('entity'),
                name=wandb_cfg.get('name'),
                config=self.cfg,
            )
    
    def _prepare_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare observation for policy (convert to float, exclude action_mask)."""
        return {
            'board': obs['board'].float(),
            'pattern_indices': obs['pattern_indices'].long(),
        }
    
    def _update_curriculum(self, timestep: int):
        """Update curriculum based on progress (LINEAR interpolation)."""
        if not self.curriculum_enabled or self.total_timesteps == 0:
            return
        
        progress = timestep / self.total_timesteps
        
        # Linear interpolation: progress 0→(1-full_ratio) maps to (initial_min/max)→(0,0)
        # progress (1-full_ratio)→1.0 stays at (0,0)
        if progress >= (1 - self.curriculum_full_ratio):
            min_t, max_t = 0, 0
        else:
            ratio = progress / (1 - self.curriculum_full_ratio)
            min_t = int(self.curriculum_initial_min * (1 - ratio))
            max_t = int(self.curriculum_initial_max * (1 - ratio))
        
        # Only update if changed
        if min_t != self.last_curriculum_min or max_t != self.last_curriculum_max:
            self.last_curriculum_min = min_t
            self.last_curriculum_max = max_t
            self.env.set_curriculum(min_t, max_t)
            print(f"[{timestep:,}] Curriculum: turns={min_t}~{max_t} (progress={progress:.1%})", flush=True)
            
            if self.use_wandb:
                wandb.log({
                    'curriculum/min_turns': min_t,
                    'curriculum/max_turns': max_t,
                })
    
    def _evaluate(self) -> tuple:
        """
        Evaluate policy without augmentation.
        Returns (mean_actions, std_actions).
        """
        # Create eval environment (no augmentation, same num_patterns setting)
        eval_env = BingoEnvGPU(
            num_envs=self.eval_num_envs,
            device=str(self.device),
            use_augmentation=False,  # No augmentation for eval
            num_patterns=self.num_patterns,
        )
        eval_env.reset()
        
        episode_action_counts = []
        action_counts = torch.zeros(self.eval_num_envs, dtype=torch.int32, device=self.device)
        
        # Run until enough episodes
        while len(episode_action_counts) < self.eval_num_episodes:
            obs = eval_env._get_obs()
            action_mask = eval_env.action_masks()
            
            obs_float = self._prepare_obs(obs)
            with torch.no_grad():
                action, _, _, _ = self.policy.get_action_and_value(
                    obs_float, action_mask, deterministic=True
                )
            
            # Count all actions (each is a pattern placement)
            action_counts += 1
            
            _, _, done, _, _ = eval_env.step(action)
            
            # Record finished episodes
            if done.any():
                for i in done.nonzero(as_tuple=True)[0]:
                    episode_action_counts.append(action_counts[i].item())
                    action_counts[i] = 0
        
        del eval_env
        torch.cuda.empty_cache()
        
        actions = episode_action_counts[:self.eval_num_episodes]
        
        return np.mean(actions), np.std(actions)
    
    def collect_rollouts(self):
        """Collect rollouts using current policy."""
        obs = self.env._get_obs()
        
        for step in range(self.n_steps):
            # Get action mask
            action_mask = self.env.action_masks()
            
            # Get action and value
            with torch.no_grad():
                obs_float = self._prepare_obs(obs)
                action, log_prob, _, value = self.policy.get_action_and_value(
                    obs_float, action_mask
                )
            
            # Step environment
            next_obs, reward, done, _, _ = self.env.step(action)
            
            # Update episode tracking
            self.ep_rewards_buffer += reward
            self.ep_lengths_buffer += 1
            
            # Log completed episodes
            if done.any():
                for i in done.nonzero(as_tuple=True)[0]:
                    self.episode_rewards.append(self.ep_rewards_buffer[i].item())
                    self.episode_lengths.append(self.ep_lengths_buffer[i].item())
                    self.ep_rewards_buffer[i] = 0
                    self.ep_lengths_buffer[i] = 0
            
            # Store transition
            self.buffer.add(obs, action, action_mask, reward, done, value, log_prob)
            
            obs = next_obs
        
        # Compute last value for GAE
        with torch.no_grad():
            obs_float = self._prepare_obs(obs)
            _, _, _, last_value = self.policy.get_action_and_value(
                obs_float, self.env.action_masks()
            )
        
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        
        return obs
    
    def train_epoch(self):
        """Train one epoch on collected rollouts."""
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        approx_kls = []
        total_losses = []
        
        for batch in self.buffer.get_batches(self.batch_size):
            obs, actions, masks, old_log_probs, advantages, returns = batch
            
            # Get current policy outputs
            _, new_log_probs, entropy, values = self.policy.get_action_and_value(
                obs, masks, action=actions
            )
            
            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean()
            clip_fractions.append(clip_frac.item())
            
            # Approx KL divergence
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            approx_kls.append(approx_kl.item())
            
            policy_loss_1 = ratio * advantages
            policy_loss_2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            total_losses.append(loss.item())
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(-entropy_loss.item())
        
        # Compute explained variance
        with torch.no_grad():
            all_returns = self.buffer.returns.view(-1)
            all_values = self.buffer.values.view(-1)
            var_returns = all_returns.var()
            explained_var = 1 - (all_returns - all_values).var() / (var_returns + 1e-8)
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions),
            'approx_kl': np.mean(approx_kls),
            'loss': np.mean(total_losses),
            'explained_variance': explained_var.item(),
        }
    
    def train(self):
        """Main training loop."""
        print("="*60)
        print("Starting Custom GPU PPO Training")
        print(f"  Num envs: {self.num_envs}")
        print(f"  N steps: {self.n_steps}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total timesteps: {self.total_timesteps}")
        print(f"  Num patterns: {self.num_patterns}")
        print("="*60)
        
        timestep = 0
        n_updates = 0
        start_time = time.time()
        
        # Initial reset
        self.env.reset()
        
        while timestep < self.total_timesteps:
            # Update curriculum
            self._update_curriculum(timestep)
            
            # Collect rollouts
            rollout_start = time.time()
            self.collect_rollouts()
            rollout_time = time.time() - rollout_start
            
            timestep += self.n_steps * self.num_envs
            
            # Train
            train_start = time.time()
            for epoch in range(self.n_epochs):
                train_info = self.train_epoch()
            train_time = time.time() - train_start
            
            self.buffer.reset()
            n_updates += 1
            
            # Logging
            elapsed = time.time() - start_time
            fps = timestep / elapsed
            
            ep_rew = np.mean(self.episode_rewards) if self.episode_rewards else 0
            ep_len = np.mean(self.episode_lengths) if self.episode_lengths else 0
            
            # Log every update
            if n_updates % 1 == 0:  # Every update
                progress = 100 * timestep / self.total_timesteps
                print(f"[{timestep:>12,} / {self.total_timesteps:,}] ({progress:5.2f}%) | "
                      f"FPS: {fps:>8,.0f} | "
                      f"ep_rew: {ep_rew:>7.2f} | ep_len: {ep_len:>5.1f} | "
                      f"p_loss: {train_info['policy_loss']:.4f} | "
                      f"v_loss: {train_info['value_loss']:.4f} | "
                      f"ent: {train_info['entropy']:.3f} | "
                      f"clip: {train_info['clip_fraction']:.3f}",
                      flush=True)
            
            # Log to wandb every update
            if self.use_wandb:
                wandb.log({
                    'rollout/ep_rew_mean': ep_rew,
                    'rollout/ep_len_mean': ep_len,
                    'train/policy_gradient_loss': train_info['policy_loss'],
                    'train/value_loss': train_info['value_loss'],
                    'train/entropy': train_info['entropy'],
                    'train/clip_fraction': train_info['clip_fraction'],
                    'train/approx_kl': train_info['approx_kl'],
                    'train/loss': train_info['loss'],
                    'train/explained_variance': train_info['explained_variance'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'time/fps': fps,
                    'time/rollout': rollout_time,
                    'time/train': train_time,
                })
            
            # Evaluation
            if timestep - self.last_eval_timestep >= self.eval_freq:
                self.last_eval_timestep = timestep
                mean_actions, std_actions = self._evaluate()
                
                print(f"[EVAL] mean_actions: {mean_actions:.2f} ± {std_actions:.2f} (best: {self.best_mean_actions:.2f})", flush=True)
                
                if self.use_wandb:
                    wandb.log({
                        'eval/mean_actions': mean_actions,
                        'eval/std_actions': std_actions,
                        'eval/best_mean_actions': min(self.best_mean_actions, mean_actions),
                    })
                
                if mean_actions < self.best_mean_actions:
                    self.best_mean_actions = mean_actions
                    save_path = self.cfg['training']['save_path'] + '.pt'
                    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                    torch.save(self.policy.state_dict(), save_path)
                    print(f"[SAVED] New best model: {mean_actions:.2f}", flush=True)
        
        # Save final model
        save_path = self.cfg['training']['final_model_path'] + '.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.policy.state_dict(), save_path)
        print(f"\nSaved model to {save_path}")
        
        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='config.yaml')
    args = parser.parse_args()
    
    trainer = PPOTrainer(args.config)
    trainer.train()
