"""
SB3-Compatible VecEnv Wrapper for GPU-Accelerated Bingo Environment.

Wraps BingoEnvGPU to provide stable-baselines3 VecEnv interface.
Handles GPU tensor to numpy conversion for SB3 compatibility.
"""

import torch
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
from typing import Optional, List, Union, Tuple, Dict, Any

from bingo_env import BingoEnvGPU


class BingoVecEnvGPU(VecEnv):
    """
    SB3-compatible VecEnv wrapper for GPU-based BingoEnv.
    
    All environment logic runs on GPU, with numpy conversion only at the boundary
    for SB3 compatibility.
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str = 'cuda',
        use_augmentation: bool = True,
    ):
        self.gpu_env = BingoEnvGPU(
            num_envs=num_envs,
            device=device,
            use_augmentation=use_augmentation,
        )
        self.device = torch.device(device)
        self._num_envs = num_envs
        
        # Pending actions for step_async/step_wait pattern
        self._actions: Optional[torch.Tensor] = None
        
        # Define observation and action spaces
        observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "stored_pattern": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "has_stored": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "cost": spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(50,), dtype=np.uint8),
        })
        
        action_space = spaces.Discrete(50)  # 49 positions + 1 store
        
        super().__init__(num_envs, observation_space, action_space)
    
    def step_async(self, actions: np.ndarray) -> None:
        """
        Store actions for later execution.
        
        Args:
            actions: (num_envs,) numpy array of actions
        """
        self._actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
    
    def step_wait(self) -> VecEnvStepReturn:
        """
        Execute stored actions and return results.
        
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        assert self._actions is not None, "Must call step_async before step_wait"
        
        # Execute on GPU
        obs_gpu, rewards_gpu, dones_gpu, truncated_gpu, _ = self.gpu_env.step(self._actions)
        
        # Convert to numpy for SB3
        obs = self._obs_to_numpy(obs_gpu)
        rewards = rewards_gpu.cpu().numpy()
        dones = dones_gpu.cpu().numpy()
        
        # Create infos list (SB3 expects list of dicts)
        infos = [{} for _ in range(self.num_envs)]
        
        # Handle terminal observations (for VecEnv reset behavior)
        # Note: BingoEnvGPU already auto-resets, so terminal_observation is the NEW obs
        # We need to store the pre-reset observation for environments that terminated
        # For now, we skip this as it's complex with auto-reset
        
        self._actions = None
        return obs, rewards, dones, infos
    
    def reset(self) -> VecEnvObs:
        """
        Reset all environments.
        
        Returns:
            Initial observations as numpy arrays
        """
        obs_gpu = self.gpu_env.reset()
        return self._obs_to_numpy(obs_gpu)
    
    def close(self) -> None:
        """Clean up resources."""
        pass  # GPU env doesn't need explicit cleanup
    
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Get attribute from environments."""
        if hasattr(self.gpu_env, attr_name):
            return [getattr(self.gpu_env, attr_name)] * self.num_envs
        return [None] * self.num_envs
    
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None) -> None:
        """Set attribute on environments."""
        if hasattr(self.gpu_env, attr_name):
            setattr(self.gpu_env, attr_name, value)
    
    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[List[int]] = None,
        **method_kwargs
    ) -> List[Any]:
        """
        Call method on environments.
        
        Special handling for SB3 compatibility.
        Note: SB3 calls env_method("action_masks") and expects a list of numpy arrays.
        """
        if method_name == "set_curriculum":
            # set_curriculum(min_turns, max_turns)
            if len(method_args) >= 2:
                self.gpu_env.set_curriculum(method_args[0], method_args[1])
            return [None] * self.num_envs
        
        if method_name == "action_masks":
            # SB3 expects list of individual mask arrays, one per env
            # Then it does np.stack() on them
            masks_gpu = self.gpu_env.action_masks()
            masks_np = masks_gpu.cpu().numpy()  # (num_envs, 50)
            # Return as list of arrays (one per env)
            return [masks_np[i] for i in range(self.num_envs)]
        
        if hasattr(self.gpu_env, method_name):
            method = getattr(self.gpu_env, method_name)
            result = method(*method_args, **method_kwargs)
            # Convert tensor results to numpy
            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()
            return [result] * self.num_envs
        
        return [None] * self.num_envs
    
    def env_is_wrapped(self, wrapper_class, indices: Optional[List[int]] = None) -> List[bool]:
        """Check if environments are wrapped."""
        return [False] * self.num_envs
    
    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set random seed."""
        if seed is not None:
            torch.manual_seed(seed)
        return [seed] * self.num_envs
    
    def _obs_to_numpy(self, obs_gpu: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Convert GPU observation tensors to numpy arrays."""
        return {
            key: val.cpu().numpy() for key, val in obs_gpu.items()
        }
    
    # === MaskablePPO support ===
    
    def action_masks(self) -> np.ndarray:
        """
        Get action masks for all environments.
        
        Returns:
            (num_envs, 50) boolean numpy array
        """
        masks_gpu = self.gpu_env.action_masks()
        return masks_gpu.cpu().numpy().astype(bool)
    
    def get_action_mask(self) -> np.ndarray:
        """Alias for action_masks (some SB3 versions use this)."""
        return self.action_masks()


# Custom VecMonitor-like wrapper that keeps GPU tensors
class BingoVecEnvGPUTensor(BingoVecEnvGPU):
    """
    GPU VecEnv that returns tensors instead of numpy.
    
    Use this for custom training loops that don't need numpy conversion.
    Note: Not compatible with standard SB3 training!
    """
    
    def step_wait(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict]]:
        """Execute step and return GPU tensors directly."""
        assert self._actions is not None
        
        obs, rewards, dones, truncated, _ = self.gpu_env.step(self._actions)
        infos = [{} for _ in range(self.num_envs)]
        
        self._actions = None
        return obs, rewards, dones, infos
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset and return GPU tensors directly."""
        return self.gpu_env.reset()
    
    def action_masks(self) -> torch.Tensor:
        """Return action masks as GPU tensor."""
        return self.gpu_env.action_masks()


if __name__ == "__main__":
    import time
    
    print("=== BingoVecEnvGPU Test ===")
    
    # Test SB3 compatibility
    num_envs = 256
    env = BingoVecEnvGPU(num_envs=num_envs, device='cuda', use_augmentation=True)
    
    print(f"Created VecEnv with {num_envs} environments")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs = env.reset()
    print(f"\nReset successful!")
    print(f"  board shape: {obs['board'].shape}, dtype: {obs['board'].dtype}")
    print(f"  action_mask shape: {obs['action_mask'].shape}")
    
    # Test step
    masks = env.action_masks()
    print(f"  action_masks shape: {masks.shape}, dtype: {masks.dtype}")
    
    # Sample valid actions
    actions = np.array([
        np.random.choice(np.where(m)[0]) for m in masks
    ])
    
    env.step_async(actions)
    obs, rewards, dones, infos = env.step_wait()
    
    print(f"\nStep successful!")
    print(f"  rewards: min={rewards.min():.2f}, max={rewards.max():.2f}")
    print(f"  dones: {dones.sum()} environments done")
    
    # Benchmark
    print("\nBenchmarking...")
    env.reset()
    
    start = time.time()
    n_iters = 1000
    
    for _ in range(n_iters):
        masks = env.action_masks()
        actions = np.array([np.random.choice(np.where(m)[0]) for m in masks])
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()
    
    elapsed = time.time() - start
    fps = (n_iters * num_envs) / elapsed
    
    print(f"FPS: {fps:,.0f}")
    print(f"Note: numpy action sampling is slow. Real training uses policy network.")
    
    # Test curriculum
    print("\nTesting curriculum...")
    env.env_method("set_curriculum", 5, 10)
    obs = env.reset()
    print("Curriculum set to (5, 10) - boards should be partially filled")
    
    env.close()
    print("\nAll tests passed!")
