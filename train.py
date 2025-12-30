from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from bingo_env import BingoEnv
from bingo_wrapper import D4AugmentationWrapper
from bingo_policy import BingoCNNExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os


def evaluate_policy_parallel(model, eval_env_fn, num_envs=4, num_episodes=1000):
    """병렬 평가 - D4 wrapper 없이 순수 성능 측정"""
    eval_env = SubprocVecEnv([eval_env_fn for _ in range(num_envs)])
    eval_env = VecMonitor(eval_env)
    
    episode_counts = []
    actions_counts = np.zeros(num_envs, dtype=np.int32)

    episodes_finished = 0

    obs = eval_env.reset()
    action_masks = obs["action_mask"]
    
    while episodes_finished < num_episodes:
        actions, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, rewards, dones, infos = eval_env.step(actions)
        action_masks = obs["action_mask"]
        
        # 보관 액션은 턴 소모 없으므로, 위치 액션만 카운트
        for i, action in enumerate(actions):
            if action != 49:  # 보관 액션 아닌 경우만
                actions_counts[i] += 1

        for i, done in enumerate(dones):
            if done:
                episode_counts.append(actions_counts[i])
                actions_counts[i] = 0
                episodes_finished += 1

    eval_env.close()
    mean_actions = np.mean(episode_counts)
    return mean_actions


class EvalAndSaveBestCallback(BaseCallback):
    def __init__(self, eval_env_fn, eval_episodes=1000, eval_freq=10_000, 
                 save_path="best_model", num_eval_envs=4, verbose=1):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.num_eval_envs = num_eval_envs
        self.save_path = save_path
        self.best_mean_actions = float("inf")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        print("Do evaluation...")
        mean_actions = evaluate_policy_parallel(
            model=self.model,
            eval_env_fn=self.eval_env_fn,
            num_envs=self.num_eval_envs,
            num_episodes=self.eval_episodes
        )

        if self.verbose > 0:
            print(f"[Evaluation] Mean actions: {mean_actions:.2f} over {self.eval_episodes} episodes.")

        if mean_actions < self.best_mean_actions:
            self.best_mean_actions = mean_actions
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"[Saved] New best model with mean actions: {mean_actions:.2f}")

        return True


def make_env(size=7, use_augmentation=True):
    """환경 생성 함수"""
    def _init():
        env = BingoEnv(size)
        if use_augmentation:
            env = D4AugmentationWrapper(env)
        return env
    return _init


def make_eval_env(size=7):
    """평가용 환경 (augmentation 없음)"""
    return BingoEnv(size)


if __name__ == "__main__":
    num_envs = 8  # 병렬 환경 수 증가
    
    # 학습 환경 (D4 augmentation 적용)
    env = SubprocVecEnv([make_env(size=7, use_augmentation=True) for _ in range(num_envs)])
    env = VecMonitor(env)

    # MaskablePPO with improved hyperparameters
    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=2048,           # 더 많은 샘플로 분산 감소
        batch_size=256,         # 배치 크기 증가
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,          # 탐험 유지 (확률적 환경 대응)
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        verbose=1,
        device="cuda",
        policy_kwargs=dict(
            features_extractor_class=BingoCNNExtractor,
            features_extractor_kwargs=dict(
                hidden_channels=64,
                num_res_blocks=3,
                kernel_size=3,
                scalar_embed_dim=32,
                features_dim=256
            ),
            net_arch=dict(pi=[256, 128], vf=[256, 128])  # Separate policy/value networks
        )
    )

    callback = EvalAndSaveBestCallback(
        eval_env_fn=make_eval_env,
        eval_episodes=500,
        eval_freq=20_000,
        save_path="model/best_model",
        num_eval_envs=8
    )

    print("Starting training...")
    print(f"Num envs: {num_envs}")
    print(f"N steps: {model.n_steps}")
    print(f"Batch size: {model.batch_size}")
    print(f"Learning rate: {model.learning_rate}")
    
    model.learn(total_timesteps=2_000_000, callback=callback)
    model.save("model/ppo_bingo_final")

    # 최종 평가
    print("\n=== Final Evaluation ===")
    final_mean = evaluate_policy_parallel(
        model=model,
        eval_env_fn=make_eval_env,
        num_envs=8,
        num_episodes=1000
    )
    print(f"Final mean actions: {final_mean:.2f}")
