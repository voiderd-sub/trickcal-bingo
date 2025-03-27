from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from bingo_env import BingoEnv
from bingo_policy import BingoCNNExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os


def evaluate_policy_parallel(model, eval_env_fn, num_envs=4, num_episodes=1000):
    # 벡터화된 평가 환경 생성 (여기서는 DummyVecEnv 사용)
    eval_env = SubprocVecEnv([eval_env_fn for _ in range(num_envs)])
    eval_env = VecMonitor(eval_env)
    
    episode_counts = []  # 각 에피소드의 액션 횟수를 저장할 리스트

    # 각 환경의 현재 에피소드에서 사용한 액션 수를 저장할 배열
    # 초기에는 모두 0
    actions_counts = np.zeros(num_envs, dtype=np.int32)

    # 각 환경의 에피소드 종료 여부
    dones = [False] * num_envs
    # 전체 에피소드 집계
    episodes_finished = 0

    # 환경 초기화
    obs = eval_env.reset()
    action_masks = obs["action_mask"]
    
    while episodes_finished < num_episodes:
        # 모델로부터 행동 예측 (병렬적으로 모든 환경에서 실행)
        actions, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, rewards, dones, infos = eval_env.step(actions)
        action_masks = obs["action_mask"]
        
        # 각 환경마다 액션 횟수를 증가
        actions_counts += 1
        
        # 종료된 환경 처리: dones는 벡터 형태
        for i, done in enumerate(dones):
            if done:
                # 종료된 환경의 액션 횟수를 기록하고 리셋
                episode_counts.append(actions_counts[i])
                actions_counts[i] = 0  # 해당 환경의 액션 수 초기화
                episodes_finished += 1
                # 필요시 각 환경을 개별 리셋할 수도 있음
                # DummyVecEnv는 자동으로 리셋하므로 별도 처리 필요 없음

    eval_env.close()
    mean_actions = np.mean(episode_counts)
    return mean_actions

class EvalAndSaveBestCallback(BaseCallback):
    def __init__(self, eval_env_fn, eval_episodes=1000, eval_freq=10_000, save_path="best_model", num_eval_envs=4, verbose=1):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.num_eval_envs = num_eval_envs
        self.save_path = save_path
        self.best_mean_actions = float("inf")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

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


def make_env(size=7):
    def _init():
        env = BingoEnv(size)
        return env
    return _init

if __name__ == "__main__":
    num_envs = 4
    env = SubprocVecEnv([make_env(size=7) for _ in range(num_envs)])
    env = VecMonitor(env)

    model = MaskablePPO(
        learning_rate = 5e-5,
        policy=MaskableActorCriticPolicy,
        env=env,
        verbose=1,
        device="cuda",
        policy_kwargs=dict(
            features_extractor_class=BingoCNNExtractor,
            features_extractor_kwargs=dict(
                cnn_channels=[32,64],
                kernel_size=3,
                cost_embed_dim=16,
                features_dim=128
            )
        )
    )
    def make_eval_env():
        return BingoEnv(7)

    callback = EvalAndSaveBestCallback(
        eval_env_fn=make_eval_env,
        eval_episodes=1_000,
        eval_freq=25_000,
        save_path="checkpoints/best_model",
        num_eval_envs=4
    )

    model.learn(total_timesteps=5_000_000, callback=callback)
    model.save("ppo_bingo_cnn")

    eval_env = BingoEnv(7)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        eval_env.render()
