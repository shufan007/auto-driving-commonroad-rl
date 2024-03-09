"""
Module for training an agent using stable baselines
"""

import os
import sys
import time
from shutil import copyfile
import numpy as np
import copy
import torch
from datetime import datetime
import yaml
from typing import Callable
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3 import PPO
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv


def time2str(timestamp):
    d = datetime.fromtimestamp(timestamp)
    timestamp_str = d.strftime('%Y-%m-%d %H:%M:%S')
    return timestamp_str


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class ModelTrain(object):
    def __init__(self,
                 n_envs=8,
                 backbone='mlp',
                 n_steps=1024,
                 n_epochs=4,
                 batch_size=32,
                 lr_init=3e-4,
                 lr_decay_rate=0.95,
                 total_timesteps=1000_0000,
                 start_timesteps=0,
                 check_point_timesteps=10_0000,
                 model_path=None,
                 log_path=None,
                 data_path=None,
                 verbose=0,
                 archive_timesteps=20_0000,
                 archive_path=None):

        self.n_envs = n_envs
        self.backbone = backbone
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.total_timesteps = total_timesteps
        self.start_timesteps = start_timesteps
        self.check_point_timesteps = check_point_timesteps
        self.model_path = model_path
        self.log_path = log_path
        self.data_path = data_path
        self.verbose = verbose
        self.archive_timesteps = archive_timesteps
        self.archive_path = archive_path
        self.PolicyModel = PPO

        self.current_timesteps = min(total_timesteps, start_timesteps)
        self.progress_remaining = 1
        self.learning_rate = lr_init
        self.min_learning_rate = 5e-6

        self.meta_scenario_path = os.path.join(self.data_path, "highD/pickles/meta_scenario")
        self.training_data_path = os.path.join(self.data_path, "highD/pickles/problem_train")
        self.testing_data_path = os.path.join(self.data_path, "highD/pickles/problem_test")
        self.info_keywords = tuple([ "is_collision",
                                     "is_time_out",
                                     "is_off_road",
                                     "is_friction_violation",
                                     "is_goal_reached"])
        self.env_configs = {}
        self.hyperparams = {}
        self.load_rL_environment_and_model_settings()
        model_train_configs = {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "tensorboard_log": log_path
        }
        self.hyperparams.update(model_train_configs)

    def load_rL_environment_and_model_settings(self):
        # Read in environment configurations
        with open("commonroad_rl/gym_commonroad/configs.yaml", "r") as config_file:
            self.env_configs = yaml.safe_load(config_file)["env_configs"]
        # Change a configuration directly
        self.env_configs["reward_type"] = "hybrid_reward"

        # Save settings for later use
        os.makedirs(self.log_path, exist_ok=True)
        with open(os.path.join(self.log_path, "environment_configurations.yml"), "w") as config_file:
            yaml.dump(self.env_configs, config_file)

        # Read in model hyperparameters
        with open("commonroad_rl/hyperparams/ppo.yml", "r") as hyperparam_file:
            self.hyperparams = yaml.safe_load(hyperparam_file)["commonroad-v1"]

        # Save settings for later use
        with open(os.path.join(self.log_path, "model_hyperparameters.yml"), "w") as hyperparam_file:
            yaml.dump(self.hyperparams, hyperparam_file)

        # Remove `normalize` as it will be handled explicitly later
        if "normalize" in self.hyperparams:
            del self.hyperparams["normalize"]

    def create_training_environment(self):
        # Create a Gym-based RL environment with specified data paths and environment configurations

        # training_env = gym.make("commonroad-v1",
        #                         meta_scenario_path=meta_scenario_path,
        #                         train_reset_config_path= training_data_path,
        #                         **env_configs)

        env_kwargs = {"meta_scenario_path": self.meta_scenario_path,
                      "train_reset_config_path": self.training_data_path
                      }
        env_kwargs.update(self.env_configs)

        training_env = CommonroadEnv(**env_kwargs)

        # Wrap the environment with a monitor to keep an record of the learning process
        # training_env = Monitor(training_env, self.log_path + "0", info_keywords=self.info_keywords)

        # # Vectorize the environment with a callable argument
        # def make_training_env():
        #     return training_env
        #
        # training_env = DummyVecEnv([make_training_env])
        #
        # # Normalize observations and rewards
        # training_env = VecNormalize(training_env, norm_obs=True, norm_reward=True)

        if self.n_envs > 1:
            # multi-worker training (n_envs=4 => 4 environments)
            training_env = make_vec_env(CommonroadEnv, n_envs=self.n_envs, seed=None, env_kwargs=env_kwargs)
        return training_env

    def create_test_environment(self):
        # Create a Gym-based RL environment with specified data paths and environment configurations

        # testing_env = gym.make("commonroad-v1",
        #                         meta_scenario_path=meta_scenario_path,
        #                         test_reset_config_path= testing_data_path,
        #                         **env_configs_test)

        env_kwargs = {"meta_scenario_path": self.meta_scenario_path,
                      "test_reset_config_path": self.testing_data_path,
                      "test_env": True
                      }
        env_kwargs.update(self.env_configs)

        testing_env = CommonroadEnv(**env_kwargs)

        # Wrap the environment with a monitor to keep an record of the testing episodes
        log_path_test = "commonroad_rl/tutorials/logs/test"
        os.makedirs(log_path_test, exist_ok=True)

        testing_env = Monitor(testing_env, log_path_test + "/0", info_keywords=self.info_keywords)

        # # Vectorize the environment with a callable argument
        # def make_testing_env():
        #     return testing_env
        #
        # testing_env = DummyVecEnv([make_testing_env])
        #
        # # Normalize only observations during testing
        # testing_env = VecNormalize(testing_env, norm_obs=True, norm_reward=False, training=False)

        return testing_env

    def model_train_step(self, check_point_timesteps, save_model_path=None):
        # 更新学习率
        self.learning_rate_update()
        self.hyperparams.update({"learning_rate": self.learning_rate})

        training_env = self.create_training_environment()
        try:
            model = self.PolicyModel.load(self.model_path, env=training_env, learning_rate=self.learning_rate)
        except Exception:
            print(f"load model from self.model_path: {self.model_path} error")
            model = self.PolicyModel(env=training_env, **self.hyperparams)

        t0 = time.time()
        # model.learn(int(2e4))
        model.learn(total_timesteps=check_point_timesteps)
        model.save(self.model_path)
        if save_model_path is not None:
            model.save(save_model_path)
        print(f"train time: {time.time() - t0}")

    def learning_rate_update(self):
        self.progress_remaining = (self.total_timesteps - self.current_timesteps) / self.total_timesteps
        # self.learning_rate = self.lr_init * self.progress_remaining + self.min_learning_rate
        self.learning_rate = max(self.learning_rate * self.lr_decay_rate, self.min_learning_rate)
        print(f"progress_remaining: {self.progress_remaining}, learning_rate: {self.learning_rate}")

    def model_train(self):
        n_check_point = int(np.ceil(self.total_timesteps / self.check_point_timesteps))
        for i in range(n_check_point):
            self.current_timesteps += self.check_point_timesteps
            model_str = f"model_{int(self.current_timesteps / 10000)}w"
            save_model_path = os.path.join(self.log_path, model_str)
            self.model_train_step(self.check_point_timesteps, save_model_path)
            if self.archive_path and (self.current_timesteps % self.archive_timesteps == 0):
                print(f"self.archive_path: {self.archive_path}")
                print(f"current_timesteps: {self.current_timesteps}, self.archive_timesteps: {self.archive_timesteps}")
                source = save_model_path + '.zip'
                target = os.path.join(self.archive_path, model_str)
                try:
                    copyfile(source, target)
                except IOError as e:
                    print("Unable to copy file. %s" % e)

    def agent_play(self, model_path, max_step=100, verbose=0):

        env = CommonroadEnv(meta_scenario_path=self.meta_scenario_path,
                            train_reset_config_path=self.testing_data_path,
                            **self.env_configs)
        model = self.PolicyModel.load(model_path)

        step = 0
        ep_len = 0
        ep_rew = 0
        ep_cnt = 0
        total_ep_len = 0
        total_ep_rew = 0

        obs, info = env.reset()
        while step < max_step:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            print(f"-- step: {step}, action: {action}, reward: {reward}")
            if verbose >= 1:
                print(f"info: {info}")

            env.render("human")

            step += 1
            ep_len += 1
            ep_rew += reward
            if done:
                print(f"step: {step}, done: {done}")
                obs, info = env.reset()
                ep_cnt += 1
                total_ep_len += ep_len
                total_ep_rew += ep_rew

                ep_rew_mean = round(ep_rew/ep_len)
                print(f"ep_len: {ep_len}, ep_rew_mean: {ep_rew_mean}")
                ep_len = 0
                ep_rew = 0
        ep_len_mean = round(total_ep_len / ep_cnt, 1)
        ep_rew_mean = round(total_ep_rew / ep_cnt, 0)
        print(f"-- total -- ep_len_mean: {ep_len_mean}, ep_rew_mean: {ep_rew_mean}")


def sb3_model_to_pth_model(PolicyModel, model_path):
    ppo_model = PolicyModel.load(model_path)
    ## 保存pth模型
    torch.save(ppo_model.policy, model_path + '.pth')

def save_pth_model(model, save_model_path):
    torch.save(model.policy, save_model_path + '.pth')

def load_pth_model(pth_model_path):
    pth_model = torch.load(pth_model_path)
    return pth_model

def save_policy_model_state_dict(model, save_model_path):
    torch.save(model.policy.state_dict(), save_model_path + '_state_dict.pt')


def transfer_policy_model_to_state_dict(model_path):
    model = PPO.load(model_path)
    torch.save(model.policy.state_dict(), model_path + '_state_dict.pt')


def load_state_dict(policy_model, state_dict_path):
    policy_model.load_state_dict(torch.load(state_dict_path))


def task_args_parser(argv, usage=None):
    """
    :param argv:
    :return:
    """
    import argparse

    parser = argparse.ArgumentParser(prog='main', usage=usage, description='model train')

    # env config
    parser.add_argument('--n_envs', type=int, default=4, help="并行环境个数")
    parser.add_argument('--backbone', type=str, default='cnn', help="特征提取骨干网络， [cnn, resnet]")
    parser.add_argument('--n_steps', type=int, default=512, help="策略网络更新步数")
    parser.add_argument('--n_epochs', type=int, default=4, help="训练n_epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--lr_init', type=float, default=5e-4, help="初始学习率")
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help="学习率衰减因子")

    parser.add_argument('--total_timesteps', type=int, default=100_0000, help="训练步数")
    parser.add_argument('--check_point_timesteps', type=int, default=10_0000, help="检查点步数")
    parser.add_argument('--start_timesteps', type=int, default=0, help="本次训练开始index")

    parser.add_argument('--log_path', type=str, help='log路径')
    parser.add_argument('--data_path', type=str, help='训练数据路径')
    parser.add_argument('--test_model_path', type=str, help='测试模型路径')
    parser.add_argument('--test_step', type=int, default=100, help="测试执行步数")
    parser.add_argument('--archive_timesteps', type=int, default=50_0000, help="存档训练步数")
    parser.add_argument('--archive_path', type=str, default='/content/drive/MyDrive/models', help='存档模型路径')

    args = parser.parse_args()
    return args


def run_train(argv):
    usage = '''
    example:
    python model_train_main.py --lr_init 0.0004 --lr_decay_rate 0.95 --n_steps 1024 --n_epochs 4 --batch_size 64 --total_timesteps 20000000 --check_point_timesteps 100000 --n_envs 4 --start_timesteps 0 --log_path ppo_mlp
    test
    python model_train_main.py --test_step 100 --test_model_path D:\dev\mygithub\auto-driving-commonroad-rl\logs\ppo_mlp\model_10w
    '''

    args = task_args_parser(argv, usage)

    base_path = 'logs'
    log_path = args.log_path if args.log_path else f"ppo_{args.backbone}/"
    log_path = os.path.join(base_path, log_path)
    print(f"log_path: {log_path}")
    print(f"archive_path: {args.archive_path}")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    model_path = os.path.join(log_path, "model")
    print(f"model_path: {model_path}")
    data_path = args.data_path if args.data_path else "commonroad_rl/tutorials/data/"

    train_obj = ModelTrain(
        n_envs=args.n_envs,
        backbone=args.backbone,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr_init=args.lr_init,
        lr_decay_rate=args.lr_decay_rate,
        total_timesteps=args.total_timesteps,
        start_timesteps=args.start_timesteps,
        check_point_timesteps=args.check_point_timesteps,
        model_path=model_path,
        log_path=log_path,
        data_path=data_path,
        archive_timesteps=args.archive_timesteps,
        archive_path=args.archive_path)

    t0 = time.time()
    if args.test_model_path:
        train_obj.agent_play(model_path=args.test_model_path, max_step=args.test_step)
    else:
        train_obj.model_train()
    print(f"end time: {time2str(time.time())}, time_elapsed: {time.time() - t0}")


# Define a customized callback function to save the vectorized and normalized environment wrapper
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        save_path_name = os.path.join(self.save_path, "vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(save_path_name)
        print("Saved vectorized and normalized environment to {}".format(save_path_name))


if __name__ == '__main__':
    run_train(sys.argv[1:])


