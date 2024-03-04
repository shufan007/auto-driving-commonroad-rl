"""
Module for training an agent using stable baselines
"""


import sys
from pathlib import Path
from warnings import warn
import os
import yaml
import copy
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3 import PPO
# import commonroad_rl.gym_commonroad
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv


class TrainMain:
    def __init__(self, log_path=None):

        self.env_configs = {}
        self.log_path = log_path
        if self.log_path is None:
            self.log_path = "commonroad_rl/tutorials/logs/"
        self.hyperparams = {}
        self.meta_scenario_path = "commonroad_rl/tutorials/data/highD/pickles/meta_scenario"
        self.training_data_path = "commonroad_rl/tutorials/data/highD/pickles/problem_train"
        self.info_keywords = tuple(["is_collision", \
                             "is_time_out", \
                             "is_off_road", \
                             "is_friction_violation", \
                             "is_goal_reached"])
        self.load_rL_environment_and_model_settings()

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

        training_env = CommonroadEnv(meta_scenario_path=self.meta_scenario_path,
                                     train_reset_config_path= self.training_data_path,
                                     **self.env_configs)

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


        # Append the additional key
        env_configs_test = copy.deepcopy(self.env_configs)
        env_configs_test["test_env"] = True

        # Create the testing environment
        testing_data_path = "commonroad_rl/tutorials/data/highD/pickles/problem_test"
        # testing_env = gym.make("commonroad-v1",
        #                         meta_scenario_path=meta_scenario_path,
        #                         test_reset_config_path= testing_data_path,
        #                         **env_configs_test)

        testing_env = CommonroadEnv(meta_scenario_path=self.meta_scenario_path,
                                    test_reset_config_path=testing_data_path,
                                    **env_configs_test)

        # Wrap the environment with a monitor to keep an record of the testing episodes
        log_path_test = "commonroad_rl/tutorials/logs/test"
        os.makedirs(log_path_test, exist_ok=True)

        testing_env = Monitor(testing_env, log_path_test + "/0", info_keywords=self.info_keywords)

        #
        # # Vectorize the environment with a callable argument
        # def make_testing_env():
        #     return testing_env
        #
        # testing_env = DummyVecEnv([make_testing_env])
        #
        # # Normalize only observations during testing
        # testing_env = VecNormalize(testing_env, norm_obs=True, norm_reward=False, training=False)

        return training_env, testing_env


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


def create_model_and_start_learning():

    log_path = "commonroad_rl/tutorials/logs/"
    train_obj = TrainMain(log_path=log_path)
    training_env, testing_env = train_obj.create_training_environment()

    # Pass the testing environment and customized saving callback to an evaluation callback
    # Note that the evaluation callback will triggers three evaluating episodes after every 500 training steps
    save_vec_normalize_callback = SaveVecNormalizeCallback(save_path=log_path)
    eval_callback = EvalCallback(testing_env,
                                 log_path=log_path,
                                 eval_freq=500,
                                 n_eval_episodes=3,
                                 callback_on_new_best=save_vec_normalize_callback)

    # Create the model together with its model hyperparameters and the training environment
    model = PPO(env=training_env, **train_obj.hyperparams)

    # Start the learning process with the evaluation callback
    n_timesteps = 1000000
    # model.learn(n_timesteps, eval_callback)
    model.learn(n_timesteps)


if __name__ == "__main__":
    create_model_and_start_learning()

