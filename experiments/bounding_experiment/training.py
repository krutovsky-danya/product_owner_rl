import datetime
import os
import sys

sys.path.append("..")
sys.path.append("../..")

from environment import CreditPayerEnv
from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from environment.reward_sytem import (
    EmpiricalCreditStageRewardSystem,
    BoundedEmpiricalCreditStageRewardSystem,
)
from pipeline.aggregator_study import update_reward_system_config
from pipeline import LoggingStudy
from experiments.training_utils import eval_agent
from main import create_usual_agent


def make_credit_study(trajectory_max_len, episode_n, bounded):
    userstory_env = UserstoryEnv(2, 0, 0)
    backlog_env = BacklogEnv(6, 0, 0, 0, 0, 0)
    if bounded:
        reward_system = BoundedEmpiricalCreditStageRewardSystem(True, config={})
    else:
        reward_system = EmpiricalCreditStageRewardSystem(True, config={})
    env = CreditPayerEnv(
        userstory_env,
        backlog_env,
        with_end=True,
        with_info=True,
        reward_system=reward_system,
        seed=None,
        card_picker_seed=None
    )
    update_reward_system_config(env, reward_system)

    agent = create_usual_agent(env, trajectory_max_len, episode_n)

    study = LoggingStudy(env, agent, trajectory_max_len)
    study.study_agent(episode_n)

    return study


def main(bound):
    episode_n = 1501
    logs_dir = f"episodes_{episode_n}"
    study = make_credit_study(200, episode_n, bound)
    now = datetime.datetime.now().strftime("%Y-%m-%d-T-%H-%M-%S")
    current_dir = os.getcwd()
    if "experiments" not in current_dir:
        current_dir = os.path.join(current_dir, "experiments", "bounding_experiment")
    current_dir = os.path.join(current_dir, logs_dir)
    os.makedirs(current_dir, exist_ok=True)
    filename = os.path.join(current_dir, f"bound_{bound}_rewards_{now}.txt")
    with open(filename, "w") as file:
        print(study.rewards_log, file=file)

    evaluations = []
    for i in range(100):
        evaluation = eval_agent(study)
        evaluations.append(evaluation)

    filename = os.path.join(current_dir, f"bound_{bound}_evals_{now}.txt")
    with open(filename, "w") as file:
        print(evaluations, file=file)


if __name__ == "__main__":
    n = 5
    for i in range(n):
        main(False)
        main(True)
