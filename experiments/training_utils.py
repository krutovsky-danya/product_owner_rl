import datetime
import os
import sys

sys.path.append("..")

from environment import CreditPayerEnv
from pipeline import LoggingStudy


def play_forward_with_empty_sprints(env: CreditPayerEnv):
    info = env.get_info()
    done = env.get_done(info)
    total_reward = 0
    context = env.game.context
    while not context.done and context.customers > 0:
        state, reward, done, info = env.step(0)
        total_reward += reward
    if context.customers <= 0:
        context.done = True
        context.is_loss = True


def eval_agent(study: LoggingStudy):
    study.agent.eval()
    state = study.env.reset(seed=None, card_picker_seed=None)
    info = study.env.get_info()
    reward, _ = study.play_trajectory(state, info)
    play_forward_with_empty_sprints(study.env)
    game_context = study.env.game.context
    is_win = game_context.is_victory
    sprint = game_context.current_sprint
    return reward, is_win, sprint
