from typing import List

from game.game import ProductOwnerGame

def build_end_game_message(game: ProductOwnerGame, reward, episode):
    sprint_n = game.context.current_sprint

    credit = game.context.credit

    termination = "none"
    if not game.context.is_new_game:
        termination = 'tutorial'
    if game.context.credit == 0:
        termination = 'credit paid'
    if game.context.customers <= 0:
        termination = 'customers lost'
    if game.context.is_victory:
        termination = "victory"
    if game.context.is_loss:
        termination = "lose"

    message = (
            f"episode: {(episode + 1):03d}\t"
            + f"total_reward: {reward:.2f}\t"
            + f"sprint_n: {sprint_n:02d}\t"
            + f"credit: {credit: 6d}\t"
            + f"termination: {termination}\t"
    )
    return message

def get_log_entry_creator(base_epoch_log_state, base_end_epoch_log_state):
    def create_log_entry(data):
        return LogEntry(base_epoch_log_state, base_end_epoch_log_state, data)
    return create_log_entry


class KeyLogState:
    DO_NOT_LOG = 0
    ONLY_LEN_LOG = 1
    FULL_LOG = 2

    @classmethod
    def is_valid(cls, state):
        return state in [cls.DO_NOT_LOG, cls.ONLY_LEN_LOG, cls.FULL_LOG]

    @classmethod
    def get_len(cls):
        return cls.FULL_LOG + 1


class LogEntry:
    def __init__(self, epoch_log_state, end_log_state, log: List):
        self._epoch_log_state = epoch_log_state
        self._end_log_state = end_log_state
        self.data = log

    def get_log_state(self, after_study):
        if after_study:
            return self._end_log_state
        return self._epoch_log_state

    def set_log_state(self, state, after_study):
        if not KeyLogState.is_valid(state):
            return
        if after_study:
            self._end_log_state = state
        else:
            self._epoch_log_state = state
