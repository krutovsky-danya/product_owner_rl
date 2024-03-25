from game.game_variables import GlobalContext
from game.game_constants import GlobalConstants


def get_current_room_cost(context: GlobalContext):
    return GlobalConstants.NEW_ROOM_COST * context.current_room_multiplier


def get_worker_cost():
    return GlobalConstants.NEW_WORKER_COST


class OfficeRoom:
    def __init__(self, worker_count, context: GlobalContext):
        self.context = context
        self._worker_count = worker_count

        if self._worker_count == 0:
            self.can_buy_robot = False
            self.can_buy_room = True
        else:
            self.can_buy_room = False
            self.can_buy_robot = True
    
    def get_workers_count(self):
        return self._worker_count

    def on_buy_robot_button_pressed(self):
        if not self.context.has_enough_money(get_worker_cost()) or not self.can_buy_robot:
            return False

        self._worker_count += 1

        if self._worker_count == GlobalConstants.MAX_WORKER_COUNT:
            self.can_buy_robot = False

        return True

    def on_buy_room_button_pressed(self):
        current_room_cost = get_current_room_cost(self.context)
        if not self.context.has_enough_money(current_room_cost) or not self.can_buy_room:
            return False

        self._worker_count += 1
        self.can_buy_room = False
        self.can_buy_robot = True

        return True

    def toggle_purchases(self, mode: bool):
        if mode:
            if self._worker_count == 0:
                self.can_buy_room = True
            else:
                self.can_buy_robot = True
        else:
            if self._worker_count == 0:
                self.can_buy_room = False
            else:
                self.can_buy_robot = False
