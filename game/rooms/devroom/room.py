from game import game_global as Global


class OfficeRoom:
    def __init__(self, worker_count=0):
        self._worker_count = worker_count

        if self._worker_count == 0:
            self.can_buy_robot = False
            self.can_buy_room = True
        else:
            self.can_buy_room = False
            self.can_buy_robot = True

    def on_buy_robot_button_pressed(self):
        if not Global.has_enough_money(Global.NEW_WORKER_COST) or not self.can_buy_robot:
            return False

        self._worker_count += 1

        if self._worker_count == Global.MAX_WORKER_COUNT:
            self.can_buy_robot = False

        return True

    def on_buy_room_button_pressed(self):
        current_room_cost = Global.NEW_ROOM_COST * Global.current_room_multiplier
        if not Global.has_enough_money(current_room_cost) or not self.can_buy_room:
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
