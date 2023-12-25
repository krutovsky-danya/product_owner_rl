from game.rooms.devroom.room import OfficeRoom
from game.game_variables import GlobalContext
from typing import List


class Offices:
    def __init__(self, context: GlobalContext):
        self.context = context
        self.offices: List[OfficeRoom] = []
        self.offices.append(OfficeRoom(self.context.available_developers_count, self.context))
        for _ in range(9):
            self.offices.append(OfficeRoom(0, self.context))

        self.ready()

    def ready(self):
        if self.context.is_new_game:
            self.toggle_purchases(False)

    def toggle_purchases(self, mode: bool):
        for room in self.offices:
            room.toggle_purchases(mode)
