from game.rooms.devroom.room import OfficeRoom
from game import game_global as Global


class Offices:
    def __init__(self):
        self.offices = []
        self.offices.append(OfficeRoom(Global.available_developers_count))
        for _ in range(9):
            self.offices.append(OfficeRoom())

        self.ready()

    def ready(self):
        if Global.is_new_game:
            self.toggle_purchases(False)

    def toggle_purchases(self, mode: bool):
        for room in self.offices:
            room.toggle_purchases(mode)
