import unittest
from environment.backlog_env import BacklogEnv
from game.backlog.backlog import Backlog
from game.backlog_card.backlog_card import Card
from game.backlog_card.card_info import CardInfo
from game.game_constants import UserCardType
from game.game_variables import GlobalContext


class TestBacklogEnv(unittest.TestCase):
    def setUp(self):
        self.context = GlobalContext()
        self.backlog = Backlog(self.context)
        self.env = BacklogEnv(backlog_commons_count=1, backlog_bugs_count=1, backlog_tech_debt_count=1,
                 sprint_commons_count=1, sprint_bugs_count=1, sprint_tech_debt_count=1)
        self.size = self.env.backlog_space_dim + self.env.sprint_space_dim
    
    def test_encode_empty_backlog(self):
        encoding = self.env.encode(self.backlog)

        self.assertSequenceEqual(encoding, [0] * self.size)
    
    def test_encode_full_backlog(self):
        queue = self.backlog.backlog
        self.fill_queue(queue)
        
        encoding = self.env.encode(self.backlog)

        self.assertSequenceEqual(encoding, [1, 2, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0, 0, 0])
    
    def test_encode_full_sprint(self):
        self.fill_queue(self.backlog.sprint)
        encoding = self.env.encode(self.backlog)

        self.assertSequenceEqual(encoding, [0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 3, 3, 4])

    def fill_queue(self, queue):
        for i in range(self.env.backlog_commons_count):
            card = Card()
            card_info = CardInfo(1, None, None, None, UserCardType.S)
            card_info.hours += 1
            card.add_data(card_info)
            queue.append(card)
        
        for i in range(self.env.backlog_bugs_count):
            card = Card()
            card_info = CardInfo(2, None, None, None, UserCardType.BUG)
            card_info.hours += 1
            card.add_data(card_info)
            queue.append(card)
        
        for i in range(self.env.backlog_tech_debt_count):
            card = Card()
            card_info = CardInfo(3, None, None, None, UserCardType.TECH_DEBT)
            card_info.hours += 1
            card.add_data(card_info)
            queue.append(card)

