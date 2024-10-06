import unittest
from environment.backlog_env import BacklogEnv
from game.backlog.backlog import Backlog
from game.backlog_card.backlog_card import Card
from game.backlog_card.card_info import CardInfo
from game.game_constants import UserCardType
from game.game_variables import GlobalContext
from game.userstory_card.bug_user_story_info import BugUserStoryInfo
from game.userstory_card.tech_debt_user_story_info import TechDebtInfo
from game.userstory_card.userstory_card_info import UserStoryCardInfo


class TestBacklogEnv(unittest.TestCase):
    def setUp(self):
        self.context = GlobalContext(seed=0, card_picker_seed=None)
        self.color_storage = self.context.color_storage
        self.backlog = Backlog(self.context)
        self.env = BacklogEnv(
            backlog_commons_count=1, backlog_bugs_count=1, backlog_tech_debt_count=1,
            sprint_commons_count=1, sprint_bugs_count=1, sprint_tech_debt_count=1
        )
        self.size = self.env.backlog_space_dim + self.env.sprint_space_dim
    
    def test_encode_empty_backlog(self):
        encoding = self.env.encode(self.backlog)

        self.assertSequenceEqual(encoding, [0] * self.size)
    
    def test_encode_full_backlog(self):
        queue = self.backlog.backlog
        self.fill_queue(queue)
        
        encoding = self.env.encode(self.backlog)
        backlog = encoding[:self.env.backlog_space_dim]
        sprint = encoding[self.env.backlog_space_dim:]

        self.assertSequenceEqual(backlog, [1, 2, 1, 0, 0.07, 2.5, 1, 2, 3, 0, -0.05, -0.5, 3, 4, 0, 1])
        self.assertSequenceEqual(sprint, [0] * self.env.sprint_space_dim)
    
    def test_encode_full_sprint(self):
        self.fill_queue(self.backlog.sprint)
        encoding = self.env.encode(self.backlog)
        backlog = encoding[:self.env.backlog_space_dim]
        sprint = encoding[self.env.backlog_space_dim:]
        
        self.assertSequenceEqual(backlog, [0] * self.env.backlog_space_dim)
        self.assertSequenceEqual(sprint, [1, 2, 1, 0, 0.07, 2.5, 1, 2, 3, 0, -0.05, -0.5, 3, 4, 0, 1])

    def fill_queue(self, queue):
        self.context.current_stories[1] = UserStoryCardInfo("S", 0, self.color_storage,
                                                            self.context.random_gen)
        for i in range(self.env.backlog_commons_count):
            card = Card()
            card_info = CardInfo(1, None, 1, None, UserCardType.S)
            card_info.hours += 1
            card.add_data(card_info)
            queue.append(card)
        
        self.context.current_stories[2] = BugUserStoryInfo(0, self.color_storage,
                                                           self.context.random_gen)
        for i in range(self.env.backlog_bugs_count):
            card = Card()
            card_info = CardInfo(2, None, 2, None, UserCardType.BUG)
            card_info.hours += 1
            card.add_data(card_info)
            queue.append(card)
        
        self.context.current_stories[3] = TechDebtInfo(0, self.color_storage,
                                                       self.context.random_gen)
        for i in range(self.env.backlog_tech_debt_count):
            card = Card()
            card_info = CardInfo(3, None, 3, None, UserCardType.TECH_DEBT)
            card_info.hours += 1
            card.add_data(card_info)
            queue.append(card)
