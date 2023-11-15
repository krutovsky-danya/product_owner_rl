from game.userstory_card.userstory_card import UserStoryCard
from game import game_global as Global
from game.userstories.userstories_generator import UserStoriesGenerator
from game.userstory_card.userstory_card_info import UserStoryCardInfo


class UserStories:
    def __init__(self):
        self.stories_list = []
        self.release = []

        self.user_survey_available = False
        self.statistical_research_available = True
        self.release_available = False
        self.available = True

        self.statistical_research_card_generator = UserStoriesGenerator(100, 0, 0, 0)
        self.user_survey_card_generator = UserStoriesGenerator(1, 59, 30, 10)

    def generate_cards_with_generator(self, count: int, gen: UserStoriesGenerator):
        cards = gen.generate_userstories(count)
        for card in cards:
            self.stories_list.append(card)
            Global.available_stories[id(card.info)] = card.info

    def add_us(self, card_info: UserStoryCardInfo):
        card = UserStoryCard(card_info)
        self.stories_list.append(card)
        Global.available_stories[id(card.info)] = card.info

    def generate_cards(self, gen: UserStoriesGenerator, cost: int, count: int):
        current_stories = len(Global.current_stories.values())
        available_stories = len(Global.available_stories.values())
        total_stories = current_stories + available_stories

        if total_stories >= 7:
            print("too many stories")
            return

        if not Global.has_enough_money(cost):
            print("not enough money")
            return

        Global.set_money(Global.get_money() - cost)
        self.generate_cards_with_generator(min(count, 7 - total_stories), gen)

    def on_start_release_pressed(self):
        if Global.is_new_game:
            self.statistical_research_available = False
            self.available = False
        Global.current_sprint_hours = self.calculate_hours_sum()
        return self.get_cards()

    def clear_release(self):
        self.release.clear()
        self.release_available = False

    def get_cards(self):
        cards_info = []

        for card in self.release:
            cards_info.append(card.info)

        return cards_info

    def calculate_hours_sum(self):
        need_hours_sum = Global.current_sprint_hours

        for card in self.release:
            us_label: UserStoryCardInfo = card.info
            if us_label is not None and us_label.label == "Bug":
                need_hours_sum += Global.available_developers_count / 2
            else:
                need_hours_sum += Global.available_developers_count

        return need_hours_sum

    def on_user_survey_pressed(self):
        self.generate_cards(self.user_survey_card_generator, Global.user_survey_cost, 1)

    def on_statistical_research_pressed(self):
        if Global.is_new_game:
            self.statistical_research_available = False
        self.generate_cards(self.statistical_research_card_generator,
                            Global.statistical_research_cost, 2)

    def on_stories_card_dropped(self, card: UserStoryCard):
        us_info: UserStoryCardInfo = card.info
        Global.available_stories[id(us_info)] = us_info
        self.stories_list.append(card)
        Global.current_stories.pop(id(us_info), None)
        self.release.remove(card)
        self.release_available = not (len(self.release) == 0)

    def on_release_card_dropped(self, card: UserStoryCard):
        us_info: UserStoryCardInfo = card.info
        Global.current_stories[id(us_info)] = us_info
        self.release.append(card)
        Global.available_stories.pop(id(us_info), None)
        self.stories_list.remove(card)
        self.release_available = not (len(self.release) == 0)

    def disable_restrictions(self):
        self.statistical_research_available = True
        self.available = True
        self.user_survey_available = True
        self.release_available = not (len(self.release) == 0)
