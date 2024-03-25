from game.userstory_card.userstory_card import UserStoryCard
from game.game_variables import GlobalContext
from game.game_constants import GlobalConstants
from game.userstories.userstories_generator import UserStoriesGenerator
from game.userstory_card.userstory_card_info import UserStoryCardInfo


class UserStories:
    def __init__(self, context: GlobalContext):
        self.context = context

        self.stories_list = []
        self.release = []

        self.user_survey_available = False
        self.statistical_research_available = True
        self.release_available = False
        self.available = True

        self.statistical_research_card_generator = UserStoriesGenerator(
            100, 0, 0, 0)
        self.user_survey_card_generator = UserStoriesGenerator(1, 59, 30, 10)

    def generate_cards_with_generator(self, count: int, gen: UserStoriesGenerator):
        cards = gen.generate_userstories(
            count, self.context.current_sprint, self.context.color_storage)
        for card in cards:
            self.stories_list.append(card)
            self.context.available_stories[id(card.info)] = card.info

    def add_us(self, card_info: UserStoryCardInfo):
        card = UserStoryCard(card_info)
        self.stories_list.append(card)
        self.context.available_stories[id(card.info)] = card.info

    def _can_generate_cards(self, total_stories: int, cost: int):
        return total_stories < 7 and self.context.has_enough_money(cost)

    def can_generate_cards_user_survey(self):
        total_stories = self.get_total_stories_count()
        return self._can_generate_cards(total_stories, GlobalConstants.user_survey_cost)

    def can_generate_cards_statistical_research(self):
        total_stories = self.get_total_stories_count()
        return self._can_generate_cards(total_stories, GlobalConstants.statistical_research_cost)

    def generate_cards(self, gen: UserStoriesGenerator, cost: int, count: int):
        total_stories = self.get_total_stories_count()
        if not self._can_generate_cards(total_stories, cost):
            return

        self.context.set_money(self.context.get_money() - cost)
        self.generate_cards_with_generator(min(count, 7 - total_stories), gen)

    def on_start_release_pressed(self):
        if self.context.is_new_game:
            self.statistical_research_available = False
            self.available = False
        self.context.current_sprint_hours = self.calculate_hours_sum()
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
        need_hours_sum = self.context.current_sprint_hours

        for card in self.release:
            us_label: UserStoryCardInfo = card.info
            if us_label is not None and us_label.label == "Bug":
                need_hours_sum += self.context.available_developers_count / 2
            else:
                need_hours_sum += self.context.available_developers_count

        return need_hours_sum

    def on_user_survey_pressed(self):
        self.generate_cards(self.user_survey_card_generator,
                            GlobalConstants.user_survey_cost, 1)

    def on_statistical_research_pressed(self):
        if self.context.is_new_game:
            self.statistical_research_available = False
        self.generate_cards(self.statistical_research_card_generator,
                            GlobalConstants.statistical_research_cost, 2)

    def on_stories_card_dropped(self, card: UserStoryCard):
        us_info: UserStoryCardInfo = card.info
        self.context.available_stories[id(us_info)] = us_info
        self.stories_list.append(card)
        self.context.current_stories.pop(id(us_info), None)
        self.release.remove(card)
        self.release_available = not (len(self.release) == 0)

    def on_release_card_dropped(self, card: UserStoryCard):
        us_info: UserStoryCardInfo = card.info
        self.context.current_stories[id(us_info)] = us_info
        self.release.append(card)
        self.context.available_stories.pop(id(us_info), None)
        self.stories_list.remove(card)
        self.release_available = not (len(self.release) == 0)
        if self.context.is_new_game:
            self.toggle_movement(False)

    def disable_restrictions(self):
        self.statistical_research_available = True
        self.available = True
        self.user_survey_available = True
        self.release_available = not (len(self.release) == 0)
        self.toggle_movement(True)

    def toggle_movement(self, is_enable: bool):
        for card in self.stories_list:
            card.is_movable = is_enable
        for card in self.release:
            card.is_movable = is_enable

    def get_total_stories_count(self) -> int:
        current_stories = len(self.context.current_stories.values())
        available_stories = len(self.context.available_stories.values())
        total_stories = current_stories + available_stories

        return total_stories
