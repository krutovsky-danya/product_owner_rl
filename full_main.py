from final_sprints_main import make_final_sprints_study
from pipeline.study_agent import load_dqn_agent
from tutorial_main import make_tutorial_study
from credit_start_main import make_credit_study
from pipeline import TUTORIAL, CREDIT_FULL, CREDIT_END, CREDIT_START, END

makers = {
    TUTORIAL: make_tutorial_study,
    CREDIT_FULL: make_credit_study,
    CREDIT_END: make_credit_study,
    CREDIT_START: make_credit_study,
    END: make_final_sprints_study
}

# fix hyper params
# repeat for original and modified versions
n = 1
trajectory_max_len = -1
episode_n = 2000
with_info = True
full_order = [CREDIT_FULL]
agents = {}
order = []

# repeat several times
for i in range(n):
    # study agents in order
    for stage in full_order:
        makers[stage](agents, order, trajectory_max_len, episode_n, stage, with_info, save_rate=100)

    # eval model
    # collect quality metrics

# analyze collected metrics
# show results

# choose best variant
