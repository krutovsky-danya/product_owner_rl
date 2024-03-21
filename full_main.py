from tutorial_main import make_tutorial_study
from credit_start_main import make_credit_start_study


# fix hyper params
# repeat for original and modified versions
n = 5
trajectory_max_len = 100
episode_n = 100

# repeat several times
for i in range(n):
    # study tutorial agent
    tutorial_study = make_tutorial_study(trajectory_max_len, episode_n)
        
    # use previous agent to study credit start agent
    tutorial_agent = tutorial_study.agent
    credit_start_study = make_credit_start_study(tutorial_agent, trajectory_max_len, episode_n)

    # use previous agents to study credit end agent
    # use previous agents to study end agent

    # eval model
    # collect quality metrics

# analize collected metrics
# show results

# choose best variant