from final_sprints_main import make_final_sprints_study
from tutorial_main import make_tutorial_study
from credit_start_main import make_credit_study


# fix hyper params
# repeat for original and modified versions
n = 1
trajectory_max_len = 100
episode_n = 100

# repeat several times
for i in range(n):
    # study tutorial agent
    tutorial_study = make_tutorial_study(trajectory_max_len, episode_n)
        
    # use previous agent to study credit start agent
    tutorial_agent = tutorial_study.agent
    agents = [tutorial_agent]
    credit_start_study = make_credit_study(agents, trajectory_max_len, episode_n, False)

    # use previous agents to study credit end agent
    agents.append(credit_start_study.agent)
    credit_end_study = make_credit_study(agents, trajectory_max_len, episode_n, True)
    
    # use previous agents to study end agent
    agents.append(credit_end_study.agent)
    final_sprints_study = make_final_sprints_study(agents, trajectory_max_len, episode_n)

    # eval model
    # collect quality metrics

# analize collected metrics
# show results

# choose best variant