from environment import ProductOwnerEnv
from main import create_usual_agent
from pipeline import AggregatorStudy
from pipeline.study_agent import load_dqn_agent, save_dqn_agent
import visualizer


def make_final_sprints_study(prev_agents, trajectory_max_len, episode_n):
    env = ProductOwnerEnv()
    agent = create_usual_agent(env, trajectory_max_len, episode_n)
    agents = prev_agents + [agent]

    study = AggregatorStudy(env, agents, trajectory_max_len)
    study.study_agent(episode_n)

    return study

def main():
    tutorial_model_path = 'models/tutorial_model.pt'
    tutorial_agent = load_dqn_agent(tutorial_model_path)

    credit_start_path = 'models/credit_start_model.pt'
    credit_start_agent = load_dqn_agent(credit_start_path)

    credit_end_path = 'models/credit_end_model.pt'
    credit_end_agent = load_dqn_agent(credit_end_path)

    agents = [tutorial_agent, credit_start_agent, credit_end_agent]

    study = make_final_sprints_study(agents, 1000, 1000)
    agent = study.agent

    visualizer.show_rewards(study, show_estimates=True, filename='figures/rewards.png')
    visualizer.show_sprints(study, filename='figures/sprints.png')
    visualizer.show_loss(study, filename='figures/loss.png')

    agent.memory = []
    save_dqn_agent(agent, 'models/final_sprints_model.pt')

if __name__ == '__main__':
    main()
