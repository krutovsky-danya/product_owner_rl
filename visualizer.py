import matplotlib.pyplot as plt

from pipeline import MetricsStudy, LoggingStudy

def show_rewards(study: MetricsStudy, show_estimates=False, filename=None):
    plt.plot(study.rewards_log, '.', label='Rewards')
    if show_estimates:
        plt.plot(study.q_value_log, '.', label='Estimates')
    plt.xlabel("Trajectory")
    plt.ylabel("Reward")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def show_sprints(study: LoggingStudy, filename=None):
    plt.plot(study.sprints_log, ".")
    plt.title("Sprints count")
    plt.xlabel("Trajectory")
    plt.ylabel("Sprint")
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def show_loss(study: LoggingStudy, filename=None):
    plt.plot(study.loss_log, ".")
    plt.title("Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    if filename is not None:
        plt.savefig(filename)
    plt.show()
