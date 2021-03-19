import numpy as np

from social_dilemmas.envs.env_creator import get_env_creator

cleanup_env = get_env_creator("cleanup", 5, {})(0)

agent_ids = ["agent-" + str(agent_number) for agent_number in range(5)]
actions = {}


def profile_cleanup():
    """
    Profiles environments steps by executing random actions in a cleanup environment with 5 agents.
    """
    for i in range(1000):
        for agent_id in agent_ids:
            actions[agent_id] = np.random.randint(8)
            cleanup_env.step(actions)


profile_cleanup()
