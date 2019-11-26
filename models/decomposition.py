from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.a3c.a3c_tf_policy_graph import A3CPolicyGraph
from ray.rllib.evaluation.postprocessing      import discount
from ray.rllib.evaluation.sample_batch        import SampleBatch


import tensorflow as tf
import numpy as np
import scipy.signal


def get_last_value_prediction(policy_graph, sample_batch):
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(len(policy_graph.model.state_in)):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy_graph._value(sample_batch["new_obs"][-1],
                                     sample_batch["actions"][-1],
                                     sample_batch["rewards"][-1], *next_state)
    return last_r


def get_exponential_smoothed_rewards(rewards, gamma, exponential_decay, last_smoothed_reward):
    """
    Get the temporal smoothed rewards e_j with rewards r_j
    e_j(t) = gamma * exp_decay * e_j(t-1) + r_j(t)

    Parameters
    ----------
    rewards: np.array()
        rewards of the sample batch
    gamma: float
        discount factor
    exponential_decay(lambda): float
        exponential decay rate to temporally smooth the rewards
    last_smoothed_reward: float
        Temporally smoothed reward of the last sampled batch

    Returns
    -------
    e : np.array()
        temporally smoothed reward
    """

    decay_rate = gamma * exponential_decay
    x = np.concatenate(([last_smoothed_reward], rewards))

    return scipy.signal.lfilter([1], [1, -decay_rate], x, axis=0)[1:]


def decomposition(sample_batch, other_agent_batches, last_r, 
                  team_coeff = 0.1, equal_const_coeff = 0.0, explore_const_coeff = 0.0, 
                  gamma = 0.99, use_gae = True, lambda_ = 1.0):
    """
    Get the subjective reward u_i with decomposing team rewards and individual reward 

    u_i(t)   = r_ext(t) + team_coeff * r_int(t)
    r_ext(t) = e_i(t)   
    r_int(t) = E(t)   # E(t) = sum(e_j(t))
    
    adv(u_i(t)) = adv(r_ext(t)) + adv(r_int(t))

    Parameters
    ----------
    sample_batch: SampleBatch
        agent i's sample batch with timestep [t: t+sample_batch_size]
    other_agent_batches: dict
        dictionary mapped with {['agent-j']:(policy, SampleBatch)}
    last_r: float
        V(s(t+1))
    altruism_coeff: float
        altruistic motivation coefficient
    gamma: float
        discount factor
    exponential_decay(lambda): float
        exponential decay rate to temporally smooth the rewards
    """
    
    traj = {}
    trajsize = len(sample_batch['actions'])
    for key in sample_batch:
        traj[key] = np.stack(sample_batch[key])  

    # Number of agents
    N = len(other_agent_batches) + 1
    r_i = sample_batch['rewards']
    vpred_i = np.concatenate([sample_batch["vf_preds"], np.array([last_r])])
    
    # Get Value function and rewards of the other agents
    Norm = np.append(np.zeros_like(sample_batch['vf_preds']), 0)
    R    = np.zeros_like(sample_batch['rewards'], dtype = np.float32)
    for agent_j in other_agent_batches.keys():
        # Policy, SampleBatch = other_agent_batches[agent-id]
        agent_j_policy, agent_j_batch = other_agent_batches[agent_j]
        r_j = agent_j_batch['rewards']
        R  += r_j
        
        # Get the value function of the other agents
        last_r_j = get_last_value_prediction(agent_j_policy, agent_j_batch)
        vpred_j  = np.concatenate([agent_j_batch["vf_preds"], np.array([last_r_j])])
        Norm    += vpred_j   
    
    # Average
    R    /= (N-1)
    Norm /= (N-1)        
    
    #########################################
    ## Generalized Advantageous Esitmation ##
    #########################################
    
    # Compute value targets and the advantage
    if use_gae:
        # GAE for extrinsic motivation
        q_ext = traj["rewards"] + gamma * vpred_i[1:]
        delta_t_ext = q_ext - vpred_i[:-1]
        adv_t_ext   = discount(delta_t_ext, gamma * lambda_)
        traj['value_targets'] = (adv_t_ext + traj['vf_preds']).copy().astype(np.float32)

        
        # GAE for intrinsic motivation
        q_int = R + gamma * Norm[1:]
        delta_t_int = q_int - Norm[:-1]
        adv_t_int   = discount(delta_t_int, gamma * lambda_)
        
        # Hang up the Constraint
        equal_const   = np.abs(q_ext - q_int) - np.abs(vpred_i[:-1] - Norm[:-1])
        equal_const   = discount(equal_const, gamma * lambda_)
        explore_const = np.square(adv_t_int) 
        
        # Advantage = intrinsic advantage + extrinsic advantage
        traj['advantages'] = ((1-team_coeff) * adv_t_ext + team_coeff * adv_t_int \
                              - equal_const_coeff * equal_const - explore_const_coeff * explore_const)
        
    return SampleBatch(traj)


class A3CDecompositionPolicyGraph(A3CPolicyGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Track the temporally extended rewards
        self.last_smoothed_rewards = {}
        for agent in self.config['multiagent']['policy_graphs'].keys():
            self.last_smoothed_rewards[agent] = 0
        # Track the temporally extended Norm
        self.last_smoothed_norm = 0
    
    
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        
        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(len(self.model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            last_r   = self._value(sample_batch["new_obs"][-1],
                                   sample_batch["actions"][-1],
                                   sample_batch["rewards"][-1], *next_state)
        
        traj = decomposition(sample_batch, other_agent_batches, last_r, 
                             team_coeff          = self.config['team_coeff'],
                             equal_const_coeff   = self.config['equal_const_coeff'],
                             explore_const_coeff = self.config['explore_const_coeff'],
                             gamma      = self.config['gamma'],
                             lambda_    = self.config['lambda'])
        
        return traj
