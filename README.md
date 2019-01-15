[![Build Status](https://travis-ci.com/eugenevinitsky/sequential_social_dilemma_games.svg?branch=master)](https://travis-ci.com/eugenevinitsky/sequential_social_dilemma_games)

# Sequential Social Dilemma Games
This repo is an open-source implementation of DeepMind's Sequential Social Dilemma (SSD) multi-agent game-theoretic environments [[1]](https://arxiv.org/abs/1702.03037). SSDs can be thought of as analogous to spatially and temporally extended Prisoner's Dilemma-like games. The reward structure poses a dilemma because individual short-term optimal strategies lead to poor long-term outcomes for the group.

## Implemented Games

* **Cleanup**: A public goods dilemma in which agents get a reward for consuming apples, but must use a cleaning beam to clean a river in order for apples to grow. While an agent is cleaning the river, other agents can exploit it by consuming the apples that appear.

* **Harvest**: A tragedy-of-the-commons dilemma in which apples regrow at a rate that depends on the amount of nearby apples. If individual agents employ an exploitative strategy by greedily consuming too many apples, the collective reward of all agents is reduced.

<img src="images/schelling.png" alt="Schelling diagrams for Harvest and Cleanup" width="953" height="352"/>

The above plot shows the empirical Schelling diagrams for both Cleanup (A) and Harvest (B) (from [[2]](https://arxiv.org/abs/1803.08884)). These diagrams show the payoff that an individual agent can expect if it follows a defecting/exploitative strategy (red) vs a cooperative strategy (blue), given the number of other agents that are cooperating.  We can see that an individual agent can almost always greedily benefit from detecting, but the more agents that defect, the worse the outcomes for all agents. 

## Relevant papers

1. Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).

2.  Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Dueñez-Guzman, E., Castañeda, A. G., Dunning, I., Zhu, T., McKee, K., Koster, R., Tina Zhu, Roff, H., Graepel, T. (2018). [Inequity aversion improves cooperation in intertemporal social dilemmas](https://arxiv.org/abs/1803.08884). In Advances in Neural Information Processing Systems (pp. 3330-3340).

3. Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P. A., Strouse, D. J., Leibo, J. Z. & de Freitas, N. (2018). [Intrinsic Social Motivation via Causal Influence in Multi-Agent RL](https://arxiv.org/abs/1810.08647). arXiv preprint arXiv:1810.08647. 


# Setup instructions
Run `python setup.py develop`
Then, activate your environment by running `source activate causal`.

To then set up the branch of Ray on which we have build the causal influence code, clone the repo to your desired folder:
`git clone https://github.com/natashamjaques/ray.git`.

Next, go to the rllib folder:
` cd ray/python/ray/rllib ` and run the script `python setup-rllib-dev.py`. This will copy the rllib folder into the pip install of Ray and allow you to use the version of RLlib that is in your local folder by creating a softlink. 

# Tests
Tests are located in the test folder and can be run individually or run by running `python -m pytest`

# Constructing new environments
The key parameter of MapEnv is reserved slots; env.reserved_slots stores tuples of length three or four 
consisting (row, col, 'char') where char is the character you wish to place 
Every environment that subclasses MapEnv needs to implement the following methods

```
  def custom_reset(self):
      """Reset custom elements of the map. For example, spawn apples and build walls"""  
      pass

  def custom_action(self, agent):
      """Add reservations to self.reserved_slots for actions that are not move or turn. For example,  
      if an agent can fire, you can add (row, col, 'F') to indicate that F should be placed at that point"""
      pass

  def custom_map_update(self):
      """Custom map updates that don't have to do with agent actions. For example, you can add
      (row, col, 'A') to env.reserved_slots to indicate an apple should be placed at that point"""
      pass

  def append_hiddens(self, new_pos, old_char, new_char):
      """Add cells that will be hidden and should be put back later

      Parameters
      ----------
      new_pos: list
          the position the new char is going to be placed at
      old_char: str
          the character that will be hidden
      new_char: str
          the character that will replace it
      """
      raise NotImplementedError

   def execute_custom_reservations(self):
    """Execute reserved slots that do not have to do with moving agents. For example, placing apples 
       or placing the fired beam. """
    raise NotImplementedError

  def setup_agents(self):
      """Construct all the agents for the environment"""
```
        
