[![Build Status](https://travis-ci.com/eugenevinitsky/sequential_social_dilemma_games.svg?branch=master)](https://travis-ci.com/eugenevinitsky/sequential_social_dilemma_games)

# Sequential Social Dilemma Games
Repo for reproductions of deepmind gridworld papers

# Setup instructions
Run `python setup.py develop`
Then, activate your environment by running `source activate causal`.

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
        
