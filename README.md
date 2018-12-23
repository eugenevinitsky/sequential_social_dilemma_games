# Sequential Social Dilemma Games
Repo for reproductions of deepmind gridworld papers

# Setup instructions
Run `python setup.py develop`

# Tests
Tests are located in the test folder and can be run individually or run by running `python -m pytest`

# Constructing new environments
The key parameter of MapEnv is reserved slots; env.reserved_slots stores tuples of length three or four 
consisting (row, col, 'char') where char is the character you wish to place 
Every environment that subclasses MapEnv needs to implement the following methods
`
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

  def clean_map(self):
      """Clean map of elements that should be removed at the start of every step"""
      pass

   def execute_custom_reservations(self):
    """Execute reserved slots that do not have to do with moving agents. For example, placing apples 
       or placing the fired beam. """
    raise NotImplementedError

  def setup_agents(self):
      """Construct all the agents for the environment"""
`
        
