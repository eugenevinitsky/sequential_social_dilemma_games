import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

tf = try_import_tf()


class AuxScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_aux_reward_weight = config["aux_reward_weight"]
        self.aux_reward_curriculum_steps = config["aux_reward_curriculum_steps"]
        self.aux_reward_curriculum_weights = config["aux_reward_curriculum_weights"]
        self.timestep = 0
        self.cur_aux_reward_weight = self.compute_weight()
        # This tensor is for logging the weight to progress.csv
        self.cur_aux_reward_weight_tensor = tf.get_variable(
            "cur_aux_reward_weight",
            initializer=self.cur_aux_reward_weight,
            trainable=False,
        )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(AuxScheduleMixIn, self).on_global_var_update(global_vars)
        self.timestep = global_vars["timestep"]
        self.cur_aux_reward_weight = self.compute_weight()
        self.cur_aux_reward_weight_tensor.load(
            self.cur_aux_reward_weight, session=self._sess
        )

    def compute_weight(self):
        """ Computes multiplier for aux reward based on training steps
        taken and curriculum parameters.
        """
        weight = np.interp(
            self.timestep,
            self.aux_reward_curriculum_steps,
            self.aux_reward_curriculum_weights,
        )
        return weight * self.baseline_aux_reward_weight
