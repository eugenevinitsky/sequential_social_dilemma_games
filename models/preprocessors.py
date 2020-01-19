import numpy as np
from ray.rllib.models.preprocessors import (
    DictFlatteningPreprocessor,
    Preprocessor,
    TupleFlatteningPreprocessor,
    gym,
    override,
)


# This preprocessor is for the uint8 observation dtype,
# It zero-mean centers the observation upon transformation.
class PreprocessorUint8(DictFlatteningPreprocessor):
    @property
    def observation_space(self):
        obs_space = gym.spaces.Box(-1.0, 1.0, self.shape, dtype=np.uint8)
        if isinstance(self, TupleFlatteningPreprocessor) or isinstance(
            self, DictFlatteningPreprocessor
        ):
            obs_space.original_space = self._obs_space
        return obs_space

    @override(Preprocessor)
    def transform(self, observation):
        self.check_shape(observation)
        array = np.zeros(self.shape, dtype=np.float32)
        self.write(observation, array, 0)
        np.divide(np.subtract(array, 128.0, out=array), 128.0, out=array)
        return array
