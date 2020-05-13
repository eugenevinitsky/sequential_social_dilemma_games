from gym.spaces import Discrete


class DiscreteWithDType(Discrete):
    def __init__(self, n, dtype):
        assert n >= 0
        self.n = n
        # Skip Discrete __init__ on purpose, to avoid setting the wrong dtype
        super(Discrete, self).__init__((), dtype)
