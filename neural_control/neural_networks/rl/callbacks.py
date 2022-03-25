from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps


class FunctionCallback(BaseCallback):
    def __init__(self, callback_fn):
        super().__init__()
        self.callback_fn = callback_fn

    def _on_step(self):
        self.callback_fn(self.n_calls)
        return True


class EveryNTimestepsFunctionCallback(EveryNTimesteps):
    def __init__(self, n_steps, callback_fn):
        super().__init__(n_steps, FunctionCallback(callback_fn))


class EveryNTimestepsPlusStartFinishFunctionCallback(EveryNTimestepsFunctionCallback):
    def __init__(self, n_steps, callback_fn):
        super().__init__(n_steps, callback_fn)

    def _on_training_start(self):
        self._on_event()

    def _on_training_end(self):
        self._on_event()
