from dataclasses import dataclass
from typing import Any, Dict, Sequence, Union

import ignite
import torch

from pybio.spec import node
from pybio.spec.utils import get_instance


@dataclass
class Config:
    # max_iterations: int
    # batch_size: int
    # num_iterations_per_update: int
    warmstart: bool



class IterationOutput:
    prediction: Any

class InferenceOutput(IterationOutput):
    pass


# class ValidationOutput(IterationOutput):
#     pass
#
# class TrainingOutput(IterationOutput):
#     pass


class Exemplum:

    def __init__(self, pybio_model: node.Model, config: Union[Dict[str, Any], Config], devices: Sequence[torch.device]):
        spec = pybio_model.spec
        self.model = get_instance(pybio_model)
        if spec.framework == "pytorch":
            assert isinstance(self.model, torch.nn.Module)
            if config.warmstart:
                state = torch.load(get_file_path(spec.prediction.weights), map_location=devices[0])
                self.model.load_state_dict(state)
        else:
            raise NotImplementedError

        inference_engine = ignite.engine.Engine(self._inference_step_function)
        # .add_event_handler(Events.STARTED, self.prepare_engine)
        # .add_event_handler(Events.COMPLETED, self.log_compute_time)

    def _inference_step_function(self, engine: ignite.engine.Engine, batch) -> InferenceOutput:
        prediction = self.model(batch)
        return InferenceOutput(prediction=prediction)


    # def _validation_step_function(self) -> ValidationOutput:
    #     return ValidationOutput()
    #
    #
    # def _training_step_function(self) -> TrainingOutput:
    #     return TrainingOutput()


    def train(self):
        pass


"""
required_kwargs: [model]
option_kwargs:
    max_iterations: 100
    batch_size: 1
    num_iterations_per_update: 5

        TRAINING_SHAPE: None,
        TRAINING_SHAPE_LOWER_BOUND: None,
        TRAINING_SHAPE_UPPER_BOUND: None,
        NUM_ITERATIONS_DONE: None,
        NUM_ITERATIONS_MAX: None,
        NUM_ITERATIONS_PER_UPDATE: None,
        LOSS_CRITERION_CONFIG: None,
        OPTIMIZER_CONFIG: None,
        TRANSFORMS: None,
