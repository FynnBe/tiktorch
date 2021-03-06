from __future__ import annotations

import typing
import threading

from torch.utils.data import DataLoader

from tiktorch.server.datasets import DynamicDataLoaderWrapper

from . import types

if typing.TYPE_CHECKING:
    from .base import TrainingWorker
    from tiktorch.server.trainer import TikTrainer
    from tiktorch.server.datasets import DynamicDataset


__all__ = [
    "ICommand",
    "AwaitableCommand",
    "PauseCmd",
    "ResumeCmd",
    "StopCmd",
    "SetDevicesCmd",
    "UpdateDatasetCmd",
    "SetMaxNumberOfIterations",
]


class Context:
    """
    Command execution context
    Contains modifiable entities as attributes
    """

    def __init__(self, *, worker: TrainingWorker, trainer: TikTrainer) -> None:
        self.worker = worker
        self.trainer = trainer


class ICommand:
    __awaitable = None

    @property
    def awaitable(self):
        if not self.__awaitable:
            self.__awaitable = AwaitableCommand(self)

        return self.__awaitable

    def execute(self, ctx: Context) -> None:
        raise NotImplementedError()


class AwaitableCommand(ICommand):
    def __init__(self, cmd: ICommand):
        self._cmd = cmd
        self._done_evt = threading.Event()

    def wait(self):
        self._done_evt.wait()

    def execute(self, ctx: Context) -> None:
        try:
            self._cmd.execute(ctx)
        finally:
            self._done_evt.set()

    def __repr__(self):
        return f"Awaitable {self._cmd!r}"


class PauseCmd(ICommand):
    def execute(self, ctx: Context) -> None:
        ctx.worker.transition_to(types.State.Paused)


class ResumeCmd(ICommand):
    def execute(self, ctx: Context) -> None:
        ctx.worker.transition_to(types.State.Running)


class StopCmd(ICommand):
    def execute(self, ctx: Context) -> None:
        ctx.worker.transition_to(types.State.Stopped)


class SetDevicesCmd(ICommand):
    def __init__(self, devices):
        self._devices = devices

        self.result = None

    def execute(self, ctx: Context) -> None:
        self.result = ctx.worker.set_devices(self._devices)


class UpdateDatasetCmd(ICommand):
    def __init__(self, name, *, raw_data, labels):
        self._name = name
        self._raw_data = raw_data
        self._labels = labels

    def execute(self, ctx: Context) -> None:
        dataset = ctx.trainer.get_dataset(self._name)
        dataset.update(self._raw_data, self._labels)


class SetMaxNumberOfIterations(ICommand):
    def __init__(self, num_iterations: int) -> None:
        self._num_iterations = num_iterations

    def execute(self, ctx: Context) -> None:
        ctx.worker.set_max_num_iterations(self._num_iterations)
