import signal
from importlib import util as imputils
from contextlib import contextmanager
import torch
from torch.autograd import Variable


class delayed_keyboard_interrupt(object):
    """
    Delays SIGINT over critical code.
    Borrowed from:
    https://stackoverflow.com/questions/842557/
    how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py
    """
    # PEP8: Context manager class in lowercase
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def assert_(condition, message='', exception_type=Exception):
    if not condition:
        raise exception_type(message)


class WannabeConvNet3D(torch.nn.Module):
    """A torch model that pretends to be a 2D convolutional network.
    This exists to just test the pickling machinery."""
    def forward(self, input_):
        assert isinstance(input_, Variable)
        # Expecting 5 dimensional inputs as (NCDHW).
        assert input_.dim() == 5
        return input_


class TinyConvNet3D(torch.nn.Module):
    """Tiny ConvNet with actual parameters."""
    def __init__(self, num_input_channels=1, num_output_channels=1):
        super(TinyConvNet3D, self).__init__()
        self.conv3d = torch.nn.Conv3d(num_input_channels, num_output_channels, 3, padding=1)

    def forward(self, *input):
        return self.conv3d(input[0])


class TinyConvNet2D(torch.nn.Module):
    """Tiny ConvNet with actual parameters."""
    def __init__(self, num_input_channels=1, num_output_channels=1):
        super(TinyConvNet2D, self).__init__()
        self.conv2d = torch.nn.Conv2d(num_input_channels, num_output_channels, 3, padding=1)

    def forward(self, *input):
        return self.conv2d(input[0])


def define_patched_model(model_file_name, model_class_name, model_init_kwargs):
    # Dynamically import file.
    module_spec = imputils.spec_from_file_location('model', model_file_name)
    module = imputils.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # Build model from file
    model: torch.nn.Module = \
        getattr(module, model_class_name)(**model_init_kwargs)
    # Monkey patch
    model._model_file_name = model_file_name
    model._model_class_name = model_class_name
    model._model_init_kwargs = model_init_kwargs
    return model

