from itertools import count
import queue
import time
import logging
from functools import reduce
from collections import deque
from argparse import Namespace
import torch
import torch.multiprocessing as mp
import threading as thr

if torch.cuda.is_available():
    mp.set_start_method("spawn", force=True)

import numpy as np
from inferno.io.transform import Compose
from inferno.io.transform.generic import Normalize
from inferno.io.transform.image import ElasticTransform, RandomFlip, RandomRotate

import tiktorch.utils as utils
import tiktorch.fast_augment as aug
import tensorboardX as tX

logger = logging.getLogger("Trainy")


# Globals
STATE_QUEUE_GET_TIMEOUT = 5


class Trainer(object):
    # Setting this to true might help training, but can amount to a lot of compute.
    USE_CACHE_KEEPING = False
    # Cache size to use. Large cache size ==> more CPU RAM.
    CACHE_SIZE = 200
    # FIXME This is a hack to invert the labels. Make sure the labels are binary to begin with, or else...
    INVERT_BINARY_LABELS = True

    def __init__(self, handler, hyperparameters=None, log_directory=None):
        # Privates
        self._handler = handler
        # Preprocessing
        self._augmentor = None
        # FIXME Deprecate:
        self._raw_preprocessor = None
        self._joint_preprocessor = None
        # Training
        self._data_queue: mp.Queue = None
        self._state_queue: mp.Queue = None
        self._hparams_queue: mp.Queue = None
        self._change_hparams_event: mp.Event = None
        self._abort_event: mp.Event = None
        self._pause_event: mp.Event = None
        self._state_request_event = None
        self._training_process: mp.Process = None
        self._ignited = False
        # Publics
        # Sane default hparams
        if hyperparameters is None:
            self.hparams = Namespace(
                optimizer_kwargs=dict(lr=0.0003, weight_decay=0.0001, amsgrad=True),
                optimizer_name="Adam",
                criterion_kwargs=dict(reduce=False),
                criterion_name="BCEWithLogitsLoss",
                batch_size=1,
                cache_size=self.CACHE_SIZE,
                augmentor_kwargs={"invert_binary_labels": self.INVERT_BINARY_LABELS},
            )
        else:
            self.hparams: Namespace = hyperparameters
        self.log_directory = log_directory

    @property
    def model(self):
        return self._handler.model

    @property
    def augmentor(self):
        if self._augmentor is None:
            self._augmentor = aug.AugmentationSuite(**self.hparams.augmentor_kwargs)
        return self._augmentor

    def share_memory(self):
        self._handler._model = self._handler._model.share_memory()
        return self

    @property
    def device(self):
        return self._handler.device

    @staticmethod
    def _train_process(
        model_state: dict,
        model_config: tuple,
        device: torch.device,
        data_queue: mp.Queue,
        augmentor: aug.AugmentationSuite,
        state_queue: mp.Queue,
        abort: mp.Event,
        pause: mp.Event,
        change_hparams: mp.Event,
        state_request: mp.Event,
        use_cache_keeping: bool,
        hparams_queue: mp.Queue,
        log_directory: str,
    ):
        logger = logging.getLogger("Trainer._train_process")
        # Build the model
        model = utils.define_patched_model(*model_config)
        # Load state dict
        model.load_state_dict(model_state)
        model = model.to(device)
        # Build tensorboard logger
        if log_directory is not None:
            tensorboard = tX.SummaryWriter(log_dir=log_directory)
            logger.info("Writing tensorboard logs to %s", log_directory)
        else:
            tensorboard = None
            logger.warning("Not writing tensorboard logs.")

        _state_lock = thr.Lock()
        _stop_server = thr.Event()

        # This guy listens if state request has been set; if it is, it serves the state_dict in
        # the queue.
        def _state_server():
            while True:
                if _stop_server.is_set():
                    logger.info("Stopping state server...")
                    break
                if state_request.is_set():
                    try:
                        logger.info("Obtained request for new state. Waiting for lock...")
                        with _state_lock:
                            state_queue.put_nowait(model.state_dict())
                            logger.info("Put most recent state in queue.")
                    except queue.Full:
                        logger.info("State queue is full.")
                        pass
                    state_request.clear()
                    logger.info("State request cleared.")
                time.sleep(0.01)

        # Set up server to listen for state requests
        logger.info("Spooling state server thread...")
        server_thread = thr.Thread(target=_state_server, args=())
        server_thread.start()

        def _kill_state_server(num_kill_attempts=5):
            logger.info("Killing state server thread...")
            _stop_server.set()
            kill_attempts = 0
            while server_thread.is_alive():
                time.sleep(1)
                kill_attempts += 1
                logger.info("Attempt %d of %d.", (kill_attempts, num_kill_attempts))
                if kill_attempts > num_kill_attempts:
                    break
            if server_thread.is_alive():
                logger.warning("Failed to kill state server after %d attempts.", num_kill_attempts)
            else:
                logger.info("State server killed.")

        logger.info("Initializing Loss and Optimizer.")
        # Set up what's needed for training
        hparams = hparams_queue.get()
        criterion = getattr(torch.nn, hparams.criterion_name)(**hparams.criterion_kwargs)
        optim = getattr(torch.optim, hparams.optimizer_name)(model.parameters(), **hparams.optimizer_kwargs)
        # Init a cache. In case there are not enough batches in data_queue,
        # we'll use it to top up the batch with what's in this cache.
        data_cache = deque(maxlen=hparams.cache_size)

        # If this returns true, the sample will be added to batch later downstream.
        def _cache_keeping(sample):
            global data_cache
            dirty_indices = []
            update_batch = True
            for idx, cache_sample in enumerate(data_cache):
                cache_data, cache_labels = cache_sample
                data, labels = sample
                # Compare data
                data_diff = data.sub(cache_data).abs_().sum().item()
                if data_diff > 1e-5:
                    # Not a match
                    continue
                else:
                    # A match - data exists in cache. But still, the labels could have
                    # been updated...
                    labels_diff = labels.sub(cache_labels).abs_().sum().item()
                    if labels_diff < 1e-5:
                        # Labels match - the sample exists 1-to-1 in the cache, so there's no
                        # need to update the batch.
                        update_batch = False
                    else:
                        # The data matches, but the labels don't match - the cache is dirty
                        # and needs updating.
                        dirty_indices.append(idx)
            # Clean out the dirties yeet
            if dirty_indices:
                # Make a new, clean cache
                data_cache = deque(
                    [_sample for _idx, _sample in enumerate(data_cache) if _idx not in dirty_indices],
                    maxlen=hparams.cache_size,
                )
            # So if we're still updating the batch, we should also update the cache
            data_cache.append(sample)
            # Done-o
            return update_batch

        # Global Training Iteration Counter
        iter_count = 0
        while True:
            if change_hparams.is_set():
                change_hparams.clear()
                try:
                    hparams = hparams_queue.get_nowait()
                except queue.Empty:
                    logger.info("Hyperparameter queue is empty.")
                    pass
                logger.info("Changing hyperparameters: initializing loss and optimizer.")
                criterion = getattr(torch.nn, hparams.criterion_name)(**hparams.criterion_kwargs)
                optim = getattr(torch.optim, hparams.optimizer_name)(model.parameters(), **hparams.optimizer_kwargs)

            # Check if a new state is requested
            if state_request.is_set():
                # First things first,
                state_request.clear()
                try:
                    state_queue.put_nowait(model.state_dict())
                except queue.Full:
                    # Welp, no new parameters
                    pass
            # Init a batch
            batch = []
            # Check if abort event is set
            if abort.is_set():
                logger.info("Aborting...")
                _kill_state_server()
                break
            if pause.is_set():
                logger.info("Waiting for resume...")
                time.sleep(1)
                continue
            try:
                try:
                    logger.info("Currently %d elements in data_queue.", data_queue.qsize())
                except NotImplementedError:
                    # This raises a Not Implemented Error on OSX
                    pass
                sample = 0
                while len(batch) < hparams.batch_size:
                    logger.info("Trying to Fetch sample %d of %d...", (sample, hparams.batch_size))
                    # Try to fetch from data queue
                    data, labels = data_queue.get(block=False)
                    try:
                        _q_size_now = data_queue.qsize()
                    except NotImplementedError:
                        _q_size_now = None
                    logger.info(
                        "Fetched sample %d of %d. Remaining items in queue: ", (sample, hparams.batch_size, _q_size_now)
                    )
                    if use_cache_keeping:
                        if _cache_keeping((data, labels)):
                            batch.append((data, labels))
                            sample += 1
                    else:
                        # Add to batch
                        batch.append((data, labels))
                        # Add to cache
                        data_cache.append((data, labels))
                        sample += 1
            except queue.Empty:
                logger.info("Queue Exhausted.")
                if len(batch) == 0 and len(data_cache) == 0:
                    # Both batch and cache empty, try again
                    logger.info("Trying to fetch again...")
                    time.sleep(0.1)
                    continue
                elif len(batch) == hparams.batch_size:
                    # Nothing to do here
                    pass
                elif len(batch) < hparams.batch_size:
                    # Batch not full, try to top it up from the cache
                    logger.info("Topping up batch, currently with %d elements...", len(batch))
                    while len(data_cache) > 0 and len(batch) < hparams.batch_size:
                        data_sample = data_cache.popleft()
                        batch.append(data_sample)
                        data_cache.append(data_sample)
                else:
                    logger.error("LOLWTF: len(batch) = %d, " f"len(data_cache) = %d", (len(batch), len(data_cache)))
                    # Stop state server before throwing up error
                    _kill_state_server()
                    raise RuntimeError

            logger.info("Updating with %d samples...", len(batch))
            # Make a batch
            logger.info("Augmenting...")
            try:
                augmented_batch = [augmentor(*sample) for sample in batch]
                data, labels, weights = zip(*augmented_batch)
                logger.debug(
                    "data.shapes = %s, label.shapes = %s, weights.shapes = %s",
                    [[list(a.shape) for a in d] for d in (data, labels, weights)],
                )
                data, labels, weights = (
                    torch.stack(data, dim=0),
                    torch.stack(labels, dim=0),
                    torch.stack(weights, dim=0),
                )
                # Ship tensors to device
                data, labels, weights = (data.to(device), labels.to(device), weights.to(device))
                logger.info("Transferred to device.")
                # Train the model
                prediction = model(data)
                logger.info("Fed forward.")
                loss = criterion(prediction, labels).mul(weights).mean()
                logger.info("Loss Evaluated. Waiting for state lock...")
                with _state_lock:
                    optim.zero_grad()
                    loss.backward()
                    logger.info("Backproped.")
                    optim.step()
                    logger.info("Stepped.")
                    iter_count += 1
                # Logging
                if tensorboard is not None:
                    tensorboard.add_scalar("loss", loss.item(), global_step=(iter_count - 1))
                    logger.info("Logged iteration %d", iter_count)
            except Exception:
                _kill_state_server()
                raise

    def ignition(self):
        # Done in this method:
        #   1. Init data queue
        #   2. Init abort event
        #   3. Start the training process
        logger = logging.getLogger("Trainer.ignition")
        logger.info("Prepping Queue and Event...")
        self._data_queue = mp.Queue()
        self._state_queue = mp.Queue()
        self._hparams_queue = mp.Queue()
        self._hparams_queue.put(self.hparams)
        self._abort_event = mp.Event()
        self._pause_event = mp.Event()
        self._state_request_event = mp.Event()
        self._change_hparams_event = mp.Event()
        logger.info("Sharing Memory...")
        # self.share_memory()
        model_state = self.model.state_dict()
        model_config = (self.model._model_file_name, self.model._model_class_name, self.model._model_init_kwargs)
        self._training_process = mp.Process(
            target=self._train_process,
            args=(
                model_state,
                model_config,
                self.device,
                self._data_queue,
                self.augmentor,
                self._state_queue,
                self._abort_event,
                self._pause_event,
                self._change_hparams_event,
                self._state_request_event,
                self.USE_CACHE_KEEPING,
                self._hparams_queue,
                self.log_directory,
            ),
        )
        logger.info("3, 2, 1...")
        self._training_process.start()  # todo: fix bug: sometimes the training process does not start
        logger.info("We have lift off.")
        self._ignited = True

    def _drain_state_queue(self):
        logger = logging.getLogger("Trainer._drain_state_queue")
        state = None
        while True:
            try:
                state = self._state_queue.get_nowait()
                logger.info("Found residual state in state_queue.")
            except queue.Empty:
                break
        return state

    def update_handler_model_state(self):
        logger = logging.getLogger("Trainer.update_handler_model_state")
        assert self._ignited, "Training process not ignited."
        logger.info("Requesting new state.")
        # Flush queue for residual states (e.g. from previously timed-out queue-get's)
        state = self._drain_state_queue()
        # Send request for parameters
        self._state_request_event.set()
        # Try to get parameters from queue
        try:
            logger.info("Waiting for new state...")
            state = self._state_queue.get(timeout=STATE_QUEUE_GET_TIMEOUT)
            logger.info("Acquired new state.")
        except queue.Empty:
            logger.info("Failed to acquire new state...")
            pass
        if state is not None:
            self.model.load_state_dict(state)
            logger.info("Loaded state.")

    def shut_down_training_process(self):
        if self._training_process is not None:
            # Shut down the training process
            logger.info("Setting Abort Event...")
            self._abort_event.set()
            for trial in range(6):
                logger.info(f"Try {trial} of 5:")
                # Give training process some time to die
                if self._training_process.is_alive():
                    logger.info(f"Process Alive.")
                    time.sleep(10)
                else:
                    break
            logger.info(f"Process Dead.")

    def __del__(self):
        # Shut down the training process
        self.shut_down_training_process()
        self._ignited = False

    # TODO Deprecate
    def _preprocess(self, data, labels):
        # labels.shape = data.shape = (c, z, y, x)
        # FIXME Not have these hard coded
        if self._raw_preprocessor is None:
            self._raw_preprocessor = Normalize()
        if self._joint_preprocessor is None:
            self._joint_preprocessor = Compose(RandomFlip(), RandomRotate(), ElasticTransform(alpha=2000.0, sigma=50.0))
        # Convert data and labels to torch tensors
        with torch.no_grad():
            # Apply transforms
            data = self._raw_preprocessor(data)
            data, labels = self._joint_preprocessor(data, labels)
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)
            # Obtain weight map
            weights = labels.gt(0)
            # Label value 0 actually corresponds to Ignore. Subtract 1 from all pixels that will be
            # weighted to account for that
            labels[weights] -= 1
        # Done
        return data, labels, weights.float()

    def ensure_ignited(self):
        if not self._ignited:
            logger.info("Ignition...")
            self.ignition()

    @property
    def is_ignited(self):
        return self._ignited

    def push(self, data, labels):
        logger = logging.getLogger("Trainer.push")
        # Done in this method:
        #   1. Augment data
        #   2. Push to queue
        self.ensure_ignited()
        logger.info(f"Feeding {len(data)} samples to queue...")
        # Augment
        for _data, _labels in zip(data, labels):
            self._data_queue.put((_data, _labels))
        logger.info(f"Fed {len(data)} samples to queue...")

    def push_hparams(self, hparams: dict):
        logger = logging.getLogger("Trainer.push_hparams")
        # Done in this method:
        # If training process is running, push hparams into queue, else set as default

        def _drain_hparams_queue():
            deprecated_hparams = None
            while True:
                try:
                    deprecated_hparams = self._hparams_queue.get_nowait()
                    logger.info("Found deprecated hyperparameters in hparams_queue")
                except queue.Empty:
                    break
            return deprecated_hparams

        hparams = Namespace(**hparams)
        if not self.is_ignited:
            logger.info("Setting new default hyperparameters")
            self.hparams = hparams
        else:
            self.hparams = hparams
            logger.info("Feeding parameters to hyperparameter queue")
            with thr.Lock():
                _drain_hparams_queue()
                self._hparams_queue.put(hparams)
                time.sleep(1)
            self._change_hparams_event.set()

    def pause(self):
        if self._ignited:
            logger.info("Pausing training...")
            self._pause_event.set()
        else:
            logger.warning("Not ignited, nothing to pause.")

    def resume(self):
        if self._ignited:
            logger.info("Resuming training...")
            self._pause_event.clear()
        else:
            logger.warning("Not ignited, nothing to resume.")

    def is_alive(self):
        if self._training_process is None:
            return False
        else:
            return self._training_process.is_alive()
