import copy

import ray
import torch

from ray.util.queue import Queue


class QueueStorage(object):
    def __init__(self, threshold=15, size=20):
        """Queue storage
        Parameters
        ----------
        threshold: int
            if the current size if larger than threshold, the data won't be collected
        size: int
            the size of the queue
        """
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()

@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, path=None):
        if not path:
            path = self.config.results_path / "model.checkpoint"

        torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
