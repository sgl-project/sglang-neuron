# SPDX-License-Identifier: Apache-2.0
"""Stage and step timers for the Neuron diffusion pipelines.

torch-neuronx queues the device work and returns, so a host clock at a stage boundary measures
submission; the output rank marks the boundaries with device timing events instead
(``StageMarks``).
"""

import time

import torch


class StageMarks:
    """Time marks at stage (or step) boundaries; ``seconds(a, b)`` is the time between two.

    On the output rank (``device_events=True``) a mark is a timing event recorded on the device
    stream: it completes, with a timestamp, once the device has finished the work queued before
    it, and recording it does not hold the host. Read ``seconds`` only after the output rank's
    image copy to the CPU, which waits for that work. On the other ranks a mark is a host clock
    reading (``time.perf_counter``).
    """

    def __init__(self, *, device_events: bool):
        self._events = device_events

    def mark(self):
        if not self._events:
            return time.perf_counter()
        event = torch.neuron.Event(enable_timing=True)
        event.record()
        return event

    def seconds(self, start, end) -> float:
        if not self._events:
            return end - start
        end.synchronize()  # already complete after the image copy; a guard, not a wait
        return start.elapsed_time(end) / 1000.0
