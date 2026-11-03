from threading import RLock
from functools import wraps

import matplotlib

matplotlib.use("Agg")
plot_lock = RLock()


def serialized_plot(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plot_lock:
            return function(*args, **kwargs)
    return wrapped
