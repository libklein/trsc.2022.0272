# coding=utf-8
import signal
from multiprocessing import Process, Queue
from os import kill, getpid
from time import sleep

from contexttimer import timeout

_Q = Queue()


class TimedOut(Exception):
    pass


def capture_return(func, *args, **kwargs):
    _Q.put(func(*args, **kwargs))


def run_with_limited_time(func, args, kwargs, time):
    """Runs a function with time limit

    :param func: The function to run
    :param args: The functions args, given as tuple
    :param kwargs: The functions keywords, given as dict
    :param time: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    p = Process(target=capture_return, args=(func, *args), kwargs=kwargs)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        raise TimedOut

    return _Q.get_nowait()


def signal_timeout_handler(*args):
    raise TimedOut()

def send_alarm(pid: int, after_time: int):
    sleep(after_time)
    kill(pid, signal.SIGALRM)

def signal_based_timeout(func, args=(), kwargs={}, timeout_duration=1):
    # set the timeout handler
    prev_handler = signal.signal(signal.SIGALRM, signal_timeout_handler)
    signaller = Process(target=send_alarm, args=(getpid(), timeout_duration))
    signaller.daemon = True
    signaller.start()
    try:
        result = func(*args, **kwargs)
        signal.signal(signal.SIGALRM, prev_handler)
        return result
    except TimedOut as exc:
        signal.signal(signal.SIGALRM, prev_handler)
        raise TimedOut()
