# coding=utf-8
from sys import gettrace
PERFORM_EXTENSIVE_CHECKS = False

def is_debug_mode():
    return gettrace() is not None


def extensive_checks() -> bool:
    return PERFORM_EXTENSIVE_CHECKS or is_debug_mode()

def set_extensive_checks(val: bool):
    global PERFORM_EXTENSIVE_CHECKS
    PERFORM_EXTENSIVE_CHECKS = val

