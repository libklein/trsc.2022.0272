from math import isclose as __isclose, log10

PRECISION = 1e-02
SIGNIFICANT_DIGITS = int(round(abs(log10(PRECISION))))

def is_close(a: float, b:float):
    return __isclose(a, b, abs_tol=PRECISION)


def clamp(val: float) -> float:
    return round(val, SIGNIFICANT_DIGITS)

def round_up(val: float, to: float) -> float:
    return to if to - 0.01 <= val <= to else val

def round_down(val: float, to: float) -> float:
    return to if to <= val <= to + 0.01 else val

def round_to(val: float, to: float) -> float:
    return to if abs(to - val) < 0.01 else val
