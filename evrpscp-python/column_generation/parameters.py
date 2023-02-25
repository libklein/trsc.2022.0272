# coding=utf-8
from math import log10
from evrpscp import EPS

BIG_M_COEF = 10000.0
INTEGRALITY_TOLERANCE = 1e-5
INTEGRALITY_PRECISION = -round(log10(INTEGRALITY_TOLERANCE))