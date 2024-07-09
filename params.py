from enum import Enum
import numpy as np

class PhysConst(Enum):
    AU2EV = 27.2114
    GAMMA = 5.35e-03
    ALPHA = 1/137

class CompParams(Enum):
    FL_LOW = 515 / PhysConst.AU2EV.value
    FL_HIGH = 540 / PhysConst.AU2EV.value
    FL_BINS = 251
    BETA = 120
    DATAFRAMES = ['abs_unpumped', 'abs_pumped', 'fl_unpumped', 'fl_pumped']
