import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from models.transfer_function import TransferFunction
from classes.defs_sympy import *
from typing import Dict

class ControlEnvironment:
    def __init__(self, name):
        self.name = name
        self.global_variables = []
        self.global_constants = {}

        self.tfs: Dict[str, TransferFunction] = {}

    def add_tf(self,name):
        self.tfs[name] = TransferFunction(self.global_variables, self.global_constants)

