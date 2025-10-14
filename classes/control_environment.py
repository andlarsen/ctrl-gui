import classes.defs_plots as defs_plots

from typing import Dict
from classes.transfer_function import TransferFunction


class ControlEnvironment:
    def __init__(self, name):
        self.name = name
        self.global_variables = []
        self.global_constants = {}

        self.tfs: Dict[str, TransferFunction] = {}

    def add_tf(self,name,description=''):
        self.tfs[name] = TransferFunction(name, description, self.global_variables, self.global_constants)

    def impulse(self, *tfs, t_range=(0, 10), n_points=1000, delay_times=None):
        if not tfs:
            tfs = self.tfs.keys()
        for name in tfs:
            if name not in self.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
        
        tf_numerics = [self.tfs[name].tf.numeric for name in tfs]
        labels = list(tfs)
        defs_plots.impulse(*tf_numerics, t_range=t_range, n_points=n_points, delay_times=delay_times, labels=labels)

    def step(self, *tfs, t_range=(0, 10), n_points=1000, delay_times=None):
        if not tfs:
            tfs = self.tfs.keys()
        for name in tfs:
            if name not in self.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
        
        tf_numerics = [self.tfs[name].tf.numeric for name in tfs]
        labels = list(tfs)
        defs_plots.step(*tf_numerics, t_range=t_range, n_points=n_points, delay_times=delay_times, labels=labels)
        
    def ramp(self, *tfs, t_range=(0, 10), n_points=1000, delay_times=None):
        if not tfs:
            tfs = self.tfs.keys()
        for name in tfs:
            if name not in self.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
        
        tf_numerics = [self.tfs[name].tf.numeric for name in tfs]
        labels = list(tfs)
        defs_plots.ramp(*tf_numerics, t_range=t_range, n_points=n_points, delay_times=delay_times, labels=labels)

    def bode(self, *tfs, w_range = (), n_points = 10000):
        if not tfs:
            tfs = self.tfs.keys()
        for name in tfs:
            if name not in self.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
        
        tf_numerics = [self.tfs[name].tf.numeric for name in tfs]
        labels = list(tfs)
        defs_plots.bode(*tf_numerics, w_range=w_range, n_points=n_points, labels=labels)