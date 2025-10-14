import itertools
import classes.defs_plots as defs_plots
import classes.defs_sympy as defs_sympy

from typing import Dict, List, Tuple, Any
from classes.transfer_function import TransferFunction


class ControlSystem:
    def __init__(self, name):
        self.name = name
        self.global_symbols = []
        self.global_constants = {}

        self.tfs: Dict[str, TransferFunction] = {}

    def update(self):
        for tf_name, tf_instance in self.tfs.items():
            tf_instance.update()

    def add_tf(self,name,description=''):
        self.tfs[name] = TransferFunction(name, description, self.global_symbols, self.global_constants)

    def remove_tf(self):
        pass

    def add_global_constant(self, name: str, value: float, description='None', unit='-', is_global=True):
        symbol = defs_sympy.add_symbol(name, is_real=True, is_positive=True)
        self.global_symbols.append(symbol)
        self.global_constants[name] = {
            "value": value,
            "description": description,
            "unit": f"[{unit}]",
            "symbol": symbol,
            "is_global": is_global}
        for tf_name, tf_instance in self.tfs.items():
            tf_instance.symbols.append(symbol)
            tf_instance.constants[name] = self.global_constants[name]
    
    def remove_global_constant(self):
        pass

    def edit_global_constant(self, name: str, value: float, description='None', unit='-',is_global=True):
        if name in self.global_constants:
            symbol = self.global_constants[name]['symbol']
        else:
            symbol = defs_sympy.add_symbol(name, is_real=True, is_positive=True)

        self.global_constants[name] = {
            "value": value,
            "description": description,
            "unit": f"[{unit}]",
            "symbol": symbol,
            "is_global": is_global}
        for tf_name in self.tfs:
            self.tfs[tf_name].edit_constant(name, value, description, unit, is_global=True)

    def impulse(self, *tfs, t_range=(0, 10), n_points=1000, delay_times=None):
        if not tfs:
            tfs = self.tfs.keys()
        for name in tfs:
            if name not in self.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
        
        tf_numerics = [self.tfs[name].tf.numeric for name in tfs]
        labels = list(tfs)
        defs_plots.impulse(*tf_numerics, t_range=t_range, n_points=n_points, delay_times=delay_times, labels=labels)

    def step(self, *tfs: str, t_range: Tuple[float, float] = (0, 10), n_points: int = 1000, delay_times: List[float] = None, sweep_params: Dict[str, List[float]] = None):    
        if not tfs:
            tfs = self.tfs.keys()
        
        tf_instances = []
        for name in tfs:
            if name not in self.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
            tf_instances.append(self.tfs[name])

        num_tfs = len(tf_instances)
        
        if delay_times is None:
            delay_times = [1] * num_tfs
        elif len(delay_times) != num_tfs:
            raise ValueError("Number of delay times must match number of transfer functions.")


        if not sweep_params or all(not values for values in sweep_params.values()):
            tf_numerics = [tf_instance.tf.numeric for tf_instance in tf_instances]
            labels = list(tfs)
            defs_plots.step(*tf_numerics, t_range=t_range, n_points=n_points, delay_times=delay_times, labels=labels)
            return
        
        sweep_variables = list(sweep_params.keys())
        sweep_value_lists = list(sweep_params.values())
        
        original_global_values = {var: self.global_constants[var]['value'] for var in sweep_variables if var in self.global_constants}

        tf_numerics_list = []
        labels_list = []
        delay_times_final = []

        for combo_values in itertools.product(*sweep_value_lists):
            combo_label_parts = []
            
            for var_name, value in zip(sweep_variables, combo_values):
                self.edit_global_constant(var_name, value=value) 
                combo_label_parts.append(f"{var_name}={value}")

            base_label = ", ".join(combo_label_parts)

            for i, tf_instance in enumerate(tf_instances):
                self.update()
                tf_numeric = tf_instance.tf.numeric 
                
                tf_numerics_list.append(tf_numeric)
                delay_times_final.append(delay_times[i])
                
                labels_list.append(f"{tf_instance.Name} ({base_label})")
        
        for var_name, value in original_global_values.items():
            self.edit_global_constant(var_name, value=value)
        
        defs_plots.step(*tf_numerics_list, t_range=t_range, n_points=n_points, delay_times=delay_times_final, labels=labels_list)

    # def step(self, *tfs, t_range=(0, 10), n_points=1000, delay_times=None):
    #     if not tfs:
    #         tfs = self.tfs.keys()
    #     for name in tfs:
    #         if name not in self.tfs:
    #             raise ValueError(f"Transfer function '{name}' not found")
        
    #     tf_numerics = [self.tfs[name].tf.numeric for name in tfs]
    #     labels = list(tfs)
    #     defs_plots.step(*tf_numerics, t_range=t_range, n_points=n_points, delay_times=delay_times, labels=labels)
        
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