
import classes.defs_plots as defs_plots
import classes.defs_sympy as defs_sympy
import classes.defs_tf as defs_tf

from typing import Dict, List, Tuple, Any
from models.model_components import ComponentsModel
from classes.transfer_function import TransferFunctionObject

class ControlSystem:
    def __init__(self, name):
        self.name = name
        self.global_symbols = []
        self.global_constants = {}

        self.components = ComponentsModel()

    def update(self):
        for tf_name, tf_instance in self.components.tfs.items():
            tf_instance.update()

    def add_tf(self,name,description=''):
        tf = TransferFunctionObject(name, description, self.global_symbols, self.global_constants)
        tf.parent = self.components
        tf.control_system = self
        self.components.tfs[name] = tf

    def remove_tf(self):
        pass

    def add_gain(self):
        pass

    def remove_gain(self):
        pass

    def add_sum(self):
        pass

    def remove_sum(self):
        pass

    def add_saturation(self):
        pass

    def remove_saturation(self):
        pass

    def add_input_generator(self):
        pass

    def remove_input_generator(self):
        pass

    def add_scope(self):
        pass

    def remove_scope(self):
        pass

    def add_global_constant(self, name: str, value: float, description='None', unit='-', is_global=True):
        symbol = defs_sympy.add_symbol(name, is_real=True, is_positive=True)
        self.global_symbols.append(symbol)
        self.global_constants[name] = {
            "value": value,
            "description": description,
            "unit": f"{unit}",
            "symbol": symbol,
            "is_global": is_global}
        for tf_name, tf_instance in self.components.tfs.items():
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
        for tf_name in self.components.tfs:
            self.components.tfs[tf_name].edit_constant(name, value, description, unit, is_global=True)

    def impulse(self, *tfs: str, delay_times: List[float] = None, sweep_params: Dict[str, List[float]] = None):    
        if not tfs:
            tfs = self.components.tfs.keys()
        
        tf_instances = []
        for name in tfs:
            if name not in self.components.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
            tf_instances.append(self.components.tfs[name])

        num_tfs = len(tf_instances)
        
        if delay_times is None:
            delay_times = [0] * num_tfs
        elif len(delay_times) != num_tfs:
            raise ValueError("Number of delay times must match number of transfer functions.")

        if not sweep_params or all(not values for values in sweep_params.values()):
            responses = [tf_instance.tf.numeric for tf_instance in tf_instances]
            labels = list(tfs)
            defs_plots.plot_response(*responses, labels=labels)
            return
        
        responses_list, labels_list = defs_tf.sweep_impulse_responses(self,tf_instances=tf_instances,delay_times=delay_times,sweep_params=sweep_params,is_global=True)
        defs_plots.plot_response(*responses_list, labels=labels_list)

    def step(self, *tfs: str, delay_times: List[float] = None, sweep_params: Dict[str, List[float]] = None):    
        if not tfs:
            tfs = self.components.tfs.keys()
        
        tf_instances = []
        for name in tfs:
            if name not in self.components.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
            tf_instances.append(self.components.tfs[name])

        num_tfs = len(tf_instances)
        
        if delay_times is None:
            delay_times = [1] * num_tfs
        elif len(delay_times) != num_tfs:
            raise ValueError("Number of delay times must match number of transfer functions.")

        if not sweep_params or all(not values for values in sweep_params.values()):
            step_responses = [tf_instance.step_response for tf_instance in tf_instances]
            labels = list(tfs)
            defs_plots.plot_response(*step_responses, labels=labels)
            return
        
        responses_list, labels_list = defs_tf.sweep_step_responses(self,tf_instances=tf_instances,delay_times=delay_times,sweep_params=sweep_params,is_global=True)
        defs_plots.plot_response(*responses_list, labels=labels_list)

    def ramp(self, *tfs: str, t_range: Tuple[float, float] = (0, 10), n_points: int = 1000, delay_times: List[float] = None, sweep_params: Dict[str, List[float]] = None):    
        if not tfs:
            tfs = self.components.tfs.keys()
        
        tf_instances = []
        for name in tfs:
            if name not in self.components.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
            tf_instances.append(self.components.tfs[name])

        num_tfs = len(tf_instances)
        
        if delay_times is None:
            delay_times = [1] * num_tfs
        elif len(delay_times) != num_tfs:
            raise ValueError("Number of delay times must match number of transfer functions.")

        if not sweep_params or all(not values for values in sweep_params.values()):
            tf_numerics = [tf_instance.tf.numeric for tf_instance in tf_instances]
            labels = list(tfs)
            defs_plots.ramp(*tf_numerics, t_range=t_range, n_points=n_points, delay_times=delay_times, labels=labels)
            return
        
        tf_numerics_list, delay_times_list, labels_list = defs_tf.sweep_tfs(self,tf_instances=tf_instances,delay_times=delay_times,sweep_params=sweep_params,is_global=True)

        defs_plots.ramp(*tf_numerics_list, t_range=t_range, n_points=n_points, delay_times=delay_times_list, labels=labels_list)

    def bode(self, *tfs: str, w_range: Tuple[float, float] = (0.1, 100), n_points: int = 1000, sweep_params: Dict[str, List[float]] = None):    
        if not tfs:
            tfs = self.components.tfs.keys()
        
        tf_instances = []
        for name in tfs:
            if name not in self.components.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
            tf_instances.append(self.components.tfs[name])

        if not sweep_params or all(not values for values in sweep_params.values()):
            tf_numerics = [tf_instance.tf.numeric for tf_instance in tf_instances]
            labels = list(tfs)
            defs_plots.bode(*tf_numerics, w_range=w_range, n_points=n_points, labels=labels)
            return
        
        tf_numerics_list, delay_times_list, labels_list = defs_tf.sweep_tfs(self,tf_instances=tf_instances,delay_times=None,sweep_params=sweep_params,is_global=True)
        
        defs_plots.bode(*tf_numerics_list, w_range=w_range, n_points=n_points, labels=labels_list)
