import sympy as sp
import numpy as np
import itertools
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import utilities.utils_transfer_function as utils_tf
import utilities.utils_plots as utils_plots
import utilities.utils_prints as utils_prints

from typing import Dict, List, Any, Tuple, Union

from models.model_transfer_function import TransferFunctionModel

# Import logger
import logging
from utilities.logger import get_logger, header, subheader, subsubheader
log = get_logger(__name__, level=logging.DEBUG, logfile='logs/main.log')


class TransferFunctionClass(TransferFunctionModel):
    def __init__(self, name, description, global_symbols, global_constants):
        super().__init__(
            name=name, 
            description=description, 
            global_symbols=global_symbols, 
            global_constants=global_constants)

## Plots
    def impulse(self, delay_times: List[float] = None, t_range: Union[Tuple[float,float],None] = None, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            header(log, f"TransferFunctionClass({self.name}): Plotting impulse response")
            utils_plots.plot_response(self, labels=[self.name], delay_times=delay_times, t_range=t_range, response_type="impulse")
            return
        header(log, f"TransferFunctionClass({self.name}): Plotting impulse response with sweep_params = {sweep_params}")
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
        tf_sweep_list, labels_list = utils_tf.sweep_parameter(self,tf_instances=self,delay_times=delay_times,sweep_params=sweep_params)    
        utils_plots.plot_response(*tf_sweep_list, labels=labels_list, delay_times=delay_times, t_range=t_range, response_type="impulse")

    def step(self, delay_times: List[float] = None, t_range: Union[Tuple[float,float],None] = None, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            header(log, f"TransferFunctionClass({self.name}): Plotting step response")
            utils_plots.plot_response(self, labels=[self.name], delay_times=delay_times, t_range=t_range, response_type="step")
            return
        header(log, f"TransferFunctionClass({self.name}): Plotting step response with sweep_params = {sweep_params}")
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
        tf_sweep_list, labels_list = utils_tf.sweep_parameter(self,tf_instances=self,delay_times=delay_times,sweep_params=sweep_params)    
        utils_plots.plot_response(*tf_sweep_list, labels=labels_list, delay_times=delay_times, t_range=t_range, response_type="step")

    def ramp(self, delay_times: List[float] = None, t_range: Union[Tuple[float,float],None] = None, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            header(log, f"TransferFunctionClass({self.name}): Plotting ramp response")
            utils_plots.plot_response(self, labels=[self.name], delay_times=delay_times, t_range=t_range, response_type="ramp")
            return
        header(log, f"TransferFunctionClass({self.name}): Plotting ramp response with sweep_params = {sweep_params}")
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
        tf_sweep_list, labels_list = utils_tf.sweep_parameter(self,tf_instances=self,delay_times=delay_times,sweep_params=sweep_params)    
        utils_plots.plot_response(*tf_sweep_list, labels=labels_list, delay_times=delay_times, t_range=t_range, response_type="ramp")

    def bode(self, w_range: Union[Tuple[float,float],None] = None, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            header(log, f"TransferFunctionClass({self.name}): Plotting bode diagram")
            utils_plots.bode(self, labels=[self.name], w_range=w_range)
            return
        header(log, f"TransferFunctionClass({self.name}): Plotting bode response with sweep_params = {sweep_params}")
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
        tf_sweep_list, labels_list = utils_tf.sweep_parameter(self,tf_instances=self,delay_times=None,sweep_params=sweep_params)  
        utils_plots.bode(*tf_sweep_list, labels=labels_list, w_range=w_range)

    def margin_plot(self, w_range: Union[Tuple[float,float],None] = None):
        header(log, f"TransferFunctionClass({self.name}): Plotting bode diagram with margins")
        utils_plots.margin_plot(self, w_range=w_range)
    
    def pzmap(self, x_range=(), y_range=(), n_points=10):
        header(log, f"TransferFunctionClass({self.name}): Plotting pole-zero map")
        utils_plots.pzmap(self, x_range=x_range, y_range=y_range, n_points=n_points)

    def nyquist(self):
        header(log, f"TransferFunctionClass({self.name}): Plotting nyquist")
        utils_plots.nyquist(self)

## Prints
    def print_all(self):
        header(log, f"TransferFunctionClass({self.name}): Printing all")
        utils_prints.print_all(self)