import sympy as sp
import numpy as np
import itertools
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import classes.defs_sympy as defs_sympy
import classes.defs_tf as defs_tf
import classes.defs_plots as defs_plots
import classes.defs_prints as defs_prints

from typing import Dict, List, Any, Tuple
from pprint import pprint

from models.model_transfer_function import TransferFunctionModel
from models.model_equation import EquationModel
from models.model_zeropole import ZeroPoleModel
from models.model_response import Response

class TransferFunction:
    def __init__(self, name, description, global_symbols, global_constants):
        self.name = defs_sympy.lowercase_first(name)
        self.Name = defs_sympy.uppercase_first(name)
        self.description = description
        self.functions =  []
        self.symbols = global_symbols[:]
        self.constants = global_constants.copy()
        self.input = None
        self.output = None
        self.differential_equation = EquationModel()
        self.tf = TransferFunctionModel()
        self.numerator = TransferFunctionModel()
        self.denominator = TransferFunctionModel()
        self.zeros = ZeroPoleModel()
        self.poles = ZeroPoleModel()
        self.w_range = ()
        self.wcg = None
        self.wcg_found = False
        self.wcp = None
        self.wcp_found = False
        self.gm = None
        self.pm = None
        self.impulse_response = Response(response_type='impulse')
        self.step_response = Response(response_type='step')
        self.ramp_response = Response(response_type='ramp')
        self.define_input('u')
        self.define_output('y')
    
    def update(self):
        self.tf.numeric = self.tf.symbolic.subs(self.get_constant_values())
        self.get_tf_info()

## Defines, adds, removes, gets
    def define_input(self, name: str):
        self.input = defs_sympy.add_function(name)
        self.functions.append(self.input)

    def define_output(self, name: str):
        self.output = defs_sympy.add_function(name)
        self.functions.append(self.output)

    def add_constant(self, name: str, value: float, description='None', unit='-', is_global=False):
        symbol = defs_sympy.add_symbol(name, is_real=True, is_positive=True)
        self.symbols.append(symbol)
        self.constants[name] = {
            "value": value,
            "description": description,
            "unit": f"{unit}",
            "symbol": symbol,
            "is_global": is_global}
        
    def remove_constant(self):
        pass

    def edit_constant(self, name: str, value: float, description='None', unit='-', is_global=False):
        if name in self.constants:
            symbol = self.constants[name]['symbol']
        else:
            symbol = defs_sympy.add_symbol(name, is_real=True, is_positive=True)

        self.constants[name] = {
            "value": value,
            "description": description,
            "unit": f"[{unit}]",
            "symbol": symbol,
            "is_global": is_global}
        self.update()
        
    def get_constant_symbols(self):
        return {name: data["symbol"] for name, data in self.constants.items()}
        
    def get_constant_symbols_dict(self):
        return {
            sp.Symbol(name): sp.Symbol(name, real=True, positive=True) 
            for name in self.constants.keys()}

    def get_constant_values(self):
        return {sp.Symbol(name): data["value"] for name, data in self.constants.items()}
        
    def remove_constant(self):
        pass

    def define_tf(self, *args, **kwargs):
        """
        Define transfer function from various input formats:
            define_tf("1/(s+1)")                            # from string
            define_tf(num_coefs=[1], den_coefs=[1, 1])      # from coefficients
            define_tf(lhs=..., rhs=...)                     # from differential equation
        """
        
        # Determine input type and get symbolic transfer function
        if len(args) == 1 and isinstance(args[0], str):
            # From string
            self.tf.symbolic = defs_tf.from_string(args[0], self.constants)
            
        elif 'num_coefs' in kwargs and 'den_coefs' in kwargs:
            # From coefficients
            num_coefs = kwargs.get('num_coefs', [])
            den_coefs = kwargs.get('den_coefs', [])
            self.tf.symbolic = defs_tf.from_coefs(num_coefs, den_coefs, self.constants)
            
        elif 'lhs' in kwargs and 'rhs' in kwargs:
            # From differential equation
            lhs = kwargs['lhs']
            rhs = kwargs['rhs']
            self.tf.symbolic = defs_tf.from_equation(lhs, rhs, self.input, self.output, self.constants)
            
        else:
            raise ValueError(
                "Invalid arguments. Use one of:\n"
                "  define_tf('string')  # e.g., '1/(s+1)'\n"
                "  define_tf(num_coefs=[...], den_coefs=[...])\n"
                "  define_tf(lhs=..., rhs=...)"
            )

        self.tf.numeric = self.tf.symbolic.subs(self.get_constant_values())
        self.get_tf_info()

    def open_loop(self, *tfs: str):
        if not tfs:
            tfs = self.parent.tfs.keys()
        
        for name in tfs:
            if name not in self.parent.tfs:
                raise ValueError(f"Transfer function '{name}' not found")
            tfs: Dict[str, TransferFunction] = {name: self.parent.tfs[name] for name in tfs}
        
        for tf in tfs.values():
                for name, data in tf.constants.items():
                    if name not in self.constants:
                        self.add_constant(name, data['value'], data['description'], data['unit'],data['is_global'])

        for i, tf in enumerate(tfs.values()):
            if i == 0:
                self.tf.symbolic = tf.tf.symbolic
            else:
                self.tf.symbolic = self.tf.symbolic*tf.tf.symbolic
        
        # Define numeric transfer function and get tf info
        self.tf.numeric = self.tf.symbolic.subs(self.get_constant_values())
        self.get_tf_info()

    def closed_loop(self, *feedforward_tfs: str, **kwargs):

        # --- Validate inputs ---
        if not feedforward_tfs:
            raise ValueError("Feedforward transfer function(s) not found")

        feedback_tf_names = kwargs.get('feedback_tfs', ['H'])

        # --- Collect feedforward transfer functions ---
        feedforward_tfs = {name: self.parent.tfs[name] for name in feedforward_tfs if name in self.parent.tfs}

        if not feedforward_tfs:
            raise ValueError("No valid feedforward transfer functions found")

        # --- Handle feedback TFs ---
        feedback_tfs = {}
        for name in feedback_tf_names:
            if name in self.parent.tfs:
                feedback_tfs[name] = self.parent.tfs[name]
            else:
                fb = TransferFunction(name=name,description='Unit feedback',global_symbols=[],global_constants={})
                fb.define_tf('1') 
                feedback_tfs[name] = fb

            for tf in feedforward_tfs.values():
                for name, data in tf.constants.items():
                    if name not in self.constants:
                        self.add_constant(name, data['value'], data['description'], data['unit'],data['is_global'])

            for tf in feedback_tfs.values():
                for name, data in tf.constants.items():
                    if name not in self.constants:
                        self.add_constant(name, data['value'], data['description'], data['unit'],data['is_global'])

        for i, tf in enumerate(feedforward_tfs.values()):
            if i == 0:
                G = tf.tf.symbolic
            else:
                G = G*tf.tf.symbolic
        for i, tf in enumerate(feedback_tfs.values()):
            if i == 0:
                H = tf.tf.symbolic
            else:
                H = H*tf.tf.symbolic

        self.tf.symbolic= sp.simplify(G / (1 + G * H))

        # Define numeric transfer function and get tf info
        self.tf.numeric = self.tf.symbolic.subs(self.get_constant_values())
        self.get_tf_info()

    def get_tf_info(self):        
        # Define numerator
        self.tf.numerator.symbolic              = defs_tf.get_numerator(self.tf.symbolic)
        self.tf.numerator.numeric               = defs_tf.get_numerator(self.tf.numeric)
        self.tf.numerator.coefficients.symbolic = defs_tf.get_coefficients(self.tf.numerator.symbolic)
        self.tf.numerator.coefficients.numeric  = defs_tf.get_coefficients(self.tf.numerator.numeric)

        # Define denominator
        self.tf.denominator.symbolic                = defs_tf.get_denominator(self.tf.symbolic)
        self.tf.denominator.numeric                 = defs_tf.get_denominator(self.tf.numeric)
        self.tf.denominator.coefficients.symbolic   = defs_tf.get_coefficients(self.tf.denominator.symbolic)
        self.tf.denominator.coefficients.numeric    = defs_tf.get_coefficients(self.tf.denominator.numeric)

        self.tf.signal = defs_tf.define_signal(self.tf.numerator.coefficients.numeric,self.tf.denominator.coefficients.numeric)

        # Define differential equation from tf
        self.differential_equation.symbolic = defs_tf.get_equation(self.tf.symbolic, self.input, self.output)
        self.differential_equation.numeric  = defs_tf.get_equation(self.tf.numeric, self.input, self.output)

        # Define zeros 
        self.zeros.symbolic         = defs_tf.get_zeros(self.tf.symbolic)
        self.zeros.numeric          = defs_tf.get_zeros(self.tf.numeric)

        # Define poles
        self.poles.symbolic         = defs_tf.get_poles(self.tf.symbolic)
        self.poles.numeric          = defs_tf.get_poles(self.tf.numeric)
        
        # Time-domain stuff
        self.impulse_response       = defs_tf.get_time_response(self.tf, response_type='impulse', t_range=None, delay_time=0, n_tau=8, tol=1e-6)
        self.step_response          = defs_tf.get_time_response(self.tf, response_type='step', t_range=None, delay_time=0, n_tau=8, tol=1e-6)
        self.ramp_response          = defs_tf.get_time_response(self.tf, response_type='ramp', t_range=None, delay_time=0, n_tau=8, tol=1e-6)

        # Frequency-domain stuff
        self.w_range = defs_tf.get_frequency_response(self.tf.numeric, w_range=(), n_points=500)
        self.gm, self.pm, self.wcg, self.wcp, self.wcg_found, self.wcp_found = defs_tf.get_margin(self.tf.numeric, w_range=(), n_points=500)

        print(self.step_response)

## Plots
    def impulse(self, delay_time: float = 0, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            defs_plots.plot_response(self.impulse_response, labels=[self.Name])
            return
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
            
        responses_list, labels_list = defs_tf.sweep_impulse_responses(self,tf_instances=self.tf,delay_times=delay_time,sweep_params=sweep_params,is_global=False)    
        defs_plots.plot_response(*responses_list, labels=labels_list)

    def step(self, delay_time: float = 1, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            defs_plots.plot_response(self.step_response,labels=[self.Name])
            return
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
            
        responses, labels_list = defs_tf.sweep_step_responses(self,tf_instances=self.tf,delay_times=delay_time,sweep_params=sweep_params,is_global=False)    
        defs_plots.plot_response(*responses, labels=labels_list)

    def ramp(self, delay_time: float = 1, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            defs_plots.plot_response(self.ramp_response,labels=[self.Name])
            return
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")

        responses, labels_list = defs_tf.sweep_ramp_responses(self,tf_instances=self.tf,delay_times=delay_time,sweep_params=sweep_params,is_global=False)    
        defs_plots.plot_response(*responses, labels=labels_list)

    def bode(self, w_range: Tuple[float, float] = (0.1, 100), n_points: int = 10000, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            defs_plots.bode(self.tf.numeric, w_range=w_range, n_points=n_points, labels=[self.Name])
            return
        
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
            
        tf_numerics_list, delay_times_list, labels_list = defs_tf.sweep_tfs(self,tf_instances=self.tf,delay_times=None,sweep_params=sweep_params,is_global=False)
        
        defs_plots.bode(*tf_numerics_list, w_range=w_range, n_points=n_points, labels=labels_list)

    def margin_plot(self, w_range=(), n_points=500):
        defs_plots.margin_plot(self.tf.numeric, w_range=w_range, n_points=n_points)
    
    def pzmap(self, x_range=(), y_range=(), n_points=500):
        defs_plots.pzmap(self.tf.numeric, x_range=x_range, y_range=y_range, n_points=n_points)

    def nyquist(self, w_range=(), n_points=500):
        defs_plots.nyquist(self.tf.numeric, w_range=w_range, n_points=n_points)

## Prints
    def print_all(self):
        defs_prints.print_all(self)