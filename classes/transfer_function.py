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

from models.model_transfer_function import TransferFunctionModel
from models.model_equation import EquationModel

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
        self.zeros = TransferFunctionModel()
        self.poles = TransferFunctionModel()
        self.wcg = None
        self.wcg_found = False
        self.wcp = None
        self.wcp_found = False
        self.gm = None
        self.pm = None
        self.define_input('u')
        self.define_output('y')
    
    def update(self):
        # Define numeric transfer function
        self.tf.numeric = self.tf.symbolic.subs(self.get_constant_values())
        
        # Define numerator
        self.numerator.symbolic     = defs_tf.get_numerator(self.tf.symbolic)
        self.numerator.numeric      = defs_tf.get_numerator(self.tf.numeric)

        # Define denominator
        self.denominator.symbolic   = defs_tf.get_denominator(self.tf.symbolic)
        self.denominator.numeric    = defs_tf.get_denominator(self.tf.numeric)
        
        # Define differential equation from tf
        self.differential_equation.symbolic = defs_tf.get_equation(self.tf.symbolic, self.input, self.output)
        self.differential_equation.numeric = defs_tf.get_equation(self.tf.numeric, self.input, self.output)
        
        # Define zeros 
        self.zeros.symbolic         = defs_tf.get_zeros(self.tf.symbolic)
        self.zeros.numeric          = defs_tf.get_zeros(self.tf.numeric)

        # Define poles
        self.poles.symbolic         = defs_tf.get_poles(self.tf.symbolic)
        self.poles.numeric          = defs_tf.get_poles(self.tf.numeric)

        # Frequency stuff
        self.gm, self.pm, self.wcg, self.wcp, self.wcg_found, self.wcp_found = defs_tf.get_margin(self.tf.numeric, w_range=(), n_points=10000)

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
            "unit": f"[{unit}]",
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
        
        # Define numeric transfer function
        self.tf.numeric = self.tf.symbolic.subs(self.get_constant_values())
        
        # Define numerator
        self.numerator.symbolic     = defs_tf.get_numerator(self.tf.symbolic)
        self.numerator.numeric      = defs_tf.get_numerator(self.tf.numeric)

        # Define denominator
        self.denominator.symbolic   = defs_tf.get_denominator(self.tf.symbolic)
        self.denominator.numeric    = defs_tf.get_denominator(self.tf.numeric)
        
        # Define differential equation from tf
        self.differential_equation.symbolic = defs_tf.get_equation(self.tf.symbolic, self.input, self.output)
        self.differential_equation.numeric = defs_tf.get_equation(self.tf.numeric, self.input, self.output)
        
        # Define zeros 
        self.zeros.symbolic         = defs_tf.get_zeros(self.tf.symbolic)
        self.zeros.numeric          = defs_tf.get_zeros(self.tf.numeric)

        # Define poles
        self.poles.symbolic         = defs_tf.get_poles(self.tf.symbolic)
        self.poles.numeric          = defs_tf.get_poles(self.tf.numeric)

        # Frequency stuff
        self.gm, self.pm, self.wcg, self.wcp, self.wcg_found, self.wcp_found = defs_tf.get_margin(self.tf.numeric, w_range=(), n_points=10000)

## Plots
    def impulse(self, t_range: Tuple[float, float] = (0, 10), n_points: int = 1000, delay_time: float = 0, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            defs_plots.step(self.tf.numeric, t_range=t_range, n_points=n_points, delay_times=[delay_time], labels=[self.Name])
            return
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
            
        tf_numerics_list, delay_times_list, labels_list = defs_tf.sweep_tfs(self,tf_instances=self.tf,delay_times=delay_time,sweep_params=sweep_params,is_global=False)

        defs_plots.impulse(*tf_numerics_list, t_range=t_range, n_points=n_points, delay_times=delay_times_list, labels=labels_list)

    def step(self, t_range: Tuple[float, float] = (0, 10), n_points: int = 1000, delay_time: float = 1, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            defs_plots.step(self.tf.numeric, t_range=t_range, n_points=n_points, delay_times=[delay_time], labels=[self.Name])
            return
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
            
        tf_numerics_list, delay_times_list, labels_list = defs_tf.sweep_tfs(self,tf_instances=self.tf,delay_times=delay_time,sweep_params=sweep_params,is_global=False)

        defs_plots.step(*tf_numerics_list, t_range=t_range, n_points=n_points, delay_times=delay_times_list, labels=labels_list)

    def ramp(self, t_range: Tuple[float, float] = (0, 10), n_points: int = 1000, delay_time: float = 1, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            defs_plots.step(self.tf.numeric, t_range=t_range, n_points=n_points, delay_times=[delay_time], labels=[self.Name])
            return
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
            
        tf_numerics_list, delay_times_list, labels_list = defs_tf.sweep_tfs(self,tf_instances=self.tf,delay_times=delay_time,sweep_params=sweep_params,is_global=False)

        defs_plots.ramp(*tf_numerics_list, t_range=t_range, n_points=n_points, delay_times=delay_times_list, labels=labels_list)

    def bode(self, w_range: Tuple[float, float] = (), n_points: int = 10000, sweep_params: Dict[str, List[float]] = None):
        if not sweep_params or all(not values for values in sweep_params.values()):
            defs_plots.bode(self.tf.numeric, w_range=w_range, n_points=n_points, labels=[self.Name])
            return
        
        for var_name in sweep_params.keys():
            if var_name not in self.constants:
                raise ValueError(f"Sweep variable '{var_name}' not found in constants.")
            
        tf_numerics_list, delay_times_list, labels_list = defs_tf.sweep_tfs(self,tf_instances=self.tf,delay_times=None,sweep_params=sweep_params,is_global=False)
        
        defs_plots.bode(*tf_numerics_list, w_range=w_range, n_points=n_points, labels=labels_list)

    def margin_plot(self, w_range=(), n_points=10000):
        defs_plots.margin_plot(self.tf.numeric, w_range=w_range, n_points=n_points)
    
    def pzmap(self, x_range=(), y_range=(), n_points=10000):
        defs_plots.pzmap(self.tf.numeric, x_range=x_range, y_range=y_range, n_points=n_points)

    def nyquist(self, w_range=(), n_points=1000):
        defs_plots.nyquist(self.tf.numeric, w_range=w_range, n_points=n_points)

## Prints
    def print_all(self):
        defs_prints.print_all(self)