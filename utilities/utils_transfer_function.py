import sympy as sp
import numpy as np
import scipy
import scipy.signal as signal
import itertools
import copy
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import utilities.utils_sympy as utils_sympy

from typing import Dict, List, Any, Tuple, Optional, Union
from classes.class_transfer_function import TransferFunctionClass


def get_poles(tf):
    s = sp.Symbol('s')
    denominator = get_denominator(tf)
    roots = sp.roots(denominator, s)
    poles = list(roots.keys())
    return poles

def get_zeros(tf):
    s = sp.Symbol('s')
    numerator = get_numerator(tf)
    roots = sp.roots(numerator, s)
    zeros = list(roots.keys())
    return zeros

def get_system_order():
    pass

def get_free_zeros():
    pass

def get_free_poles():
    pass

def get_coefficients(polynomium):
    s = sp.Symbol('s')
    polynomium = sp.Poly(polynomium, s)
    coefficients = []
    for c in polynomium.all_coeffs():
        try:
            coefficients.append(float(c))  
        except:
            coefficients.append(c)
    return coefficients

def get_numerator(tf):
    if tf == 1:
        numerator = sp.S.One
        return numerator
    numerator, denominator = sp.fraction(tf)
    return numerator

def get_denominator(tf):
    if tf == 1:
        denominator = sp.S.One
        return denominator
    numerator, denominator = sp.fraction(tf)
    return denominator

def get_differential_equation(tf,input,output,symbols):
    try:
        s = symbols['s']['symbol']
        t = symbols['t']['symbol']
    except KeyError:
        print("Error: SymPy symbols 's' and 't' must be defined in symbols before calling define_input.")
        return 

    u = input[0]
    y = output[0]
    U = input[1]
    Y = output[1]

    G_num, G_den = sp.fraction(tf)
    RHS = sp.expand(G_num * U)
    LHS = sp.expand(G_den * Y)

    eq = sp.Eq(LHS,RHS)

    rhs = sp.inverse_laplace_transform(eq.rhs, s, t, noconds=True)
    lhs = sp.inverse_laplace_transform(eq.lhs, s, t, noconds=True)

    rhs = rhs.replace(sp.InverseLaplaceTransform(U, s, t, None), u)
    lhs = lhs.replace(sp.InverseLaplaceTransform(Y, s, t, None), y)

    equation = sp.Eq(lhs, rhs)

    return equation

## Define transfer function
def define_signal(numerator_coefficients: sp.Poly=None, denominator_coefficients: sp.Poly=None) -> signal.TransferFunction:
    system = signal.TransferFunction(numerator_coefficients, denominator_coefficients)
    return system

def from_string(tf_str,symbols):
    sympy_symbols = {name: data_dict['symbol'] for name, data_dict in symbols.items()}
    locals().update(sympy_symbols)
    tf = sp.together(eval(utils_sympy.translate_string(tf_str)))
    return tf

def from_coefs(num_coefs,den_coefs,symbols):
    try:
        s = symbols['s']['symbol']
    except KeyError:
        print("Error: SymPy symbols 's' and 't' must be defined in symbols before calling define_input.")
        return
    
    def evaluate_coefficient(coef_input):
        if isinstance(coef_input, (sp.Expr, sp.Number)):
            return coef_input
        return sp.sympify(coef_input, locals=symbols)

    num_exprs = [evaluate_coefficient(c) for c in num_coefs]
    den_exprs = [evaluate_coefficient(c) for c in den_coefs]
    
    order_num = len(num_exprs) - 1
    num = sum(coef * s**(order_num - i) for i, coef in enumerate(num_exprs))
    
    order_den = len(den_exprs) - 1
    den = sum(coef * s**(order_den - i) for i, coef in enumerate(den_exprs))

    tf = sp.together(num / den)

    return tf

def from_equation(lhs: str, rhs: str, input: Tuple[sp.Expr,sp.Expr], output, symbols):
    try:
        s = symbols['s']['symbol']
        t = symbols['t']['symbol']
    except KeyError:
        print("Error: SymPy symbols 's' and 't' must be defined in symbols before calling define_input.")
        return

    # Load inputs and outputs
    u = input[0]
    y = output[0]
    U = input[1]
    Y = output[1]
    
    # Create initial condition symbols and values
    subs_ic = {}
    ics = {}
    u_subs, u_symbols, u_ics = utils_sympy.make_ic_subs(rhs,u,t)
    y_subs, y_symbols, y_ics = utils_sympy.make_ic_subs(lhs,y,t)
    subs_ic = {**u_subs, **y_subs}
    ics = {**u_ics, **y_ics}

    # Translate the input strings (lhs and rhs)
    lhs = eval(utils_sympy.translate_string(lhs))
    rhs = eval(utils_sympy.translate_string(rhs))

    # Time-domain differential equation
    eq = sp.Eq(lhs,rhs)

    # Take Laplace transform of both sides
    lhs_L = sp.laplace_transform(eq.lhs, t, s, noconds=True)
    rhs_L = sp.laplace_transform(eq.rhs, t, s, noconds=True)

    # Replace LaplaceTransform(x(t), t, s) with X(s), LaplaceTransform(f(t), t, s) with F(s)
    lhs_L = lhs_L.replace(sp.LaplaceTransform(y, t, s), Y)
    rhs_L = rhs_L.replace(sp.LaplaceTransform(u, t, s), U)

    # Substitute initial conditions
    lhs_L = lhs_L.subs(subs_ic)
    rhs_L = rhs_L.subs(subs_ic)

    # Simplify Laplace-domain equation
    laplace_eq = sp.Eq(lhs_L,rhs_L)
    Y_s = sp.solve(laplace_eq, Y)[0]
    Y_s = sp.simplify(Y_s)
    Y_s = Y_s.subs(ics)

    # Transfer function G(s) = X(s)/F(s)
    tf = sp.together(Y_s / U)
    return tf


def sweep_parameter(
    self,
    tf_instances: List["TransferFunctionClass"],  
    sweep_params: Dict[str, List[float]],
    delay_times: Optional[List[float]] = None
) -> Tuple[List["TransferFunctionClass"], List[str]]:

    if not sweep_params:
        raise ValueError("sweep_params cannot be empty")
    
    if not tf_instances:
        raise ValueError("tf_instances cannot be empty")
    
    if not isinstance(tf_instances, list):
        tf_instances = [tf_instances]
    
    sweep_variables = list(sweep_params.keys())
    sweep_value_lists = list(sweep_params.values())

    # Store original values so we can restore them after the sweep
    original_values = {
        var: self.constants[var]['value']
        for var in sweep_variables
        if var in self.constants
    }
    
    tf_sweep_list = []
    labels_list = []

    try:
        # Generate all combinations of parameter values
        for combo_values in itertools.product(*sweep_value_lists):
            combo_dict = dict(zip(sweep_variables, combo_values))

            for tf_instance in tf_instances:
                tf_sweep = copy.deepcopy(tf_instance)
                
                # Apply new parameter values to this copy
                for var_name, value in combo_dict.items():
                    tf_sweep.edit_constant(var_name, value)
                
                # (Optional) Force recalculation if not automatic
                if hasattr(tf_sweep, "update"):
                    tf_sweep.update()

                combo_label = ", ".join([f"{k}={v}" for k, v in combo_dict.items()])

                tf_sweep_list.append(tf_sweep)
                labels_list.append(f"{tf_instance.name} ({combo_label})")
    
    finally:
        for tf_instance in tf_instances:
            for var_name, value in original_values.items():
                tf_instance.edit_constant(var_name, value)
    
    return tf_sweep_list, labels_list

