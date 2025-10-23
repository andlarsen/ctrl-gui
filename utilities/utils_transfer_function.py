"""
Transfer Function Utilites

Author: Andreas
Date: 2025
"""

# Import of modules
import sympy as sp
import numpy as np
import scipy
import scipy.signal as signal
import itertools
import copy
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from typing import Dict, List, Any, Tuple, Optional, Union
from classes.class_transfer_function import TransferFunctionClass

# Import all utilities
import utilities.utils_sympy as utils_sympy

# Import logger
import logging
from utilities.logger import get_logger
log = get_logger(__name__, level=logging.DEBUG, logfile='logs/main.log')


# ============================================================================
# Transfer function properties
# ============================================================================

def get_poles(tf):
    log.debug(f"get_poles() called with tf: {tf}")

    # Input validation
    if tf is None:
        log.warning("get_poles(): Received None as transfer function.")
        return []
    if not isinstance(tf, sp.Basic):
        log.warning(f"get_poles(): Invalid type {type(tf)}. Expected SymPy expression.")
        return []

    try:
        s = sp.Symbol('s')
        denominator = get_denominator(tf)
        roots = sp.roots(denominator, s)
        poles = list(roots.keys())
        log.debug(f"get_poles(): Found poles: {poles}")
        return poles
    except Exception as e:
        log.error(f"get_poles(): Failed to compute poles: {e}", exc_info=True)
        return []

def get_zeros(tf):
    log.debug(f"get_zeros() called with tf: {tf}")

    if tf is None:
        log.warning("get_zeros(): Received None as transfer function.")
        return []
    if not isinstance(tf, sp.Basic):
        log.warning(f"get_zeros(): Invalid type {type(tf)}. Expected SymPy expression.")
        return []

    try:
        s = sp.Symbol('s')
        numerator = get_numerator(tf)
        roots = sp.roots(numerator, s)
        zeros = list(roots.keys())
        log.debug(f"get_zeros(): Found zeros: {zeros}")
        return zeros
    except Exception as e:
        log.error(f"get_zeros(): Failed to compute zeros: {e}", exc_info=True)
        return []

def get_coefficients(polynomium):
    log.debug(f"get_coefficients() called with polynomium: {polynomium}")

    if polynomium is None:
        log.warning("get_coefficients(): Received None as input.")
        return []

    try:
        s = sp.Symbol('s')
        poly = sp.Poly(polynomium, s)
        coefficients = []
        for c in poly.all_coeffs():
            try:
                coefficients.append(float(c))
            except Exception:
                coefficients.append(c)
        log.debug(f"get_coefficients(): Extracted coefficients: {coefficients}")
        return coefficients
    except Exception as e:
        log.error(f"get_coefficients(): Failed to extract coefficients: {e}", exc_info=True)
        return []

def get_numerator(tf):
    log.debug(f"get_numerator() called with tf: {tf}")

    if tf is None:
        log.warning("get_numerator(): Received None as transfer function.")
        return None

    try:
        if tf == 1:
            return sp.S.One
        numerator, _ = sp.fraction(tf)
        log.debug(f"get_numerator(): Extracted numerator: {numerator}")
        return numerator
    except Exception as e:
        log.error(f"get_numerator(): Failed to extract numerator: {e}", exc_info=True)
        return None

def get_denominator(tf):
    log.debug(f"get_denominator() called with tf: {tf}")

    if tf is None:
        log.warning("get_denominator(): Received None as transfer function.")
        return None

    try:
        if tf == 1:
            return sp.S.One
        _, denominator = sp.fraction(tf)
        log.debug(f"get_denominator(): Extracted denominator: {denominator}")
        return denominator
    except Exception as e:
        log.error(f"get_denominator(): Failed to extract denominator: {e}", exc_info=True)
        return None

def get_differential_equation(tf, input, output, symbols):
    log.debug(f"get_differential_equation() called with tf: {tf}")

    # Input validation
    if not isinstance(symbols, dict) or 's' not in symbols or 't' not in symbols:
        log.error("get_differential_equation(): Missing required symbols 's' and 't'.")
        return None
    if not input or not output or len(input) < 2 or len(output) < 2:
        log.error("get_differential_equation(): Input/output must contain [symbolic, Laplace] pairs.")
        return None

    try:
        s = symbols['s'].symbol
        t = symbols['t'].symbol
        u, U = input
        y, Y = output

        G_num, G_den = sp.fraction(tf)
        RHS = sp.expand(G_num * U)
        LHS = sp.expand(G_den * Y)

        eq = sp.Eq(LHS, RHS)
        rhs = sp.inverse_laplace_transform(eq.rhs, s, t, noconds=True)
        lhs = sp.inverse_laplace_transform(eq.lhs, s, t, noconds=True)

        rhs = rhs.replace(sp.InverseLaplaceTransform(U, s, t, None), u)
        lhs = lhs.replace(sp.InverseLaplaceTransform(Y, s, t, None), y)

        equation = sp.Eq(lhs, rhs)
        log.debug(f"get_differential_equation(): Derived equation: {equation}")
        return equation
    except Exception as e:
        log.error(f"get_differential_equation(): Failed to derive equation: {e}", exc_info=True)
        return None

# Placeholders for future implementations
def get_system_order():
    log.debug("get_system_order() not yet implemented.")
    pass

def get_free_zeros():
    log.debug("get_free_zeros() not yet implemented.")
    pass

def get_free_poles():
    log.debug("get_free_poles() not yet implemented.")
    pass


# ============================================================================
# Transfer function definitions
# ============================================================================

def define_signal(numerator_coefficients: sp.Poly = None, denominator_coefficients: sp.Poly = None) -> signal.TransferFunction:
    log.debug("define_signal() called")

    if numerator_coefficients is None or denominator_coefficients is None:
        log.warning("define_signal(): Missing numerator or denominator coefficients")
        raise ValueError("Both numerator and denominator coefficients must be provided")
    try:
        system = signal.TransferFunction(numerator_coefficients, denominator_coefficients)
        log.debug(f"define_signal(): Created TransferFunctionContinuous(num={system.num.tolist()}, den={system.den.tolist()}, dt={system.dt})")
        return system
    except Exception as e:
        log.error(f"define_signal(): Failed to create TransferFunction: {e}", exc_info=True)
        raise

def from_string(tf_str: str, symbols: Dict[str, Any]) -> sp.Expr:
    log.debug(f"from_string() called with tf_str: {tf_str}")

    if not tf_str or not isinstance(tf_str, str):
        log.error("from_string(): Transfer function string is invalid")
        raise ValueError("Transfer function string cannot be empty or non-string")

    if not symbols or not isinstance(symbols, dict):
        log.error("from_string(): Symbols dictionary is invalid")
        raise ValueError("Symbols dictionary cannot be empty")

    symbols_dict = {name: data.symbol for name, data in symbols.items()}

    try:
        tf_str_translated = utils_sympy.translate_string(tf_str)
        log.debug(f"from_string(): Translated string: '{tf_str_translated}'")
    except Exception as e:
        log.error(f"from_string(): Failed to translate string '{tf_str}': {e}")
        raise ValueError(f"Invalid transfer function syntax: {e}")

    try:
        tf_eval = eval(tf_str_translated, {"__builtins__": {}}, symbols_dict)
    except NameError as e:
        log.error(f"from_string(): Undefined symbol in '{tf_str}': {e}", exc_info=True)
        raise KeyError(f"Transfer function references undefined symbol: {e}")
    except SyntaxError as e:
        log.error(f"from_string(): Syntax error in '{tf_str}': {e}", exc_info=True)
        raise ValueError(f"Invalid transfer function syntax: {e}")
    except Exception as e:
        log.error(f"from_string(): Failed to evaluate '{tf_str}': {e}", exc_info=True)
        raise ValueError(f"Could not evaluate transfer function: {e}")

    try:
        tf = sp.together(tf_eval)
        log.debug(f"from_string(): Successfully defined transfer function: {tf}")
        return tf
    except Exception as e:
        log.error(f"from_string(): Failed to simplify expression: {e}", exc_info=True)
        raise ValueError(f"Could not simplify transfer function: {e}") from e

def from_coefs(num_coefs, den_coefs, symbols):
    log.debug("from_coefs() called")

    if not num_coefs or not den_coefs:
        log.error("from_coefs(): Coefficient lists cannot be empty")
        raise ValueError("Numerator and denominator coefficient lists must be provided")

    try:
        s = symbols['s'].symbol
    except KeyError:
        log.error("from_coefs(): Missing 's' symbol in symbols dictionary")
        raise KeyError("SymPy symbol 's' must be defined in symbols")

    def evaluate_coefficient(coef_input):
        if isinstance(coef_input, (sp.Expr, sp.Number)):
            return coef_input
        return sp.sympify(coef_input, locals=symbols)

    try:
        num_exprs = [evaluate_coefficient(c) for c in num_coefs]
        den_exprs = [evaluate_coefficient(c) for c in den_coefs]

        order_num = len(num_exprs) - 1
        num = sum(coef * s**(order_num - i) for i, coef in enumerate(num_exprs))

        order_den = len(den_exprs) - 1
        den = sum(coef * s**(order_den - i) for i, coef in enumerate(den_exprs))

        tf = sp.together(num / den)
        log.debug(f"from_coefs(): Constructed transfer function: {tf}")
        return tf
    except Exception as e:
        log.error(f"from_coefs(): Failed to construct transfer function: {e}", exc_info=True)
        raise

def from_equation(lhs: str, rhs: str, input: Tuple[sp.Expr, sp.Expr], output, symbols):
    log.debug("from_equation() called")

    try:
        s = symbols['s'].symbol
        t = symbols['t'].symbol
    except KeyError:
        log.error("from_equation(): Missing 's' or 't' in symbols")
        raise KeyError("Symbols 's' and 't' must be defined")

    try:
        u, U = input
        y, Y = output

        u_subs, _, u_ics = utils_sympy.make_ic_subs(rhs, u, t)
        y_subs, _, y_ics = utils_sympy.make_ic_subs(lhs, y, t)
        subs_ic = {**u_subs, **y_subs}
        ics = {**u_ics, **y_ics}

        lhs_eval = eval(utils_sympy.translate_string(lhs))
        rhs_eval = eval(utils_sympy.translate_string(rhs))

        eq = sp.Eq(lhs_eval, rhs_eval)
        lhs_L = sp.laplace_transform(eq.lhs, t, s, noconds=True)
        rhs_L = sp.laplace_transform(eq.rhs, t, s, noconds=True)

        lhs_L = lhs_L.replace(sp.LaplaceTransform(y, t, s), Y)
        rhs_L = rhs_L.replace(sp.LaplaceTransform(u, t, s), U)

        lhs_L = lhs_L.subs(subs_ic)
        rhs_L = rhs_L.subs(subs_ic)

        laplace_eq = sp.Eq(lhs_L, rhs_L)
        Y_s = sp.solve(laplace_eq, Y)[0]
        Y_s = sp.simplify(Y_s).subs(ics)

        tf = sp.together(Y_s / U)
        log.debug(f"from_equation(): Derived transfer function: {tf}")
        return tf
    except Exception as e:
        log.error(f"from_equation(): Failed to derive transfer function: {e}", exc_info=True)
        raise

def sweep_parameter(
    self,
    tf_instances: List["TransferFunctionClass"],
    sweep_params: Dict[str, List[float]],
    delay_times: Optional[List[float]] = None
) -> Tuple[List["TransferFunctionClass"], List[str]]:

    log.debug("sweep_parameter() called")

    if not sweep_params:
        log.error("sweep_parameter(): sweep_params is empty")
        raise ValueError("sweep_params cannot be empty")

    if not tf_instances:
        log.error("sweep_parameter(): tf_instances is empty")
        raise ValueError("tf_instances cannot be empty")

    if not isinstance(tf_instances, list):
        tf_instances = [tf_instances]

    sweep_variables = list(sweep_params.keys())
    sweep_value_lists = list(sweep_params.values())

    original_values = {
        name: self.constants[name].value
        for name in sweep_variables
        if name in self.constants
    }

    tf_sweep_list = []
    labels_list = []

    try:
        for combo_values in itertools.product(*sweep_value_lists):
            combo_dict = dict(zip(sweep_variables, combo_values))

            for tf_instance in tf_instances:
                tf_sweep = copy.deepcopy(tf_instance)

                for var_name, value in combo_dict.items():
                    tf_sweep.edit_constant(var_name, value)

                combo_label = ", ".join([f"{k}={v}" for k, v in combo_dict.items()])
                tf_sweep_list.append(tf_sweep)
                labels_list.append(f"{tf_instance.name} ({combo_label})")

        log.debug(f"sweep_parameter(): Generated {len(tf_sweep_list)} swept transfer functions")
    except Exception as e:
        log.error(f"sweep_parameter(): Error during sweep: {e}", exc_info=True)
        raise
    finally:
        for tf_instance in tf_instances:
            for var_name, value in original_values.items():
                tf_instance.edit_constant(var_name, value)
        log.debug("sweep_parameter(): Restored original constant values")

    return tf_sweep_list, labels_list
