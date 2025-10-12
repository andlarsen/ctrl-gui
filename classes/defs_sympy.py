import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

def define_st():
    s = add_symbol('s')
    t = add_symbol('t', is_real=True)
    return s, t

def add_symbol(name: str, is_real=None, is_positive=None, print_output=False):
    symbol = sp.symbols(name, real = is_real, positive = is_positive)
    if print_output:
            print(f"  {symbol}, is_real: {symbol.is_real}, is_positive: {symbol.is_positive}")
    return symbol

def remove_symbol(name: str, symbol_list: list, print_output=False):
    for symbol in symbol_list:
        if symbol == name:
            symbol_list.remove(symbol)
            if print_output:
                print(f"Removed symbol: {name}")
            return
    if print_output:
        print(f"Symbol '{name}' not found.")   
    return  

def get_numerator(function):
    num, den = sp.fraction(sp.simplify(function))
    return num

def get_denominator(function):
    num, den = sp.fraction(sp.simplify(function))
    return den

def roots(polynomium):
    s, t = define_st()
    roots = sp.roots(polynomium, s)
    roots = list(roots.keys())
    return roots

def L(f):
    s, t = define_st()
    # var = add_symbol(var)
    return sp.laplace_transform(f, t, s, noconds=True)

def invL(F):
    s, t = define_st()
    # var = add_symbol(var)
    return sp.inverse_laplace_transform(F, s, t)

def delay_function(delay_time):
    s, t = define_st()
    return sp.exp(-t*s).subs(t,delay_time)

def impulse_function(delay_time):
    delay = delay_function(delay_time)
    impulse = 1*delay
    return impulse

def step_function(delay_time):
    s, t = define_st()
    delay = delay_function(delay_time)
    step = 1/s*delay
    return step

def ramp_function(delay_time):
    s, t = define_st()
    delay = delay_function(delay_time)
    ramp = 1/s**2*delay
    return ramp

def lambdify(y, t_range=(0, 10), num_points = 1000):
    s, t = define_st()
    t_vals = np.linspace(t_range[0], t_range[1], num_points)
    y_func = sp.lambdify(t, y, modules=['numpy'])
    y_vals = y_func(t_vals)
    return t_vals, y_vals

def string_input(func: str):
    if 'exp' in func and 'sp.exp' not in func:
        func = func.replace('exp', 'sp.exp')
    if 'sin' in func and 'sp.sin' not in func:
        func = func.replace('sin', 'sp.sin')
    if 'cos' in func and 'sp.cos' not in func:
        func = func.replace('cos', 'sp.cos')
    if 'sqrt' in func and 'sp.sqrt' not in func:
        func = func.replace('sqrt', 'sp.sqrt')
    return func

def tf_from_string(tf_str):
    return sp.simplify(string_input(tf_str))

def tf_from_coefs(num_coefs,den_coefs):
    s, t = define_st()

    num_coefs = [to_sympy_expr(c) for c in num_coefs]
    den_coefs = [to_sympy_expr(c) for c in den_coefs]

    # Build numerator and denominator polynomials from coefficient lists
    num = sum(coef * s**i for i, coef in enumerate(reversed(num_coefs)))
    den = sum(coef * s**i for i, coef in enumerate(reversed(den_coefs)))
    
    # Create the transfer function (symbolically)
    return sp.simplify(num / den)

def to_sympy_expr(x):
        if isinstance(x, str):
            return sp.sympify(x)
        return sp.sympify(x)

