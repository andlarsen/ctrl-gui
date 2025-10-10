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

def L(f):
    s, t = define_st()
    return sp.laplace_transform(f, t, s, noconds=True)

def invL(F):
    s, t = define_st()
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
