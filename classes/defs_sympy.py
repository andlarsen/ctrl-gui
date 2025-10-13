import sympy as sp
import numpy as np
import re
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

def add_function(name: str):
    s,t = define_st()
    ft = sp.Function(name)(t)
    Fs = sp.Function(capitalize_first(name))(s)
    return ft, Fs

def remove_function():
    pass

def get_numerator(tf):
    num, den = sp.fraction(sp.simplify(tf))
    return num

def get_denominator(tf):
    num, den = sp.fraction(sp.simplify(tf))
    return den

def get_equation(tf,input_symbol,output_symbol,constants):
    s, t = define_st()

    # Load inputs and outputs
    u = input_symbol[0]
    y = output_symbol[0]
    U = input_symbol[1]
    Y = output_symbol[1]

    # Load constants
    sympy_symbols = {}
    for name, const_data in constants.items():
        symbol_value = const_data["symbol"]
        sympy_symbols[name] = symbol_value
    locals().update(sympy_symbols)

    G_num, G_den = sp.fraction(tf)
    RHS_L = sp.expand(G_num * U)
    LHS_L = sp.expand(G_den * Y)

    # U = sp.Function('U')
    # Y = sp.Function('Y')
    subs_dict_corrected = {
        s**10*Y: sp.Derivative(y, (t, 10)),
        s**9*Y: sp.Derivative(y, (t, 9)),
        s**8*Y: sp.Derivative(y, (t, 8)),
        s**7*Y: sp.Derivative(y, (t, 7)),
        s**6*Y: sp.Derivative(y, (t, 6)),
        s**5*Y: sp.Derivative(y, (t, 5)),
        s**4*Y: sp.Derivative(y, (t, 4)),
        s**3*Y: sp.Derivative(y, (t, 3)),
        s**2*Y: sp.Derivative(y, (t, 2)),
        s*Y: sp.Derivative(y, t),
        Y: y
    }

    print(LHS_L)
    lhs = LHS_L.subs(subs_dict_corrected)
    print(lhs)
    subs_dict_corrected = {
        s**2*U: sp.Derivative(u, (t, 2)),
        s*U: sp.Derivative(u, t),
        U: u
    }
    rhs = RHS_L.subs(subs_dict_corrected)

    return lhs, rhs

def roots(polynomium):
    s, t = define_st()
    roots = sp.roots(polynomium, s)
    roots = list(roots.keys())
    return roots

def L(ft):
    s, t = define_st()
    return sp.laplace_transform(ft, t, s, noconds=True)

def invL(Fs):
    s, t = define_st()
    return sp.inverse_laplace_transform(Fs, s, t)

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

def tf_from_string(tf_str,constants):
    s, t = define_st()
    symbols = {'s': s, 't': t}
    for name, const_data in constants.items():
        symbols[name] = sp.Symbol(name)
    locals().update(symbols)

    tf = eval(translate_string(tf_str))

    return sp.simplify(tf)

def tf_from_coefs(num_coefs,den_coefs,constants):
    s, t = define_st() 
    symbols = {'s': s, 't': t}
    for name, const_data in constants.items():
        symbols[name] = sp.Symbol(name) 
    locals().update(symbols)
    
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

    tf = sp.simplify(num / den)

    return tf

def to_sympy_expr(x):
        if isinstance(x, str):
            return sp.sympify(x)
        return sp.sympify(x)

def integrate(function):
    s, t = define_st()
    return sp.integrate(function,t)

def translate_string(string_input: str):
    # Insert functions as sympy
    if 'exp' in string_input and 'sp.exp' not in string_input:
        string_input = string_input.replace('exp', 'sp.exp')
    if 'sin' in string_input and 'sp.sin' not in string_input:
        string_input = string_input.replace('sin', 'sp.sin')
    if 'cos' in string_input and 'sp.cos' not in string_input:
        string_input = string_input.replace('cos', 'sp.cos')
    if 'sqrt' in string_input and 'sp.sqrt' not in string_input:
        string_input = string_input.replace('sqrt', 'sp.sqrt')
    # Replace dot() with diff(var, t, x) where x is the number of derivatives. Number of derivatives is counted from number of d's in string_input
    if 'dot(' in string_input:
        matches = re.findall(r'(d+)ot\(([^)]+)\)', string_input)
        for d_string, var in matches:
            derivative_order = len(d_string) 
            pattern = f'{d_string}ot\\({re.escape(var)}\\)'
            replacement = f'sp.diff({var}, t, {derivative_order})'
            string_input = re.sub(pattern, replacement, string_input)
    return string_input

def find_n_derivatives(expr: str):
    matches = re.findall(r'd+ot\(', expr)
    orders = [m.count('d') for m in matches]
    highest_order = max(orders) if orders else 0
    return highest_order

def generate_ic_symbols(func_name, order):
    if order == 0:
        return f"{func_name}0"
    else:
        for i in range(order):
            func_name = f"{func_name}d"
        return f"{func_name}ot0"
    
def make_ic_subs(expr: str, func, var, ic_values_in=[]):
    max_order = find_n_derivatives(expr)
    if ic_values_in == []:
        ic_values_in = np.zeros(max_order+1)
    func_name = str(func.func)
    ic_subs, ic_symbols, ic_values = {}, {}, {}
    for n in range(0,max_order + 1):
        name = generate_ic_symbols(func_name, n)
        sym = sp.Symbol(name)
        ic_symbols[name] = sym
        ic_subs[sp.diff(func, var, n).subs(var, 0)] = sym
        ic_values[sym] = ic_values_in[n]
    return ic_subs, ic_symbols, ic_values

def capitalize_first(string_input: str) -> str:
    if len(string_input) == 0:
        return string_input
    return string_input[0].upper() + string_input[1:]

def define_constant_symbols(constants):
    return [sp.Symbol(name) for name, data in constants.items()]

def define_constant_values(constants):
    return {sp.Symbol(name): data["value"] for name, data in constants.items()}

def tf_from_equation(lhs,rhs,input_symbol,output_symbol,constants):
    s, t = define_st()
    symbols = {'s': s, 't': t}
    for name, const_data in constants.items():
        symbols[name] = sp.Symbol(name)
    locals().update(symbols)

    # Load inputs and outputs
    u = input_symbol[0]
    y = output_symbol[0]
    U = input_symbol[1]
    Y = output_symbol[1]
    
    # Create initial condition symbols and values
    subs_ic = {}
    ics = {}
    u_subs, u_symbols, u_ics = make_ic_subs(rhs,u,t)
    y_subs, y_symbols, y_ics = make_ic_subs(lhs,y,t)
    subs_ic = {**u_subs, **y_subs}
    ics = {**u_ics, **y_ics}

    # Translate the input strings (lhs and rhs)
    lhs = eval(translate_string(lhs))
    rhs = eval(translate_string(rhs))

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
    tf = sp.simplify(Y_s / U)
    return tf