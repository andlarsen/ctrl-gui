import sympy as sp
import numpy as np
import classes.defs_sympy as defs_sympy
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

def get_poles(tf):
    denominator = get_denominator(tf)
    poles = defs_sympy.roots(denominator)
    return poles

def get_zeros(tf):
    numerator = get_numerator(tf)
    zeros = defs_sympy.roots(numerator)
    return zeros

def get_system_order():
    pass

def get_free_zeros():
    pass

def get_free_poles():
    pass

def get_numerator(tf):
    numerator, denominator = sp.fraction(tf)
    return numerator

def get_denominator(tf):
    numerator, denominator = sp.fraction(tf)
    return denominator

def get_equation(tf,input_symbol,output_symbol):
    s, t = defs_sympy.define_st()

    u = input_symbol[0]
    y = output_symbol[0]
    U = input_symbol[1]
    Y = output_symbol[1]

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

def get_w_range(zeros,poles):
    all_freqs = []
    if poles is not None:
        all_freqs += [abs(p.evalf().as_real_imag()[0]) for p in poles]
    if zeros is not None:
        all_freqs += [abs(z.evalf().as_real_imag()[0]) for z in zeros]
    if all_freqs:
        min_freq, max_freq = float(min(all_freqs)), float(max(all_freqs))
        min_freq = 10**(np.floor(np.log10(min_freq)))/10 if min_freq > 0 else 0.01
        max_freq = 10**(np.ceil(np.log10(max_freq)))*10 if max_freq > 0 else 10
        w_range = (min_freq, max_freq)
        return w_range
    else:
        w_range = (0.1, 100)
        return w_range
    
def get_margin(tf_numeric, w_range=(), n_points=10000):
    w_vals, F_vals = get_frequency_response(tf_numeric, w_range=w_range, n_points=n_points)

    magnitude = 20 * np.log10(np.abs(F_vals))
    phase = np.unwrap(np.angle(F_vals)) * (180/np.pi)

    gain_crossings = np.where(np.diff(np.sign(magnitude)))[0]
    phase_crossings = np.where(np.diff(np.sign(phase + 180)))[0]

    if gain_crossings.size > 0:
        wcg_found = True
        wcg = w_vals[gain_crossings[0]]
        pm = 180 + phase[gain_crossings[0]]
    else:
        wcg_found = False
        wcg = None
        pm = None

    if phase_crossings.size > 0:
        wcp_found = True
        wcp = w_vals[phase_crossings[0]]
        gm = -magnitude[phase_crossings[0]]
    else:
        wcp_found = False
        wcp = None
        gm = None
    return gm, pm, wcg, wcp, wcg_found, wcp_found
    
def get_frequency_response(tf_numeric, w_range=(), n_points=10000):
    if tf_numeric is None:
        raise ValueError("Laplace-domain function F(s) is not defined.")
    s, t = defs_sympy.define_st()
    zeros = get_zeros(tf_numeric)
    poles = get_poles(tf_numeric)
    w_range = get_w_range(zeros,poles)
    w_min = 10**(np.floor(np.log10(w_range[0])))
    w_max = 10**(np.ceil(np.log10(w_range[1])))
    w_vals = np.logspace(np.log10(w_min), np.log10(w_max), n_points)
    s_vals = 1j * w_vals
    F_func = sp.lambdify(s, tf_numeric, modules=['numpy'])
    F_vals = F_func(s_vals)
    return w_vals, F_vals


## Define transfer function
def from_string(tf_str,constants):
    s, t = defs_sympy.define_st()
    symbols = {'s': s, 't': t}
    for name, const_data in constants.items():
        symbols[name] = sp.Symbol(name) 
    locals().update(symbols)
    tf = eval(defs_sympy.translate_string(tf_str))
    return sp.simplify(tf)

def from_coefs(num_coefs,den_coefs,constants):
    s, t = defs_sympy.define_st() 
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

    tf = num / den

    return sp.simplify(tf)

def from_equation(lhs,rhs,input_symbol,output_symbol,constants):
    s, t = defs_sympy.define_st()
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
    u_subs, u_symbols, u_ics = defs_sympy.make_ic_subs(rhs,u,t)
    y_subs, y_symbols, y_ics = defs_sympy.make_ic_subs(lhs,y,t)
    subs_ic = {**u_subs, **y_subs}
    ics = {**u_ics, **y_ics}

    # Translate the input strings (lhs and rhs)
    lhs = eval(defs_sympy.translate_string(lhs))
    rhs = eval(defs_sympy.translate_string(rhs))

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