import sympy as sp
import numpy as np
import scipy
import scipy.signal as signal
import itertools
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import classes.defs_sympy as defs_sympy

from typing import Dict, List, Any, Tuple, Optional, Union
from models.model_response import Response, ImpulseResponseInfo, StepResponseInfo, RampResponseInfo
from models.model_transfer_function import TransferFunctionModel
from models.model_polynomium import Polynomium
from models.model_coefficients import Coefficients
from models.model_metric import Metric

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

def get_coefficients(polynomium) -> list:
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
    
def get_margin(tf_numeric, w_range=(), n_points=500):
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
    
def get_frequency_response(tf_numeric, w_range=(0.1,100), n_points=500):
    if tf_numeric is None:
        raise ValueError("Laplace-domain function F(s) is not defined.")
    
    s, t = defs_sympy.define_st()
    zeros = get_zeros(tf_numeric)
    poles = get_poles(tf_numeric)
    if w_range is None or not w_range:
        try:
            w_range = get_w_range(zeros, poles)
        except Exception:
            w_range = (0.1, 10)
        w_min = 10**(np.floor(np.log10(w_range[0])))
        w_max = 10**(np.ceil(np.log10(w_range[1]))+1)
    else:
        w_min = 10**(np.floor(np.log10(w_range[0])))
        w_max = 10**(np.ceil(np.log10(w_range[1])))
    w_vals = np.logspace(np.log10(w_min), np.log10(w_max), n_points)
    if tf_numeric == 1:
        F_vals = np.ones_like(w_vals, dtype=complex)
        return w_vals, F_vals
    s_vals = 1j * w_vals
    F_func = sp.lambdify(s, tf_numeric, modules=['numpy'])
    F_vals = F_func(s_vals)
    return w_vals, F_vals

## Define transfer function
def define_signal(numerator_coefficients: sp.Poly=None, denominator_coefficients: sp.Poly=None) -> signal.TransferFunction:
    system = signal.TransferFunction(numerator_coefficients, denominator_coefficients)
    return system

def from_string(tf_str,constants):
    s, t = defs_sympy.define_st()
    symbols = {'s': s, 't': t}
    for name, const_data in constants.items():
        symbols[name] = sp.Symbol(name) 
    locals().update(symbols)
    tf = sp.together(eval(defs_sympy.translate_string(tf_str)))
    return tf

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

    tf = sp.together(num / den)

    return tf

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
    tf = sp.together(Y_s / U)
    return tf

### Time responses

def auto_scale_t_range(
        tf:         TransferFunctionModel, 
        t_range:    Optional[tuple] = (), 
        delay_time: float = 0, 
        n_tau:      float = 8, 
        tol:        float = 1e-6) -> tuple[float,float]:
    
    poles = tf.signal.poles
    
    t_start = t_range[0] if t_range else 0.0
    t_end = t_range[1] if t_range else None

    if t_end is None or t_end <= t_start:
            
        # Initialize T_sim
        T_sim = 10 

        # Find the dominant pole for stable systems
        if len(poles) > 0:
            # Check stability: any pole with Re(s) >= 0 (excluding numerical noise)
            unstable_or_marginal = np.any(np.real(poles) >= -tol)

            if not unstable_or_marginal:
                # System is stable. Estimate settling time.
                
                # Characteristic Time Constant (tau) is 1 / |Re(p_dominant)|
                # The dominant pole is the one closest to the jw-axis (smallest |Re(p)|)
                stable_poles = poles[np.real(poles) < -tol]
                if len(stable_poles) > 0:
                    
                    # Find the stable pole with the smallest magnitude of its real part
                    abs_real_parts = np.abs(np.real(stable_poles))
                    tau_max = 1.0 / np.min(abs_real_parts)
                    
                    # Simulation time T_sim is based on n_tau * tau_max
                    T_settle_factor = n_tau # Default set to 6 for impulse
                    T_sim = T_settle_factor * tau_max
            
            # Add delay time to the required simulation time
            t_end = T_sim + delay_time
            
        else:
            # If system has no poles (e.g., pure gain), use a default
            t_end = T_sim + delay_time

    t_range = (float(t_start),float(t_end))

    return t_range

def get_time_response(
        tf: TransferFunctionModel, 
        response_type: str = 'impulse',  # 'impulse', 'step', or 'ramp'
        t_range: Optional[tuple] = (), 
        delay_time: float = 0, 
        n_points: int = 500, 
        n_tau: float = 6, 
        tol: float = 1e-6) -> Response:  # Returns unified Response model
    
    if not t_range:
        t_range = auto_scale_t_range(tf=tf, t_range=t_range, delay_time=delay_time, n_tau=n_tau, tol=tol)
    
    t_start = t_range[0]
    t_end = t_range[1]
    t_end = max(t_end, delay_time + 1e-3)
    
    t_vals = np.linspace(t_start, t_end, n_points)
    mask_after = t_vals >= delay_time
    y_vals = np.zeros_like(t_vals)
    r_vals = np.zeros_like(t_vals)
    
    if np.any(mask_after):
        t_shifted = t_vals[mask_after] - delay_time
        
        # Choose the appropriate response function and define input signal
        if response_type == 'impulse':
            _, y_after = signal.impulse(tf.signal, T=t_shifted)
            # Impulse approximation
            if len(t_shifted) > 0:
                r_vals[mask_after][0] = 1.0 / (t_shifted[1] - t_shifted[0]) if len(t_shifted) > 1 else 1.0
                
        elif response_type == 'step':
            _, y_after = signal.step(tf.signal, T=t_shifted)
            r_vals[mask_after] = 1.0
            
        elif response_type == 'ramp':
            num = tf.signal.num
            den = np.concatenate([tf.signal.den, [0]])
            ramp_tf = signal.TransferFunction(num, den)
            _, y_after = signal.step(ramp_tf, T=t_shifted)
            r_vals[mask_after] = t_shifted
            
        else:
            raise ValueError(f"Unknown response_type: {response_type}. Choose 'impulse', 'step', or 'ramp'")
        
        y_vals[mask_after] = y_after
    
    t_range = (t_vals[0], t_vals[-1])
    
    # Get appropriate info based on response type
    if response_type == 'impulse':
        info = get_impulse_response_info(t_vals, y_vals, r_vals, delay_time, tol)
    elif response_type == 'step':
        info = get_step_response_info(t_vals, y_vals, r_vals, delay_time, tol)
    elif response_type == 'ramp':
        info = get_ramp_response_info(t_vals, y_vals, r_vals, delay_time, tol)
    
    return Response(  # Single unified return type
        t_vals=t_vals.tolist(), 
        y_vals=y_vals.tolist(),
        r_vals=r_vals.tolist(),
        t_range=t_range,
        delay_time=delay_time,
        response_type=response_type,
        info=info
    )

def get_impulse_response_info(
        t_vals: list[float] = (), 
        y_vals: list[float] = (), 
        r_vals: list[float] = (),  # Added but not used for impulse
        delay_time: float = 0, 
        tol: float = 0.02) -> ImpulseResponseInfo:
    
    y_vals = np.real_if_close(y_vals)
    
    # Remove values before delay
    mask = t_vals >= delay_time
    t = t_vals[mask]
    y = y_vals[mask]
    
    if len(y) == 0:
        return ImpulseResponseInfo()
    
    # --- Peak characteristics ---
    peak_idx = np.argmax(np.abs(y))
    t_peak = t[peak_idx] - delay_time
    y_peak = y[peak_idx]
    
    # --- Integral (DC gain) ---
    integral = np.trapz(y, t)
    
    # --- Energy ---
    energy = np.trapz(y**2, t)
    
    # --- Settling time ---
    y_final = y[-1]  # Should approach 0 for stable system
    threshold = tol * np.abs(y_peak)
    settled = np.abs(y) < threshold
    
    if np.any(settled):
        # Find last time it exceeds threshold
        last_outside = np.where(~settled)[0]
        if len(last_outside) > 0 and last_outside[-1] < len(t) - 1:
            t_settling = t[last_outside[-1] + 1] - delay_time
        else:
            t_settling = np.nan
    else:
        t_settling = np.nan
    
    # --- Time to half peak ---
    half_peak = y_peak / 2

    try:
        after_peak = y[peak_idx:]
        t_after_peak = t[peak_idx:]
        half_idx = np.where(np.abs(after_peak) <= np.abs(half_peak))[0][0]
        t_half = t_after_peak[half_idx] - delay_time
    except (IndexError, ValueError):
        t_half = np.nan
    
    # --- Decay rate (exponential fit after peak) ---
    try:
        if peak_idx < len(y) - 10:  # Need some points after peak
            y_decay = np.abs(y[peak_idx:])
            t_decay = t[peak_idx:] - t[peak_idx]
            
            # Fit exponential: y = y_peak * exp(-decay_rate * t)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_y = np.log(y_decay / y_decay[0])
                valid = np.isfinite(log_y)
                if np.sum(valid) > 5:
                    # Linear fit in log space
                    coeffs = np.polyfit(t_decay[valid], log_y[valid], 1)
                    decay_rate = -coeffs[0]  # Negative of slope
                else:
                    decay_rate = np.nan
        else:
            decay_rate = np.nan
    except:
        decay_rate = np.nan
    
    # --- Count oscillations (zero crossings) ---
    zero_crossings = np.where(np.diff(np.sign(y)))[0]
    num_oscillations = len(zero_crossings)
    
    # --- Estimate damping ratio and natural frequency (if oscillatory) ---
    if num_oscillations >= 2:
        # Use logarithmic decrement method
        peaks_idx = []
        for i in range(1, len(y)-1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                peaks_idx.append(i)
        
        if len(peaks_idx) >= 2:
            # Period from two consecutive peaks
            T_d = (t[peaks_idx[1]] - t[peaks_idx[0]])
            omega_d = 2 * np.pi / T_d  # Damped natural frequency
            
            # Logarithmic decrement
            delta = np.log(np.abs(y[peaks_idx[0]]) / np.abs(y[peaks_idx[1]]))
            damping_ratio = delta / np.sqrt((2*np.pi)**2 + delta**2)
            
            # Undamped natural frequency
            natural_freq = omega_d / np.sqrt(1 - damping_ratio**2) if damping_ratio < 1 else omega_d
        else:
            damping_ratio = None
            natural_freq = None
    else:
        damping_ratio = None
        natural_freq = None

    defaults = ImpulseResponseInfo._field_defaults
    
    calculated_data = {
        "t_peak": t_peak,
        "y_peak": y_peak,
        "t_settling": t_settling,
        "t_half": t_half,
        "decay_rate": decay_rate,
        "integral": integral,
        "energy": energy,
        "num_oscillations": num_oscillations,
        "damping_ratio": damping_ratio,
        "natural_freq": natural_freq,
    }
    
    info_args = {}
    for field_name, value in calculated_data.items():
        default_metric = defaults.get(field_name)
        
        if default_metric:
            info_args[field_name] = default_metric._replace(value=value)
        else:
            info_args[field_name] = Metric(value=value, label=field_name)
    
    return ImpulseResponseInfo(**info_args)

def get_step_response_info(
        t_vals: list[float] = (), 
        y_vals: list[float] = (), 
        r_vals: list[float] = (),
        delay_time: float = 0, 
        tol: float = 0.02) -> StepResponseInfo:
        
    y_vals = np.real_if_close(y_vals)

    # Remove values before delay
    mask = t_vals >= delay_time
    t = t_vals[mask]
    y = y_vals[mask]
    
    if len(y) == 0:
        return StepResponseInfo
    
    t_relative = t - delay_time
    
    # --- Steady-State Value ---
    y_final = y[-1]
    y_initial = y[0] if len(y) > 0 else 0.0
    
    # --- Peak Characteristics ---
    peak_idx = np.argmax(y)
    y_peak = y[peak_idx]
    t_peak = t_relative[peak_idx]
    
    # --- Overshoot ---
    overshoot_percent = None
    if y_final != 0 and y_peak > y_final:
        overshoot_percent = ((y_peak - y_final) / np.abs(y_final)) * 100.0
    
    # --- Rise Time (10% to 90%) ---
    target_10 = y_initial + 0.1 * (y_final - y_initial)
    target_90 = y_initial + 0.9 * (y_final - y_initial)
    
    t_10, t_90 = np.nan, np.nan
    try:
        idx_10 = np.where(y >= target_10)[0][0]
        t_10 = t_relative[idx_10]
    except IndexError:
        pass 
        
    try:
        idx_90 = np.where(y >= target_90)[0][0]
        t_90 = t_relative[idx_90]
    except IndexError:
        pass 
    
    t_rise = t_90 - t_10 if not np.isnan(t_10) and not np.isnan(t_90) and t_90 > t_10 else np.nan
    
    # --- Settling Time ---
    t_settling = np.nan
    threshold_low = y_final * (1 - tol)
    threshold_high = y_final * (1 + tol)
    
    outside_band = (y < threshold_low) | (y > threshold_high)
    
    if np.any(outside_band):
        last_outside_idx = np.where(outside_band)[0][-1]
        
        if last_outside_idx < len(y) - 1:
            t_settling = t_relative[last_outside_idx + 1]
        else:
            t_settling = np.nan
    elif len(y) > 0:
        t_settling = t_relative[0] 

    # --- Number of Oscillations (Crossings of Final Value) ---
    # Look at the difference between y and y_final
    error = y - y_final
    # Zero crossings of the error signal
    zero_crossings = np.where(np.diff(np.sign(error)))[0]
    num_oscillations = len(zero_crossings)
    
    # --- Damping Ratio ($\zeta$) and Natural Frequency ($\omega_n$) Estimation ---
    damping_ratio, natural_freq = None, None
    
    if overshoot_percent is not None and overshoot_percent > 0 and y_final != 0:
        try:
            M_p = overshoot_percent / 100.0
            ln_M_p = np.log(M_p)
            damping_ratio = -ln_M_p / np.sqrt(np.pi**2 + ln_M_p**2)
            
            if t_peak > 0 and damping_ratio < 1:
                natural_freq = np.pi / (t_peak * np.sqrt(1 - damping_ratio**2))
        except (ValueError, ZeroDivisionError, TypeError):
            pass 

    defaults = StepResponseInfo._field_defaults
    
    calculated_data = {
        "y_final": y_final,
        "y_initial": y_initial,
        "t_rise": t_rise,
        "t_peak": t_peak,
        "y_peak": y_peak,
        "overshoot_percent": overshoot_percent,
        "t_settling": t_settling,
        "num_oscillations": num_oscillations,
        "damping_ratio": damping_ratio,
        "natural_freq": natural_freq,
    }
    
    info_args = {}
    for field_name, value in calculated_data.items():
        default_metric = defaults.get(field_name)
        
        if default_metric:
            info_args[field_name] = default_metric._replace(value=value)
        else:
            info_args[field_name] = Metric(value=value, label=field_name)
    return StepResponseInfo(**info_args)

def get_ramp_response_info(
        t_vals: list[float] = (), 
        y_vals: list[float] = (), 
        r_vals: list[float] = (), 
        delay_time: float = 0, 
        tol: float = 0.02) -> RampResponseInfo:
    
    y_vals = np.real_if_close(y_vals)
    
    # --- Data Filtering ---
    mask = t_vals >= delay_time
    t = t_vals[mask]
    r = r_vals[mask]
    y = y_vals[mask]
    
    if len(y) < 20: # Need enough points to determine steady-state behavior
        return RampResponseInfo(
            steady_state_error=Metric(), velocity_error_const=Metric(), t_lag=Metric(), 
            max_tracking_error=Metric(), delay_time=Metric(value=delay_time, label="Time Delay", unit="s"), 
            y_final=Metric(), t_peak=Metric(), y_peak=Metric()
        )
    
    # --- 1. Steady-State Error (ess) ---
    # The error is e(t) = r(t) - y(t)
    error = r - y
    
    # Calculate steady-state error by averaging the error over the last 10% of the simulation
    n_avg = max(10, len(error) // 10)
    e_final = np.mean(error[-n_avg:])
    
    # --- 2. Velocity Error Constant (Kv) ---
    # For a Type 1 stable system: ess = 1 / Kv
    Kv = 1.0 / e_final if e_final != 0 else np.inf
    
    # --- 3. Maximum Tracking Error ---
    e_max_tracking = np.max(np.abs(error))
    
    # --- 4. Time Lag (t_lag) ---
    # If the system tracks the ramp, the error is constant (ess). 
    # This constant error relates to the time lag: ess ≈ t_lag * slope (slope=1 for standard ramp)
    t_lag = e_final 
    
    # --- 5. Peak Characteristics (less common for ramp, but good to check) ---
    peak_idx = np.argmax(y)
    t_peak = t[peak_idx] - delay_time
    y_peak = y[peak_idx]
    
    # --- 6. Final Values ---
    y_final = y[-1]

    defaults = RampResponseInfo._field_defaults
    
    calculated_data = {
        "y_final": y_final,
        "t_peak": t_peak,
        "y_peak": y_peak,
        "e_final": e_final,
        "Kv": Kv,
        "t_lag": t_lag,
        "e_max_tracking": e_max_tracking,
    }
    
    info_args = {}
    for field_name, value in calculated_data.items():
        default_metric = defaults.get(field_name)
        
        if default_metric:
            info_args[field_name] = default_metric._replace(value=value)
        else:
            info_args[field_name] = Metric(value=value, label=field_name)

    return RampResponseInfo(**info_args)

### Sweep functions

def sweep_tfs(self,tf_instances, delay_times=None, sweep_params: Dict[str, List[float]] = None, is_global: bool = False):
    sweep_variables = list(sweep_params.keys())
    sweep_value_lists = list(sweep_params.values())
    
    if is_global == True:
        original_values = {var: self.global_constants[var]['value'] for var in sweep_variables if var in self.global_constants}
    else:
        original_values = {var: self.constants[var]['value'] for var in sweep_variables if var in self.constants}

    tf_numerics_list = []
    labels_list = []
    delay_times_list = []

    for combo_values in itertools.product(*sweep_value_lists):
        combo_label_parts = []
        
        for var_name, value in zip(sweep_variables, combo_values):
            if is_global == True:
                self.edit_global_constant(var_name, value=value) 
            else:
                self.edit_constant(var_name, value=value) 
            combo_label_parts.append(f"{var_name}={value}")

        base_label = ", ".join(combo_label_parts)

        try:
            for i, tf_instance in enumerate(tf_instances):
                self.update()
                tf_numeric = tf_instance.tf.numeric 
                
                tf_numerics_list.append(tf_numeric)
                if delay_times is not None:
                    delay_times_list.append(delay_times[i])
                labels_list.append(f"{tf_instance.Name} ({base_label})")
        except:
            self.update()
            tf_numeric = self.tf.numeric
            tf_numerics_list.append(tf_numeric)
            if delay_times is not None:
                delay_times_list.append(delay_times)
            labels_list.append(f"{self.Name} ({base_label})")
    
        if is_global == True:
            for var_name, value in original_values.items():
                self.edit_global_constant(var_name, value=value)
        else:
            for var_name, value in original_values.items():
                self.edit_constant(var_name, value=value)
                
    return tf_numerics_list, delay_times_list, labels_list

def sweep_impulse_responses(self,tf_instances, delay_times=None, sweep_params: Dict[str, List[float]] = None, is_global: bool = False):
    sweep_variables = list(sweep_params.keys())
    sweep_value_lists = list(sweep_params.values())
    
    if is_global == True:
        original_values = {var: self.global_constants[var]['value'] for var in sweep_variables if var in self.global_constants}
    else:
        original_values = {var: self.constants[var]['value'] for var in sweep_variables if var in self.constants}

    responses_list = []
    labels_list = []

    for combo_values in itertools.product(*sweep_value_lists):
        combo_label_parts = []
        
        for var_name, value in zip(sweep_variables, combo_values):
            if is_global == True:
                self.edit_global_constant(var_name, value=value) 
            else:
                self.edit_constant(var_name, value=value) 
            combo_label_parts.append(f"{var_name}={value}")

        base_label = ", ".join(combo_label_parts)

        try:
            for i, tf_instance in enumerate(tf_instances):
                self.update()
                # tf_numeric = tf_instance.tf.numeric 
                impulse_response = tf_instance.impulse_response
                
                responses_list.append(impulse_response)
                labels_list.append(f"{tf_instance.Name} ({base_label})")
        except:
            self.update()
            # tf_numeric = self.tf.numeric
            impulse_response = tf_instance.impulse_response
            responses_list.append(impulse_response)
            labels_list.append(f"{self.Name} ({base_label})")
    
        if is_global == True:
            for var_name, value in original_values.items():
                self.edit_global_constant(var_name, value=value)
        else:
            for var_name, value in original_values.items():
                self.edit_constant(var_name, value=value)
                
    return responses_list, labels_list

def sweep_step_responses(self,tf_instances, delay_times=None, sweep_params: Dict[str, List[float]] = None, is_global: bool = False):
    sweep_variables = list(sweep_params.keys())
    sweep_value_lists = list(sweep_params.values())
    
    if is_global == True:
        original_values = {var: self.global_constants[var]['value'] for var in sweep_variables if var in self.global_constants}
    else:
        original_values = {var: self.constants[var]['value'] for var in sweep_variables if var in self.constants}

    responses_list = []
    labels_list = []

    for combo_values in itertools.product(*sweep_value_lists):
        combo_label_parts = []
        
        for var_name, value in zip(sweep_variables, combo_values):
            if is_global == True:
                self.edit_global_constant(var_name, value=value) 
            else:
                self.edit_constant(var_name, value=value) 
            combo_label_parts.append(f"{var_name}={value}")

        base_label = ", ".join(combo_label_parts)

        try:
            for i, tf_instance in enumerate(tf_instances):
                self.update()
                # tf_numeric = tf_instance.tf.numeric 
                step_response = tf_instance.step_response
                
                responses_list.append(step_response)
                labels_list.append(f"{tf_instance.Name} ({base_label})")
        except:
            self.update()
            # tf_numeric = self.tf.numeric
            step_response = tf_instance.step_response
            responses_list.append(step_response)
            labels_list.append(f"{self.Name} ({base_label})")
    
        if is_global == True:
            for var_name, value in original_values.items():
                self.edit_global_constant(var_name, value=value)
        else:
            for var_name, value in original_values.items():
                self.edit_constant(var_name, value=value)
                
    return responses_list, labels_list


def sweep_ramp_responses(self,tf_instances, delay_times=None, sweep_params: Dict[str, List[float]] = None, is_global: bool = False):
    sweep_variables = list(sweep_params.keys())
    sweep_value_lists = list(sweep_params.values())
    
    if is_global == True:
        original_values = {var: self.global_constants[var]['value'] for var in sweep_variables if var in self.global_constants}
    else:
        original_values = {var: self.constants[var]['value'] for var in sweep_variables if var in self.constants}

    responses_list = []
    labels_list = []

    for combo_values in itertools.product(*sweep_value_lists):
        combo_label_parts = []
        
        for var_name, value in zip(sweep_variables, combo_values):
            if is_global == True:
                self.edit_global_constant(var_name, value=value) 
            else:
                self.edit_constant(var_name, value=value) 
            combo_label_parts.append(f"{var_name}={value}")

        base_label = ", ".join(combo_label_parts)

        try:
            for i, tf_instance in enumerate(tf_instances):
                self.update()
                # tf_numeric = tf_instance.tf.numeric 
                step_response = tf_instance.ramp_response
                
                responses_list.append(step_response)
                labels_list.append(f"{tf_instance.Name} ({base_label})")
        except:
            self.update()
            # tf_numeric = self.tf.numeric
            step_response = tf_instance.ramp_response
            responses_list.append(step_response)
            labels_list.append(f"{self.Name} ({base_label})")
    
        if is_global == True:
            for var_name, value in original_values.items():
                self.edit_global_constant(var_name, value=value)
        else:
            for var_name, value in original_values.items():
                self.edit_constant(var_name, value=value)
                
    return responses_list, labels_list