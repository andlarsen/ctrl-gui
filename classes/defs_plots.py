import sympy as sp
import numpy as np
import classes.defs_sympy as defs_sympy
import classes.defs_tf as defs_tf
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

def impulse(*tf_numerics, t_range=(0, 10), n_points=1000, delay_times=None, labels=None):
    if not tf_numerics:
        raise ValueError("At least one transfer function must be provided.")
    
    if labels is None:
        labels = [f"TF {i+1}" for i in range(len(tf_numerics))]
    elif len(labels) != len(tf_numerics):
        raise ValueError("Number of labels must match number of transfer functions.")
    
    if delay_times is None:
        delay_times = [0] * len(tf_numerics)
    elif len(delay_times) != len(tf_numerics):
        raise ValueError("Number of delay times must match number of transfer functions.")
    
    plt.figure(figsize=(8, 5))
    
    for tf, label, delay_time in zip(tf_numerics, labels, delay_times):
        if tf is None:
            raise ValueError(f"Laplace-domain function for '{label}' is not defined.")
        
        step = defs_sympy.impulse_function(delay_time)
        yt = defs_sympy.invL(tf * step)
        new_t_range = (t_range[0]-delay_time,t_range[1]-delay_time)
        t_vals, yt_vals = defs_sympy.lambdify(yt, new_t_range, n_points)
        plt.plot(t_vals+delay_time, yt_vals, label=f"{label}")
    
    plt.title('Impulse Response')
    plt.xlim(np.min(t_range),np.max(t_range))
    plt.xlabel('Time (s)')
    plt.ylabel('Response y(t)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def step(*tf_numerics, t_range=(0, 10), n_points=1000, delay_times=None, labels=None):
    if not tf_numerics:
        raise ValueError("At least one transfer function must be provided.")
    
    if labels is None:
        labels = [f"TF {i+1}" for i in range(len(tf_numerics))]
    elif len(labels) != len(tf_numerics):
        raise ValueError("Number of labels must match number of transfer functions.")
    
    if delay_times is None:
        delay_times = [1] * len(tf_numerics)
    elif len(delay_times) != len(tf_numerics):
        raise ValueError("Number of delay times must match number of transfer functions.")
    
    plt.figure(figsize=(8, 5))
    
    for tf, label, delay_time in zip(tf_numerics, labels, delay_times):
        if tf is None:
            raise ValueError(f"Laplace-domain function for '{label}' is not defined.")
        
        step = defs_sympy.step_function(delay_time)
        yt = defs_sympy.invL(tf * step)
        new_t_range = (t_range[0]-delay_time,t_range[1]-delay_time)
        t_vals, yt_vals = defs_sympy.lambdify(yt, new_t_range, n_points)
        plt.plot(t_vals+delay_time, yt_vals, label=f"{label}")
    
    plt.title('Step Response')
    plt.xlim(np.min(t_range),np.max(t_range))
    plt.xlabel('Time (s)')
    plt.ylabel('Response y(t)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def ramp(*tf_numerics, t_range=(0, 10), n_points=1000, delay_times=None, labels=None):
    if not tf_numerics:
        raise ValueError("At least one transfer function must be provided.")
    
    if labels is None:
        labels = [f"TF {i+1}" for i in range(len(tf_numerics))]
    elif len(labels) != len(tf_numerics):
        raise ValueError("Number of labels must match number of transfer functions.")
    
    if delay_times is None:
        delay_times = [1] * len(tf_numerics)
    elif len(delay_times) != len(tf_numerics):
        raise ValueError("Number of delay times must match number of transfer functions.")
    
    plt.figure(figsize=(8, 5))
    
    for tf, label, delay_time in zip(tf_numerics, labels, delay_times):
        if tf is None:
            raise ValueError(f"Laplace-domain function for '{label}' is not defined.")
        
        step = defs_sympy.ramp_function(delay_time)
        yt = defs_sympy.invL(tf * step)
        new_t_range = (t_range[0]-delay_time,t_range[1]-delay_time)
        t_vals, yt_vals = defs_sympy.lambdify(yt, new_t_range, n_points)
        plt.plot(t_vals+delay_time, yt_vals, label=f"{label}")
    
    plt.title('Ramp Response')
    plt.xlim(np.min(t_range),np.max(t_range))
    plt.xlabel('Time (s)')
    plt.ylabel('Response y(t)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def bode(*tf_numerics, w_range=(), n_points=10000, labels=None):    
    if not tf_numerics:
        raise ValueError("At least one transfer function must be provided.")
    
    if labels is None:
        labels = [f"TF {i+1}" for i in range(len(tf_numerics))]
        
    plt.figure(figsize=(10, 8))
    
    # Magnitude subplot
    plt.subplot(2, 1, 1)
    iter = 1
    for tf, label in zip(tf_numerics, labels):
        w_vals, F_vals = defs_tf.get_frequency_response(tf, w_range=w_range, n_points=n_points)
        print(f"w_range: {w_range}")
        magnitude = 20 * np.log10(np.abs(F_vals))
        plt.semilogx(w_vals, magnitude, label=label)
        if iter == 1:
            w_max = np.max(w_vals)
            w_min = np.min(w_vals)
            print(w_min,w_max)
        else:
            w_max = max(w_max, np.max(w_vals))
            w_min = min(w_min, np.min(w_vals))
            print(w_min,w_max)
        iter += 1
        

    
    plt.title('Bode Plot')
    plt.xlim(w_min,w_max)
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    
    # Phase subplot
    plt.subplot(2, 1, 2)
    iter = 1
    for tf, label in zip(tf_numerics, labels):
        w_vals, F_vals = defs_tf.get_frequency_response(tf, w_range=w_range, n_points=n_points)
        phase = np.unwrap(np.angle(F_vals)) * (180/np.pi)
        plt.semilogx(w_vals, phase, label=label)
        if iter == 1:
            w_max = np.max(w_vals)
            w_min = np.min(w_vals)
        else:
            w_max = max(w_max, np.max(w_vals))
            w_min = min(w_min, np.min(w_vals))
        iter += 1
    
    plt.xlim(w_min,w_max)
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def margin_plot(tf_numeric, w_range=(), n_points=10000):
    w_vals, F_vals = defs_tf.get_frequency_response(tf_numeric, w_range=w_range, n_points=n_points)
    magnitude = 20 * np.log10(np.abs(F_vals))
    phase = np.unwrap(np.angle(F_vals)) * (180/np.pi)
    gm, pm, wcp, wcg, wcp_found, wcg_found = defs_tf.get_margin(tf_numeric, w_range=w_range, n_points=n_points)
    w_min = np.min(w_vals)
    w_max = np.max(w_vals)
    mag_max = np.ceil(np.max(magnitude) / 20) * 20
    mag_min = np.floor(np.min(magnitude) / 20) * 20
    phase_max = np.ceil(np.max(phase) / 90) * 90
    phase_min = np.floor(np.min(phase) / 90) * 90
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.semilogx(w_vals, magnitude)
    plt.title('Bode Plot with Margins')
    plt.ylabel('Magnitude (dB)')
    plt.xlim(w_min,w_max)
    plt.yticks(np.arange(mag_min, mag_max, 20))
    plt.grid(True)
    if wcp is not None:
        plt.axvline(wcp, color='red', linestyle='--')
        plt.axhline(-gm, color='red', linestyle='--')
        plt.text(wcp, gm, f' GM: {gm:.2f} dB at {wcp:.2f} rad/s', color='red', verticalalignment='bottom')
    if wcg is not None:
        plt.axvline(wcg, color='green', linestyle='--')
    plt.subplot(2, 1, 2)
    plt.semilogx(w_vals, phase)
    plt.xlabel('Frequency (rad/s)')
    plt.xlim(w_min,w_max)
    plt.ylabel('Phase (degrees)')
    plt.yticks(np.arange(phase_min, phase_max, 45))
    plt.grid(True)
    if wcp is not None:
        plt.axvline(wcp, color='red', linestyle='--')
    if wcg is not None:
        plt.axvline(wcg, color='green', linestyle='--')
        plt.axhline(-180 + pm, color='green', linestyle='--')
        plt.text(wcg, -180 + pm, f' PM: {pm:.2f}° at {wcg:.2f} rad/s', color='green', verticalalignment='bottom')
    plt.tight_layout()
    plt.show()

def pzmap(tf_numeric, x_range=(), y_range=(), n_points=1000):
    zeros = defs_tf.get_zeros(tf_numeric)
    poles = defs_tf.get_poles(tf_numeric)

    if x_range == ():
        all_points = []
        all_points += [(0.0, 0.0)]
        if poles is not None:
            all_points += [p.evalf().as_real_imag() for p in poles]
        if zeros is not None:
            all_points += [z.evalf().as_real_imag() for z in zeros]
        if all_points:
            real_parts = [pt[0] for pt in all_points]
            imag_parts = [pt[1] for pt in all_points]

            real_min = float(min(real_parts))
            real_max = float(max(real_parts))
            imag_max = float(max(np.abs(imag_parts)))

            if real_min < 0:
                x_min = -10**(np.ceil(np.log10(abs(real_min))))
            else:
                x_min = 10**(np.floor(np.log10(abs(real_min)))) if real_min != 0 else -1
                                
            if real_max < 0:
                x_max = -10**(np.floor(np.log10(abs(real_max))))
            else:
                x_max = 10**(np.ceil(np.log10(abs(real_max)))) if real_max != 0 else 0.1

            y_max = 10**(np.ceil(np.log10(abs(imag_max)))) * (1 if imag_max > 0 else -1) if imag_max != 0 else 1

            x_range = (x_min, x_max)
            y_range = (-y_max, y_max)

    plt.figure()
    if zeros:
        plt.scatter([z.evalf().as_real_imag()[0] for z in zeros],
                    [z.evalf().as_real_imag()[1] for z in zeros],
                    marker='o', color='blue', label='Zeros')
    if poles:
        plt.scatter([p.evalf().as_real_imag()[0] for p in poles],
                    [p.evalf().as_real_imag()[1] for p in poles],
                    marker='x', color='red', label='Poles')
    zeta_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for zeta in zeta_vals:
        if zeta >= 1:
            continue
        theta = np.arccos(zeta)
        if np.tan(theta) != 0:
            slope = np.tan(theta)
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            y_vals_pos = slope * x_vals
            y_vals_neg = -slope * x_vals
            plt.plot(x_vals, y_vals_pos, linestyle='--', color='gray', linewidth=0.5)
            plt.plot(x_vals, y_vals_neg, linestyle='--', color='gray', linewidth=0.5)
            if zeta < 0.8:
                y_text = y_range[1] * 0.8
                x_text = y_text / -slope
                plt.text(x_text, y_text, f'ζ={zeta}', color='gray', horizontalalignment='left')
            else:
                x_text = x_range[0] * 0.8
                y_text = -slope * x_text
                plt.text(x_text, y_text, f'ζ={zeta}', color='gray', horizontalalignment='left')
    wn_vals = np.linspace(0, x_range[0], 5)
    for wn in wn_vals:
        circle = plt.Circle((0, 0), wn, color='gray', fill=False, linestyle='--', linewidth=0.5)
        plt.gca().add_artist(circle)
        plt.text(wn, 0, f'ωn={wn}', color='gray', horizontalalignment='left')
    plt.axvline(0, color='black', lw=0.5, linestyle='-')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.title('Pole-Zero Map')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.show()

def nyquist(tf_numeric, w_range=(), n_points=10000):
    w_vals, F_vals = defs_tf.get_frequency_response(tf_numeric, w_range=w_range, n_points=n_points)
    
    plt.figure()
    plt.plot(F_vals.real, F_vals.imag, label='Nyquist Plot')
    plt.plot(F_vals.real, -F_vals.imag, linestyle='--', color='gray')
    plt.title('Nyquist Plot')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid(True)
    plt.show()

