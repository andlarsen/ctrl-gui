import sympy as sp
import numpy as np
import scipy.signal as signal
import utilities.utils_sympy as utils_sympy
import utilities.utils_transfer_function as defs_tf
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from typing import Dict, List, Any, Tuple, Optional, Union


def plot_response(
    *tf_instances, 
    labels: List[str] = None, 
    delay_times: List[float] = None, 
    t_range: Union[Tuple[float,float], None] = None,
    response_type: str = "step"):
    
    if not tf_instances:
        raise ValueError("At least one transfer function instance must be provided.")
    
    responses = []
    for i, tf_instance in enumerate(tf_instances):
        if response_type == "impulse":
            responses.append(tf_instance.impulse_response)
            title_string = "Impulse response"
        elif response_type == "step":
            responses.append(tf_instance.step_response)
            title_string = "Step response"
        elif response_type == "ramp":
            responses.append(tf_instance.ramp_response)
            title_string = "Ramp response"
        else:
            print("Warning: Response type is not recognized")
    
    n_responses = len(responses) 
    if labels is None or len(labels) != n_responses:
        if labels is not None:
             print(f"Warning: Length of 'labels' ({len(labels)}) does not match number of responses ({n_responses}). Generating default labels.")
        labels = [f"Response {i+1}" for i in range(n_responses)]
    
    if delay_times is None:
        delay_times = [0.0] * n_responses
    elif len(delay_times) == 1 and n_responses > 1:
        print(f"Note: Applying single delay time of {delay_times[0]}s to all {n_responses} responses.")
        delay_times = [delay_times[0]] * n_responses
    elif len(delay_times) != n_responses:
        delay_times = [0.0] * n_responses
    
    t_min = 0 
    t_max = float('-inf')
    plt.figure(figsize=(8, 5))
    for response, label, delay_time in zip(responses, labels, delay_times):
        t_vals = np.array(response.t_vals) + delay_time 
        y_vals = np.array(response.y_vals)

        if t_vals[0] > 0:
            t_patch_start = np.array([0.0])
            y_patch_start = np.array([0.0])

            t_vals = np.concatenate((t_patch_start, t_vals))
            y_vals = np.concatenate((y_patch_start, y_vals))

        if t_range is not None and len(t_range) > 1 and t_vals[-1] < t_range[1]:
            
            final_y = y_vals[-1] 
            t_max_plot = t_range[1]

            t_patch_end = np.array([t_max_plot])
            y_patch_end = np.array([final_y])

            t_vals = np.concatenate((t_vals, t_patch_end))
            y_vals = np.concatenate((y_vals, y_patch_end))

        plt.plot(t_vals, y_vals, label=f"{label}")
        
        if t_range is not None and len(t_range) >= 2: 
            t_min = t_range[0]
            t_max = t_range[1]
        else:
            t_min = min(t_min, np.min(t_vals))
            t_max = max(t_max, np.max(t_vals))

    plt.title(title_string)
    plt.xlim(t_min,t_max)
    plt.xlabel('Time (s)')
    plt.ylabel('Response y(t)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def bode(
    *tf_instances, 
    labels: List[str] = None, 
    w_range: Union[Tuple[float,float], None] = None):

    if not tf_instances:
        raise ValueError("At least one transfer function must be provided.")
        
    plt.figure(figsize=(10, 8))

    responses = []
    for i, tf_instance in enumerate(tf_instances):
        responses.append(tf_instance.frequency_response)
    
    n_responses = len(responses) 
    if labels is None:
        labels = [f"Response {i+1}" for i in range(n_responses)]
    elif len(labels) != n_responses:
        raise ValueError("Number of labels must match number of transfer functions.")

    # Magnitude subplot
    w_min = float('inf')
    w_max = float('-inf')
    plt.subplot(2, 1, 1)
    for response, label in zip(responses, labels):
        w_vals = response.w_vals
        mag_vals = response.mag_vals
        plt.semilogx(w_vals, mag_vals, label=label)

        if w_range is not None and len(w_range) >= 2:
            w_min = w_range[0]
            w_max = w_range[1]
        else:
            w_min = min(w_min, np.min(w_vals))
            w_max = max(w_max, np.max(w_vals))
        
    plt.title('Bode Plot')
    plt.xlim(w_min,w_max)
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    
    # Phase subplot
    plt.subplot(2, 1, 2)
    for response, label in zip(responses, labels):
        w_vals = response.w_vals
        phase_vals = response.phase_vals
        plt.semilogx(w_vals, phase_vals, label=label)
    plt.xlim(w_min,w_max)
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def margin_plot(
        tf_instance,
        w_range: Union[Tuple[float,float],None] = None):
    
    if not tf_instance:
        raise ValueError("At least one transfer function must be provided.")
    
    w_vals = tf_instance.frequency_response.w_vals
    magnitude = tf_instance.frequency_response.mag_vals
    phase = tf_instance.frequency_response.phase_vals

    if w_range is not None and len(w_range) >= 2:
        w_min = w_range[0]
        w_max = w_range[1]
    else:
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
    if tf_instance.margin.w_phase_crossover is not None:
        plt.axvline(tf_instance.margin.w_phase_crossover, color='red', linestyle='--')
        plt.axhline(-tf_instance.margin.gain_margin, color='red', linestyle='--')
        plt.text(tf_instance.margin.w_phase_crossover, tf_instance.margin.gain_margin, f' gain_margin: {tf_instance.margin.gain_margin:.2f} dB at {tf_instance.margin.w_phase_crossover:.2f} rad/s', color='red', verticalalignment='bottom')
    if tf_instance.margin.w_gain_crossover is not None:
        plt.axvline(tf_instance.margin.w_gain_crossover, color='green', linestyle='--')
    plt.subplot(2, 1, 2)
    plt.semilogx(w_vals, phase)
    plt.xlabel('Frequency (rad/s)')
    plt.xlim(w_min,w_max)
    plt.ylabel('Phase (degrees)')
    plt.yticks(np.arange(phase_min, phase_max, 45))
    plt.grid(True)
    if tf_instance.margin.w_phase_crossover is not None:
        plt.axvline(tf_instance.margin.w_phase_crossover, color='red', linestyle='--')
    if tf_instance.margin.w_gain_crossover is not None:
        plt.axvline(tf_instance.margin.w_gain_crossover, color='green', linestyle='--')
        plt.axhline(-180 + tf_instance.margin.phase_margin, color='green', linestyle='--')
        plt.text(tf_instance.margin.w_gain_crossover, -180 + tf_instance.margin.phase_margin, f' phase_margin: {tf_instance.margin.phase_margin:.2f}° at {tf_instance.margin.w_gain_crossover:.2f} rad/s', color='green', verticalalignment='bottom')
    plt.tight_layout()
    plt.show()

def pzmap(tf_instance, x_range=(), y_range=(), n_points=100):
    if not tf_instance:
        raise ValueError("At least one transfer function must be provided.")
    
    zeros = np.array(tf_instance.zeros.numeric)
    poles = np.array(tf_instance.poles.numeric)
    
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
                x_min = 10**(np.floor(np.log10(abs(real_min)-1))) if real_min != 0 else -1
            x_max = np.abs(x_min)/10
            x_min = x_min*1.25

            y_max = 10**(np.ceil(np.log10(abs(imag_max)))) * (1 if imag_max > 0 else -1) if imag_max != 0 else 1
            y_max = y_max*1.25

            if np.abs(y_max) < np.abs(x_min):
                y_max = np.abs(x_min)
                
            if np.abs(y_max) > np.abs(x_min):
                x_min = -y_max

            x_range = (x_min, x_max)
            y_range = (-y_max, y_max)
            
    plt.figure()
    if zeros.size > 0:
        plt.scatter([z.evalf().as_real_imag()[0] for z in zeros],
                    [z.evalf().as_real_imag()[1] for z in zeros],
                    marker='o', color='blue', label='Zeros')
    if poles.size > 0:
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
            y_text = y_range[1] * 0.9
            x_text = y_text / -slope
            if x_text < x_range[0] * 0.95:
                x_text = x_range[0] * 0.95
                y_text = -slope * x_text
            plt.text(x_text, y_text, f'ζ={zeta}', color='gray', horizontalalignment='left')
    wn_vals = np.linspace(0, x_range[0], 6)
    for wn in wn_vals[1:]:
        circle = plt.Circle((0, 0), wn, color='gray', fill=False, linestyle='--', linewidth=0.5)
        plt.gca().add_artist(circle)
        plt.text(wn, 0, f'ωn={wn:.2f}', color='gray', horizontalalignment='left')
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

def nyquist(tf_instance):
    if not tf_instance:
        raise ValueError("At least one transfer function must be provided.")
    F_vals = np.array(tf_instance.frequency_response.F_vals)
    
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

