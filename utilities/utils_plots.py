import sympy as sp
import numpy as np
import scipy.signal as signal
import utilities.utils_sympy as utils_sympy
import utilities.utils_transfer_function as defs_tf
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from typing import Dict, List, Any, Tuple, Optional, Union

# Import logger
import logging
from utilities.logger import get_logger
log = get_logger(__name__, level=logging.DEBUG, logfile='logs/main.log')


def plot_response(
    *tf_instances, 
    labels: List[str] = None, 
    delay_times: List[float] = None, 
    t_range: Union[Tuple[float, float], None] = None,
    response_type: str = "step"
):
    log.debug("plot_response() called")

    if not tf_instances:
        log.error("plot_response(): No transfer function instances provided")
        raise ValueError("At least one transfer function instance must be provided.")

    responses = []
    title_string = ""
    for i, tf_instance in enumerate(tf_instances):
        try:
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
                log.warning(f"plot_response(): Unrecognized response type '{response_type}'")
                raise ValueError(f"Response type '{response_type}' is not supported.")
        except Exception as e:
            log.error(f"plot_response(): Failed to retrieve {response_type} response from instance {i}: {e}", exc_info=True)
            raise

    n_responses = len(responses)
    if labels is None or len(labels) != n_responses:
        if labels is not None:
            log.warning(f"plot_response(): Length of 'labels' ({len(labels)}) does not match number of responses ({n_responses}). Generating default labels.")
        labels = [f"Response {i+1}" for i in range(n_responses)]

    if delay_times is None:
        delay_times = [0.0] * n_responses
    elif len(delay_times) == 1 and n_responses > 1:
        log.info(f"plot_response(): Applying single delay time of {delay_times[0]}s to all {n_responses} responses.")
        delay_times = [delay_times[0]] * n_responses
    elif len(delay_times) != n_responses:
        log.warning("plot_response(): Mismatch in delay_times length. Defaulting to zero delays.")
        delay_times = [0.0] * n_responses

    t_min = 0
    t_max = float('-inf')
    plt.figure(figsize=(8, 5))

    for i, (response, label, delay_time) in enumerate(zip(responses, labels, delay_times)):
        try:
            t_vals = np.array(response.t_vals) + delay_time
            y_vals = np.array(response.y_vals)

            if t_vals[0] > 0:
                t_vals = np.concatenate(([0.0], t_vals))
                y_vals = np.concatenate(([0.0], y_vals))

            if t_range and len(t_range) > 1 and t_vals[-1] < t_range[1]:
                final_y = y_vals[-1]
                t_vals = np.concatenate((t_vals, [t_range[1]]))
                y_vals = np.concatenate((y_vals, [final_y]))

            plt.plot(t_vals, y_vals, label=label)

            if t_range and len(t_range) >= 2:
                t_min = t_range[0]
                t_max = t_range[1]
            else:
                t_min = min(t_min, np.min(t_vals))
                t_max = max(t_max, np.max(t_vals))

            log.debug(f"plot_response(): Plotted response {i+1} with label '{label}' and delay {delay_time}s")
        except Exception as e:
            log.error(f"plot_response(): Failed to plot response {i+1}: {e}", exc_info=True)
            raise

    plt.title(title_string)
    plt.xlim(t_min, t_max)
    plt.xlabel('Time (s)')
    plt.ylabel('Response y(t)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    log.info("plot_response(): Plot displayed successfully")


def bode(*tf_instances, labels: List[str] = None, w_range: Union[Tuple[float, float], None] = None):
    log.debug("bode() called")

    if not tf_instances:
        log.error("bode(): No transfer function instances provided")
        raise ValueError("At least one transfer function must be provided.")

    try:
        responses = [tf.frequency_response for tf in tf_instances]
        n_responses = len(responses)

        if labels is None:
            labels = [f"Response {i+1}" for i in range(n_responses)]
        elif len(labels) != n_responses:
            log.error("bode(): Label count mismatch")
            raise ValueError("Number of labels must match number of transfer functions.")

        plt.figure(figsize=(10, 8))
        
        # Magnitude plot
        w_min, w_max = float('inf'), float('-inf')
        plt.subplot(2, 1, 1)
        print(w_range)
        for response, label in zip(responses, labels):
            w_vals, mag_vals = response.w_vals, response.mag_vals
            plt.semilogx(w_vals, mag_vals, label=label)
            w_min = min(w_min, np.min(w_vals)) if not w_range else w_range[0]
            w_max = max(w_max, np.max(w_vals)) if not w_range else w_range[1]
            log.debug(f"bode(): Plotted magnitude for '{label}'")

        plt.title('Bode Plot')
        plt.xlim(w_min, w_max)
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.legend()

        # Phase plot
        plt.subplot(2, 1, 2)
        for response, label in zip(responses, labels):
            plt.semilogx(response.w_vals, response.phase_vals, label=label)
            log.debug(f"bode(): Plotted phase for '{label}'")

        plt.xlim(w_min, w_max)
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Phase (degrees)')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        log.info("bode(): Bode plot displayed successfully")
    except Exception as e:
        log.error(f"bode(): Failed to generate plot: {e}", exc_info=True)
        raise


def margin_plot(tf_instance, w_range: Union[Tuple[float, float], None] = None):
    log.debug("margin_plot() called")

    if not tf_instance:
        log.error("margin_plot(): No transfer function instance provided")
        raise ValueError("At least one transfer function must be provided.")

    try:
        w_vals = tf_instance.frequency_response.w_vals
        magnitude = tf_instance.frequency_response.mag_vals
        phase = tf_instance.frequency_response.phase_vals

        w_min, w_max = w_range if w_range else (np.min(w_vals), np.max(w_vals))
        mag_max = np.ceil(np.max(magnitude) / 20) * 20
        mag_min = np.floor(np.min(magnitude) / 20) * 20
        phase_max = np.ceil(np.max(phase) / 90) * 90
        phase_min = np.floor(np.min(phase) / 90) * 90

        plt.figure(figsize=(10, 8))

        # Magnitude
        plt.subplot(2, 1, 1)
        plt.semilogx(w_vals, magnitude)
        plt.title('Bode Plot with Margins')
        plt.ylabel('Magnitude (dB)')
        plt.xlim(w_min, w_max)
        plt.yticks(np.arange(mag_min, mag_max, 20))
        plt.grid(True)

        if tf_instance.margin.w_phase_crossover is not None:
            plt.axvline(tf_instance.margin.w_phase_crossover, color='red', linestyle='--')
            plt.axhline(-tf_instance.margin.gain_margin, color='red', linestyle='--')
            plt.text(tf_instance.margin.w_phase_crossover, tf_instance.margin.gain_margin,
                     f' gain_margin: {tf_instance.margin.gain_margin:.2f} dB at {tf_instance.margin.w_phase_crossover:.2f} rad/s',
                     color='red', verticalalignment='bottom')

        if tf_instance.margin.w_gain_crossover is not None:
            plt.axvline(tf_instance.margin.w_gain_crossover, color='green', linestyle='--')

        # Phase
        plt.subplot(2, 1, 2)
        plt.semilogx(w_vals, phase)
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Phase (degrees)')
        plt.xlim(w_min, w_max)
        plt.yticks(np.arange(phase_min, phase_max, 45))
        plt.grid(True)

        if tf_instance.margin.w_phase_crossover is not None:
            plt.axvline(tf_instance.margin.w_phase_crossover, color='red', linestyle='--')

        if tf_instance.margin.w_gain_crossover is not None:
            plt.axvline(tf_instance.margin.w_gain_crossover, color='green', linestyle='--')
            plt.axhline(-180 + tf_instance.margin.phase_margin, color='green', linestyle='--')
            plt.text(tf_instance.margin.w_gain_crossover, -180 + tf_instance.margin.phase_margin,
                     f' phase_margin: {tf_instance.margin.phase_margin:.2f}° at {tf_instance.margin.w_gain_crossover:.2f} rad/s',
                     color='green', verticalalignment='bottom')

        plt.tight_layout()
        plt.show()
        log.info("margin_plot(): Margin plot displayed successfully")
    except Exception as e:
        log.error(f"margin_plot(): Failed to generate plot: {e}", exc_info=True)
        raise

def pzmap(tf_instance, x_range=(), y_range=(), n_points=100):
    log.debug("pzmap() called")

    if not tf_instance:
        log.error("pzmap(): No transfer function instance provided")
        raise ValueError("At least one transfer function must be provided.")

    try:
        zeros = np.array(tf_instance.zeros.numeric)
        poles = np.array(tf_instance.poles.numeric)

        # Auto-range calculation
        if x_range == ():
            all_points = [(0.0, 0.0)]
            if poles is not None:
                all_points += [p.evalf().as_real_imag() for p in poles]
            if zeros is not None:
                all_points += [z.evalf().as_real_imag() for z in zeros]

            real_parts = [pt[0] for pt in all_points]
            imag_parts = [pt[1] for pt in all_points]

            real_min = float(min(real_parts))
            real_max = float(max(real_parts))
            imag_max = float(max(np.abs(imag_parts)))

            x_min = -10**(np.ceil(np.log10(abs(real_min)))) if real_min < 0 else -1
            x_max = abs(x_min) / 10
            x_min *= 1.25

            y_max = 10**(np.ceil(np.log10(abs(imag_max)))) if imag_max != 0 else 1
            y_max *= 1.25

            if abs(y_max) < abs(x_min):
                y_max = abs(x_min)
            if abs(y_max) > abs(x_min):
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

        # Damping ratio lines
        zeta_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        for zeta in zeta_vals:
            if zeta >= 1:
                continue
            theta = np.arccos(zeta)
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

        # Natural frequency circles
        wn_vals = np.linspace(0, x_range[0], 6)
        for wn in wn_vals[1:]:
            circle = plt.Circle((0, 0), wn, color='gray', fill=False, linestyle='--', linewidth=0.5)
            plt.gca().add_artist(circle)
            plt.text(wn, 0, f'ωn={wn:.2f}', color='gray', horizontalalignment='left')

        plt.axvline(0, color='black', lw=0.5)
        plt.axhline(0, color='black', lw=0.5)
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.title('Pole-Zero Map')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()
        log.info("pzmap(): Pole-zero map displayed successfully")
    except Exception as e:
        log.error(f"pzmap(): Failed to generate plot: {e}", exc_info=True)
        raise


def nyquist(tf_instance):
    log.debug("nyquist() called")

    if not tf_instance:
        log.error("nyquist(): No transfer function instance provided")
        raise ValueError("At least one transfer function must be provided.")

    try:
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
        plt.legend()
        plt.tight_layout()
        plt.show()
        log.info("nyquist(): Nyquist plot displayed successfully")
    except Exception as e:
        log.error(f"nyquist(): Failed to generate plot: {e}", exc_info=True)
        raise
