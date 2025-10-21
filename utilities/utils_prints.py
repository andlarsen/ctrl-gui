import numpy as np
import sympy as sp
from dataclasses import asdict

## Prints
def print_name(self):
    print('')
    print(f'{self.name}(s) = {self.output[1]} / {self.input[1]}')
    if self.description:
        print('')
        print(f'Description: {self.description}')

def print_symbols(self):
    print("Symbols:")
    # for symbol in self.symbols:
    for symbol, data in self.symbols.items():
        print(f"  {symbol}, \tis_real: {data['is_real']}, \tis_positive: {data['is_positive']}, \tis_constant: {data['is_constant']}\t - {data['description']}")

def print_constants(self):
    print('Constants:')
    for symbol, data in self.constants.items():
        if data['is_global'] == True:
            print(f"  (global)  {symbol}: {data['value']} \t[{data['unit']}] \t- {data['description']}")
        else:
            print(f"  (local)   {symbol}: {data['value']} \t[{data['unit']}] \t- {data['description']}")

def print_input(self):
    print(f"Input function: {self.input[0]}, {self.input[1]}")

def print_output(self):
    print(f"Output function: {self.output[0]}, {self.output[1]}")

def print_differential_equation(self):
    print(f"Equation:")
    print(f"Symbolic:   {self.differential_equation.symbolic.lhs} = {self.differential_equation.symbolic.rhs}")
    print(f"Numeric:    {self.differential_equation.numeric.lhs} = {self.differential_equation.numeric.rhs}")

def print_tf(self):
    if self.symbolic is None:
        print("Laplace-domain function F(s) is not defined.")
        return
    print(f"Transfer function:")
    print(f"Symbolic:   {self.tf_name}(s) = {self.symbolic}")
    print(f"Numeric:    {self.tf_name}(s) = {self.numeric}")

def print_numerator(self):
    print(f"Numerator:")
    print(f"Symbolic:   N(s) = {self.numerator.symbolic}")
    print(f"Numeric:    N(s) = {self.numerator.numeric}")

def print_denominator(self):
    print(f"Denominator:")
    print(f"Symbolic:   D(s) = {self.denominator.symbolic}")
    print(f"Numeric:    D(s) = {self.denominator.numeric}")

def print_zeros(self):
    if self.zeros.numeric:
        print("Zeros: (numeric)")
        for z in self.zeros.numeric:
            print(f"  {z}")
    else:
        print("No zeros exist")
        
import sympy as sp
def print_poles(self):
    if self.poles.numeric:
        print("Poles: (numeric)")
        for p in self.poles.numeric:
            print(f"  {p}")
    else:
        print("No poles exist")

def print_margin(self):
    if self.margin.w_gain_crossover is not None:
        print(f"Gain Crossover Frequency (wcg): {self.margin.w_gain_crossover:.2f} rad/s")
        print(f"Phase Margin (PM): {self.margin.phase_margin:.2f} degrees")
    else:
        print("No Gain Crossover Frequency found.")

    if self.margin.w_phase_crossover is not None:
        print(f"Phase Crossover Frequency (wcp): {self.margin.w_phase_crossover:.2f} rad/s")
        print(f"Gain Margin (GM): {self.margin.gain_margin:.2f} dB")
    else:
        print("No Phase Crossover Frequency found.")

def print_impulse_response_info(self):
    print(f"\nImpulse response info:")
    print("=" * 40)
    
    info_dict = self.impulse_response.info._asdict()

    print_order = [
        "t_peak", "y_peak",
        "t_settling","t_half",
        "decay_rate", "integral", "energy",
        "num_oscillations", "damping_ratio", "natural_freq"
    ]
    
    for key in print_order:
        metric = info_dict.get(key)
        
        if metric is None:
            continue
            
        label = metric.label
        val = metric.value
        unit = metric.unit
        
        is_valid = val is not None and (not isinstance(val, float) or not np.isnan(val))
        
        if is_valid:
            print(f"{label:.<40} {val: >8.4f} [{unit}]")
        else:
            print(f"{label:.<40} {'N/A': >8}")

def print_step_response_info(self):
    print(f"\nStep Response Info:")
    print("=" * 40)
    
    info_dict = self.step_response.info._asdict()

    print_order = [
        "y_final", "y_initial",
        "t_peak", "y_peak", "overshoot_percent",
        "t_rise", "t_settling",
        "num_oscillations", "damping_ratio", "natural_freq"
    ]

    for key in print_order:
        metric = info_dict.get(key)
        
        if metric is None:
            continue
            
        label = metric.label
        val = metric.value
        unit = metric.unit
        
        is_valid = val is not None and (not isinstance(val, float) or not np.isnan(val))
        
        if is_valid:
            print(f"{label:.<40} {val: >8.4f} [{unit}]")
        else:
            print(f"{label:.<40} {'N/A': >8}")

def print_ramp_response_info(self):
    print(f"\nRamp Response Info:")
    print("=" * 40)
    
    info_dict = self.ramp_response.info._asdict()

    print_order = [
        "y_final",
        "t_peak", "y_peak",
        "Kv", "e_final", "e_max_tracking",
        "t_lag"
    ]

    for key in print_order:
        metric = info_dict.get(key)
        
        if metric is None:
            continue
            
        label = metric.label
        val = metric.value
        unit = metric.unit
        
        is_valid = val is not None and (not isinstance(val, float) or not np.isnan(val))
        
        if is_valid:
            print(f"{label:.<40} {val: >8.4f} [{unit}]")
        else:
            print(f"{label:.<40} {'N/A': >8}")


def print_all(self):
    print("==================================================")
    print_name(self)
    print("")
    print_input(self)
    print_output(self)
    print('')
    print("==================================================")
    print_symbols(self)
    print('')
    print_constants(self)
    print('')
    print("==================================================")
    print_differential_equation(self)
    print('')
    print("==================================================")
    print_tf(self)
    print('')
    print_numerator(self)
    print('')
    print_denominator(self)
    print('')
    print("==================================================")
    print_zeros(self)
    print('')
    print_poles(self)
    print("==================================================")
    print_margin(self)
    print('')
    print("==================================================")
    print_impulse_response_info(self)
    print('')
    print_step_response_info(self)
    print('')
    print_ramp_response_info(self)
    print('')