import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from models.function import Function
from classes.defs_sympy import *
from typing import Dict

class ControlEnvironment:
    def __init__(self, name):
        self.name = name
        self.global_variables = []
        self.global_constants = {}

        self.functions: Dict[str, Function] = {}

    def add_function(self,name):
        self.functions[name] = Function(self.global_variables, self.global_constants)


    # def print(self):
    #     print(f"Control Environment: {self.name}")
    #     print("Symbols:")
    #     for sym in self.symbols_list:
    #         print(f"  {sym}")
    #     print("Constants:")
    #     for const, val in self.constants_list.items():
    #         print(f"  {const} = {val}")
    #     print(f"f(t): {self.f}")
    #     print(f"F(s): {self.F}")

    # def clear(self):
    #     self.symbols_list = []
    #     self.constants_list = {}
    #     self.f = None
    #     self.F = None
    #     self.define_variable('s')
    #     self.define_variable('t', real=True)

    # def bode(self, w_range=(), num_points=10000):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")

    #     w_range = self.determine_w_range() if not w_range else (0.1, 100)

    #     w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
    #     s_vals = 1j * w_vals
    #     F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
    #     F_vals = F_func(s_vals)

    #     magnitude = 20 * np.log10(np.abs(F_vals))
    #     phase = np.unwrap(np.angle(F_vals))  * (180/np.pi)
        
    #     mag_max = np.ceil(np.max(magnitude) / 20) * 20
    #     mag_min = np.floor(np.min(magnitude) / 20) * 20

    #     phase_max = np.ceil(np.max(phase) / 90) * 90
    #     phase_min = np.floor(np.min(phase) / 90) * 90

    #     plt.figure(figsize=(10, 8))

    #     plt.subplot(2, 1, 1)
    #     plt.semilogx(w_vals, magnitude)
    #     plt.title('Bode Plot')
    #     plt.ylabel('Magnitude (dB)')
    #     plt.xlim(w_range)
    #     plt.yticks(np.arange(mag_min, mag_max, 20))
    #     plt.grid(True)

    #     plt.subplot(2, 1, 2)
    #     plt.semilogx(w_vals, phase)
    #     plt.xlabel('Frequency (rad/s)')
    #     plt.xlim(w_range)
    #     plt.ylabel('Phase (degrees)')
    #     plt.yticks(np.arange(phase_min, phase_max, 45))
    #     plt.grid(True)

    #     plt.tight_layout()
    #     plt.show()
    
    # def poles(self, print_poles=True):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")
        
    #     num, den = sp.fraction(sp.simplify(self.F.subs(self.constants_list)))
    #     poles = sp.roots(den, s)
    #     poles = list(poles.keys())
    #     if print_poles:
    #         print("Poles:")
    #         for p in poles:
    #             print(f"  {p}")
    #     return poles

    # def zeros(self,print_zeros=True):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")
        
    #     num, den = sp.fraction(sp.simplify(self.F.subs(self.constants_list)))
    #     zeros = sp.roots(num, s)
    #     zeros = list(zeros.keys())
    #     if print_zeros:
    #         print("Zeros:")
    #         for z in zeros:
    #             print(f"  {z}")
    #     return zeros

    # def nyquist(self, w_range=(0.1, 100), num_points=1000):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")

    #     w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
    #     s_vals = 1j * w_vals
    #     F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
    #     F_vals = F_func(s_vals)

    #     plt.figure()
    #     plt.plot(F_vals.real, F_vals.imag, label='Nyquist Plot')
    #     plt.plot(F_vals.real, -F_vals.imag, linestyle='--', color='gray')
    #     plt.title('Nyquist Plot')
    #     plt.xlabel('Real')
    #     plt.ylabel('Imaginary')
    #     plt.axhline(0, color='black', lw=0.5)
    #     plt.axvline(0, color='black', lw=0.5)
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()

    # def root_locus(self, k_range=(0, 100), num_points=1000):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")

    #     num, den = sp.fraction(sp.simplify(self.F))
    #     num_coeffs = sp.Poly(num, s).all_coeffs()
    #     den_coeffs = sp.Poly(den, s).all_coeffs()

    #     num_coeffs = [float(coef.subs(self.constants_list)) for coef in num_coeffs]
    #     den_coeffs = [float(coef.subs(self.constants_list)) for coef in den_coeffs]

    #     k_vals = np.linspace(k_range[0], k_range[1], num_points)
    #     poles_list = []

    #     for k in k_vals:
    #         char_eq_coeffs = den_coeffs + [-k * coeff for coeff in num_coeffs]
    #         roots = np.roots(char_eq_coeffs)
    #         poles_list.append(roots)

    #     plt.figure()
    #     for i in range(len(poles_list[0])):
    #         plt.plot([p[i].real for p in poles_list], [p[i].imag for p in poles_list], 'b-')

    #     plt.title('Root Locus')
    #     plt.xlabel('Real')
    #     plt.ylabel('Imaginary')
    #     plt.axhline(0, color='black', lw=0.5)
    #     plt.axvline(0, color='black', lw=0.5)
    #     plt.grid(True)
    #     plt.show()
    
    # def margin(self, print_margins=True, w_range=(), num_points=10000):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")
    #     gm, pm, wcp, wcg = None, None, None, None

    #     w_range = self.determine_w_range() if not w_range else (0.1, 100)

    #     w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
    #     s_vals = 1j * w_vals
    #     F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
    #     F_vals = F_func(s_vals)

    #     magnitude = 20 * np.log10(np.abs(F_vals))
    #     phase = np.unwrap(np.angle(F_vals)) * (180/np.pi)

    #     gain_crossings = np.where(np.diff(np.sign(magnitude)))[0]
    #     phase_crossings = np.where(np.diff(np.sign(phase + 180)))[0]

    #     if gain_crossings.size > 0:
    #         wcg = w_vals[gain_crossings[0]]
    #         pm = 180 + phase[gain_crossings[0]]
    #         if print_margins:
    #             print(f"Gain Crossover Frequency (wcg): {wcg:.2f} rad/s")
    #             print(f"Phase Margin (PM): {pm:.2f} degrees")
    #     else:
    #         if print_margins:
    #             print("No Gain Crossover Frequency found.")

    #     if phase_crossings.size > 0:
    #         wcp = w_vals[phase_crossings[0]]
    #         gm = -magnitude[phase_crossings[0]]
    #         if print_margins:
    #             print(f"Phase Crossover Frequency (wcp): {wcp:.2f} rad/s")
    #             print(f"Gain Margin (GM): {gm:.2f} dB")
    #     else:
    #         if print_margins:
    #             print("No Phase Crossover Frequency found.")

    #     return gm, pm, wcp, wcg

    # def determine_w_range(self,print_range=False):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")
    #     poles = self.poles(print_poles=False)
    #     zeros = self.zeros(print_zeros=False)
    #     all_freqs = []
    #     if poles is not None:
    #         all_freqs += [abs(p.evalf().as_real_imag()[0]) for p in poles]
    #     if zeros is not None:
    #         all_freqs += [abs(z.evalf().as_real_imag()[0]) for z in zeros]
    #     if all_freqs:
    #         min_freq, max_freq = float(min(all_freqs)), float(max(all_freqs))
    #         min_freq = 10**(np.floor(np.log10(min_freq)))/10 if min_freq > 0 else 0.1
    #         max_freq = 10**(np.ceil(np.log10(max_freq)))*10 if max_freq > 0 else 10
    #         w_range = (min_freq, max_freq)
    #         if print_range:
    #             print(f"Auto-determined frequency range: {w_range}")
    #         return w_range
    #     else:
    #         w_range = (0.1, 100)
    #         if print_range:
    #             print(f"Default frequency range: {w_range}")
    #         return w_range

    # def margin_plot(self, w_range=(), num_points=10000):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")
        
    #     w_range = self.determine_w_range() if not w_range else (0.1, 100)

    #     w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
    #     s_vals = 1j * w_vals
    #     F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
    #     F_vals = F_func(s_vals)

    #     magnitude = 20 * np.log10(np.abs(F_vals))
    #     phase = np.unwrap(np.angle(F_vals)) * (180/np.pi)
        
    #     mag_max = np.ceil(np.max(magnitude) / 20) * 20
    #     mag_min = np.floor(np.min(magnitude) / 20) * 20

    #     phase_max = np.ceil(np.max(phase) / 90) * 90
    #     phase_min = np.floor(np.min(phase) / 90) * 90

    #     gm, pm, wcp, wcg = self.margin(print_margins=False)

    #     plt.figure(figsize=(10, 8))

    #     plt.subplot(2, 1, 1)
    #     plt.semilogx(w_vals, magnitude)
    #     plt.title('Bode Plot with Margins')
    #     plt.ylabel('Magnitude (dB)')
    #     plt.xlim(w_range)
    #     plt.yticks(np.arange(mag_min, mag_max, 20))
    #     plt.grid(True)
    #     if wcp is not None:
    #         plt.axvline(wcp, color='red', linestyle='--')
    #         plt.axhline(-gm, color='red', linestyle='--')
    #         plt.text(wcp, -gm, f' GM: {gm:.2f} dB at {wcp:.2f} rad/s', color='red', verticalalignment='bottom')

    #     plt.subplot(2, 1, 2)
    #     plt.semilogx(w_vals, phase)
    #     plt.xlabel('Frequency (rad/s)')
    #     plt.xlim(w_range)
    #     plt.ylabel('Phase (degrees)')
    #     plt.yticks(np.arange(phase_min, phase_max, 45))
    #     plt.grid(True)
    #     if wcg is not None:
    #         plt.axvline(wcg, color='green', linestyle='--')
    #         plt.axhline(-180 + pm, color='green', linestyle='--')
    #         plt.text(wcg, -180 + pm, f' PM: {pm:.2f}° at {wcg:.2f} rad/s', color='green', verticalalignment='bottom')

    #     plt.tight_layout()
    #     plt.show()

    # def pzmap(self, x_range=(), y_range=(), num_points=1000):
    #     if self.F is None:
    #         raise ValueError("Laplace-domain function F(s) is not defined.")

    #     poles = self.poles(print_poles=False)
    #     zeros = self.zeros(print_zeros=False)

    #     if x_range == ():
    #         all_points = []
    #         all_points += [(0.0, 0.0)]
    #         if poles is not None:
    #             all_points += [p.evalf().as_real_imag() for p in poles]
    #         if zeros is not None:
    #             all_points += [z.evalf().as_real_imag() for z in zeros]
    #         if all_points:
    #             real_parts = [pt[0] for pt in all_points]
    #             imag_parts = [pt[1] for pt in all_points]

    #             real_min = float(min(real_parts))
    #             real_max = float(max(real_parts))
    #             imag_max = float(max(np.abs(imag_parts)))

    #             if real_min < 0:
    #                 x_min = -10**(np.ceil(np.log10(abs(real_min))))
    #             else:
    #                 x_min = 10**(np.floor(np.log10(abs(real_min)))) if real_min != 0 else -1
                                    
    #             if real_max < 0:
    #                 x_max = -10**(np.floor(np.log10(abs(real_max))))
    #             else:
    #                 x_max = 10**(np.ceil(np.log10(abs(real_max)))) if real_max != 0 else 0.1

    #             y_max = 10**(np.ceil(np.log10(abs(imag_max)))) * (1 if imag_max > 0 else -1) if imag_max != 0 else 1

    #             x_range = (x_min, x_max)
    #             y_range = (-y_max, y_max)

    #     plt.figure()
    #     if zeros:
    #         plt.scatter([z.evalf().as_real_imag()[0] for z in zeros],
    #                     [z.evalf().as_real_imag()[1] for z in zeros],
    #                     marker='o', color='blue', label='Zeros')
    #     if poles:
    #         plt.scatter([p.evalf().as_real_imag()[0] for p in poles],
    #                     [p.evalf().as_real_imag()[1] for p in poles],
    #                     marker='x', color='red', label='Poles')
    #     zeta_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    #     for zeta in zeta_vals:
    #         if zeta >= 1:
    #             continue
    #         theta = np.arccos(zeta)
    #         if np.tan(theta) != 0:
    #             slope = np.tan(theta)
    #             x_vals = np.linspace(x_range[0], x_range[1], num_points)
    #             y_vals_pos = slope * x_vals
    #             y_vals_neg = -slope * x_vals
    #             plt.plot(x_vals, y_vals_pos, linestyle='--', color='gray', linewidth=0.5)
    #             plt.plot(x_vals, y_vals_neg, linestyle='--', color='gray', linewidth=0.5)
    #             if zeta < 0.8:
    #                 y_text = y_range[1] * 0.8
    #                 x_text = y_text / -slope
    #                 plt.text(x_text, y_text, f'ζ={zeta}', color='gray', horizontalalignment='left')
    #             else:
    #                 x_text = x_range[0] * 0.8
    #                 y_text = -slope * x_text
    #                 plt.text(x_text, y_text, f'ζ={zeta}', color='gray', horizontalalignment='left')
    #     wn_vals = np.linspace(0, x_range[0], 5)
    #     for wn in wn_vals:
    #         circle = plt.Circle((0, 0), wn, color='gray', fill=False, linestyle='--', linewidth=0.5)
    #         plt.gca().add_artist(circle)
    #         plt.text(wn, 0, f'ωn={wn}', color='gray', horizontalalignment='left')
    #     plt.axvline(0, color='black', lw=0.5, linestyle='-')
    #     plt.xlim(x_range)
    #     plt.ylim(y_range)
    #     plt.axhline(0, color='black', lw=0.5)
    #     plt.axvline(0, color='black', lw=0.5)
    #     plt.title('Pole-Zero Map')
    #     plt.xlabel('Real')
    #     plt.ylabel('Imaginary')
    #     plt.grid(True)
    #     plt.legend(loc='lower left')
    #     plt.show()
    