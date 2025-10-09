import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

class ControlEnvironment:
    def __init__(self, name):
        self.name = name
        self.symbols_list = []
        self.constants_list = {}
        self.f = None
        self.F = None
        self.define_variable('s')
        self.define_variable('t', real=True)

    def evalf(self, func: str):
        if 'exp' in func and '.exp' not in func:
            func = func.replace('exp', 'sp.exp')
        if 'sin' in func and '.sin' not in func:
            func = func.replace('sin', 'sp.sin')
        if 'cos' in func and '.cos' not in func:
            func = func.replace('cos', 'sp.cos')
        if 'sqrt' in func and '.sqrt' not in func:
            func = func.replace('sqrt', 'sp.sqrt')
        return eval(func)

    def define_variable(self, name: str, print_output=False, **assumptions):
        symbol = sp.symbols(name, **assumptions)
        self.symbols_list.append(symbol)
        self.define_global_symbols()
        if print_output:
            print(f"Defined symbol: {symbol} with assumptions {assumptions}")
        return symbol

    def define_constant(self, name, value, print_output=False):
        symbol = self.define_variable(name, real=True)
        self.constants_list[symbol] = value
        if print_output:
            print(f"Defined constant: {symbol} = {value}")

    def define_global_symbols(self):
        for sym in self.symbols_list:
            globals()[str(sym)] = sym

    def define_f(self, f: str, print_output=False):
        self.f = self.evalf(f)
        self.F = self.L(self.f)
        if print_output:
            print(f"Defined equation: f(t) = {self.f}")
            print(f"Laplace Transform F(s) = {self.F}")

    def define_F(self, F: str, print_output=False):
        self.F = self.evalf(F)
        self.f = self.invL(self.F)
        if print_output:
            print(f"Defined equation: F(s) = {self.F}")
            print(f"Inverse Laplace Transform f(t) = {self.f}")

    def print_symbols(self):
        print("Defined symbols:")
        for sym in self.symbols_list:
            print(f"  {sym}, is_real: {sym.is_real}, is_positive: {sym.is_positive}")

    def get_symbol(self, name):
        for sym in self.symbols_list:
            if sym.name == name:
                return sym
        raise ValueError(f"Symbol '{name}' not found.")

    def L(self,f):
        return sp.laplace_transform(f, t, s, noconds=True)

    def invL(self,F):
        return sp.inverse_laplace_transform(F, s, t)
    
    def print(self):
        print(f"Control Environment: {self.name}")
        print("Symbols:")
        for sym in self.symbols_list:
            print(f"  {sym}")
        print("Constants:")
        for const, val in self.constants_list.items():
            print(f"  {const} = {val}")
        print(f"f(t): {self.f}")
        print(f"F(s): {self.F}")

    def clear(self):
        self.symbols_list = []
        self.constants_list = {}
        self.f = None
        self.F = None
        self.define_variable('s')
        self.define_variable('t', real=True)

    def plot_time_response(self, t_range=(0, 10), num_points=1000):

        if self.f is None:
            raise ValueError("Time-domain function f(t) is not defined.")

        t_vals = np.linspace(t_range[0], t_range[1], num_points)
        f_func = sp.lambdify(t, self.f.subs(self.constants_list), modules=['numpy'])
        f_vals = f_func(t_vals)

        plt.figure()
        plt.plot(t_vals, f_vals, label='f(t)')
        plt.title('Time Response')
        plt.xlabel('Time (s)')
        plt.ylabel('f(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def impulse(self, t_range=(0, 10), num_points=1000, delay_time=0):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        delay = sp.exp(-t*s).subs(t,delay_time)
        impulse = 1*delay
        Y = self.F*impulse
        y = self.invL(Y)

        t_vals = np.linspace(t_range[0], t_range[1], num_points)
        y_func = sp.lambdify(t, y.subs(self.constants_list), modules=['numpy'])
        y_vals = y_func(t_vals)

        plt.figure()
        plt.plot(t_vals, y_vals, label='f(t)')
        plt.title('Impulse Response')
        plt.xlabel('Time (s)')
        plt.ylabel('f(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def step(self, t_range=(0, 10), num_points=1000, delay_time=1):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        delay = sp.exp(-t*s).subs(t,delay_time)
        step = 1/s*delay
        Y = self.F*step
        y = self.invL(Y)

        t_vals = np.linspace(t_range[0], t_range[1], num_points)
        y_func = sp.lambdify(t, y.subs(self.constants_list), modules=['numpy'])
        y_vals = y_func(t_vals)

        plt.figure()
        plt.plot(t_vals, y_vals, label='f(t)')
        plt.title('Step Response')
        plt.xlabel('Time (s)')
        plt.ylabel('f(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def ramp(self, t_range=(0, 10), num_points=1000, delay_time=1):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        delay = sp.exp(-t*s).subs(t,delay_time)
        ramp = 1/s**2*delay
        Y = self.F*ramp
        y = self.invL(Y)

        t_vals = np.linspace(t_range[0], t_range[1], num_points)
        y_func = sp.lambdify(t, y.subs(self.constants_list), modules=['numpy'])
        y_vals = y_func(t_vals)

        plt.figure()
        plt.plot(t_vals, y_vals, label='f(t)')
        plt.title('Ramp Response')
        plt.xlabel('Time (s)')
        plt.ylabel('f(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def bode(self, w_range=(), num_points=10000):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        w_range = self.determine_w_range() if not w_range else (0.1, 100)

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
        F_vals = F_func(s_vals)

        magnitude = 20 * np.log10(np.abs(F_vals))
        phase = np.unwrap(np.angle(F_vals))  * (180/np.pi)
        
        mag_max = np.ceil(np.max(magnitude) / 20) * 20
        mag_min = np.floor(np.min(magnitude) / 20) * 20

        phase_max = np.ceil(np.max(phase) / 90) * 90
        phase_min = np.floor(np.min(phase) / 90) * 90

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.semilogx(w_vals, magnitude)
        plt.title('Bode Plot')
        plt.ylabel('Magnitude (dB)')
        plt.xlim(w_range)
        plt.yticks(np.arange(mag_min, mag_max, 20))
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.semilogx(w_vals, phase)
        plt.xlabel('Frequency (rad/s)')
        plt.xlim(w_range)
        plt.ylabel('Phase (degrees)')
        plt.yticks(np.arange(phase_min, phase_max, 45))
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    def poles(self, print_poles=True):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        
        num, den = sp.fraction(sp.simplify(self.F.subs(self.constants_list)))
        poles = sp.roots(den, s)
        poles = list(poles.keys())
        if print_poles:
            print("Poles:")
            for p in poles:
                print(f"  {p}")
        return poles

    def zeros(self,print_zeros=True):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        
        num, den = sp.fraction(sp.simplify(self.F.subs(self.constants_list)))
        zeros = sp.roots(num, s)
        zeros = list(zeros.keys())
        if print_zeros:
            print("Zeros:")
            for z in zeros:
                print(f"  {z}")
        return zeros

    def nyquist(self, w_range=(0.1, 100), num_points=1000):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
        F_vals = F_func(s_vals)

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
        plt.show()

    def root_locus(self, k_range=(0, 100), num_points=1000):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        num, den = sp.fraction(sp.simplify(self.F))
        num_coeffs = sp.Poly(num, s).all_coeffs()
        den_coeffs = sp.Poly(den, s).all_coeffs()

        num_coeffs = [float(coef.subs(self.constants_list)) for coef in num_coeffs]
        den_coeffs = [float(coef.subs(self.constants_list)) for coef in den_coeffs]

        k_vals = np.linspace(k_range[0], k_range[1], num_points)
        poles_list = []

        for k in k_vals:
            char_eq_coeffs = den_coeffs + [-k * coeff for coeff in num_coeffs]
            roots = np.roots(char_eq_coeffs)
            poles_list.append(roots)

        plt.figure()
        for i in range(len(poles_list[0])):
            plt.plot([p[i].real for p in poles_list], [p[i].imag for p in poles_list], 'b-')

        plt.title('Root Locus')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.grid(True)
        plt.show()
    
    def margin(self, print_margins=True, w_range=(), num_points=10000):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        gm, pm, wcp, wcg = None, None, None, None

        w_range = self.determine_w_range() if not w_range else (0.1, 100)

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
        F_vals = F_func(s_vals)

        magnitude = 20 * np.log10(np.abs(F_vals))
        phase = np.unwrap(np.angle(F_vals)) * (180/np.pi)

        gain_crossings = np.where(np.diff(np.sign(magnitude)))[0]
        phase_crossings = np.where(np.diff(np.sign(phase + 180)))[0]

        if gain_crossings.size > 0:
            wcg = w_vals[gain_crossings[0]]
            pm = 180 + phase[gain_crossings[0]]
            if print_margins:
                print(f"Gain Crossover Frequency (wcg): {wcg:.2f} rad/s")
                print(f"Phase Margin (PM): {pm:.2f} degrees")
        else:
            if print_margins:
                print("No Gain Crossover Frequency found.")

        if phase_crossings.size > 0:
            wcp = w_vals[phase_crossings[0]]
            gm = -magnitude[phase_crossings[0]]
            if print_margins:
                print(f"Phase Crossover Frequency (wcp): {wcp:.2f} rad/s")
                print(f"Gain Margin (GM): {gm:.2f} dB")
        else:
            if print_margins:
                print("No Phase Crossover Frequency found.")

        return gm, pm, wcp, wcg

    def determine_w_range(self,print_range=False):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        poles = self.poles(print_poles=False)
        zeros = self.zeros(print_zeros=False)
        all_freqs = []
        if poles is not None:
            all_freqs += [abs(p.evalf().as_real_imag()[0]) for p in poles]
        if zeros is not None:
            all_freqs += [abs(z.evalf().as_real_imag()[0]) for z in zeros]
        if all_freqs:
            min_freq, max_freq = float(min(all_freqs)), float(max(all_freqs))
            min_freq = 10**(np.floor(np.log10(min_freq)))/10 if min_freq > 0 else 0.1
            max_freq = 10**(np.ceil(np.log10(max_freq)))*10 if max_freq > 0 else 10
            w_range = (min_freq, max_freq)
            if print_range:
                print(f"Auto-determined frequency range: {w_range}")
            return w_range
        else:
            w_range = (0.1, 100)
            if print_range:
                print(f"Default frequency range: {w_range}")
            return w_range

    def margin_plot(self, w_range=(), num_points=10000):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        
        w_range = self.determine_w_range() if not w_range else (0.1, 100)

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
        F_vals = F_func(s_vals)

        magnitude = 20 * np.log10(np.abs(F_vals))
        phase = np.unwrap(np.angle(F_vals)) * (180/np.pi)
        
        mag_max = np.ceil(np.max(magnitude) / 20) * 20
        mag_min = np.floor(np.min(magnitude) / 20) * 20

        phase_max = np.ceil(np.max(phase) / 90) * 90
        phase_min = np.floor(np.min(phase) / 90) * 90

        gm, pm, wcp, wcg = self.margin(print_margins=False)

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.semilogx(w_vals, magnitude)
        plt.title('Bode Plot with Margins')
        plt.ylabel('Magnitude (dB)')
        plt.xlim(w_range)
        plt.yticks(np.arange(mag_min, mag_max, 20))
        plt.grid(True)
        # Gain Margin line
        if wcp is not None:
            plt.axvline(wcp, color='red', linestyle='--')
            plt.axhline(-gm, color='red', linestyle='--')
            plt.text(wcp, -gm, f' GM: {gm:.2f} dB at {wcp:.2f} rad/s', color='red', verticalalignment='bottom')

        plt.subplot(2, 1, 2)
        plt.semilogx(w_vals, phase)
        plt.xlabel('Frequency (rad/s)')
        plt.xlim(w_range)
        plt.ylabel('Phase (degrees)')
        plt.yticks(np.arange(phase_min, phase_max, 45))
        plt.grid(True)
        # Phase Margin line
        if wcg is not None:
            plt.axvline(wcg, color='green', linestyle='--')
            plt.axhline(-180 + pm, color='green', linestyle='--')
            plt.text(wcg, -180 + pm, f' PM: {pm:.2f}° at {wcg:.2f} rad/s', color='green', verticalalignment='bottom')

        plt.tight_layout()
        plt.show()