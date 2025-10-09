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

    def define_variable(self, *names, **assumptions):
        symbol = sp.symbols(' '.join(names), **assumptions)
        
        if not isinstance(symbol, tuple):
            symbol = (symbol,)

        for sym in symbol:
            self.symbols_list.append(sym)
            print(f"Defined symbol: {sym} with assumptions {assumptions}")
        self.define_global_symbols()

    def define_constant(self, name, value):
        symbol = sp.symbols(name, real=True)
        self.symbols_list.append(symbol)
        self.constants_list[symbol] = value
        print(f"Defined constant: {symbol} = {value}")
        self.define_global_symbols()

    def define_global_symbols(self):
        for sym in self.symbols_list:
            globals()[str(sym)] = sym

    def define_f(self, f: str):
        if 'exp' in f and '.exp' not in f:
            f = f.replace('exp', 'sp.exp')
        self.f = eval(f)
        print(f"Defined equation: f(t) = {self.f}")
        self.F = self.L(self.f)
        print(f"Laplace Transform F(s) = {self.F}")

    def define_F(self, F: str):
        self.F = eval(F)
        print(f"Defined equation: F(s) = {self.F}")
        self.f = self.invL(self.F)
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
        plt.title('Time Response')
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
        plt.title('Time Response')
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
        plt.title('Time Response')
        plt.xlabel('Time (s)')
        plt.ylabel('f(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def bode(self, w_range=(0.1, 100), num_points=1000):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
        F_vals = F_func(s_vals)

        magnitude = 20 * np.log10(np.abs(F_vals))
        phase = np.angle(F_vals, deg=True)

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.semilogx(w_vals, magnitude)
        plt.title('Bode Plot')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.semilogx(w_vals, phase)
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Phase (degrees)')
        plt.yticks(np.arange(-180, 0, 45))
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    def poles(self):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        
        num, den = sp.fraction(sp.simplify(self.F.subs(self.constants_list)))
        poles = sp.roots(den, s)
        print("Poles:")
        for p in poles:
            print(f"  {p}")

    def zeros(self):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        
        num, den = sp.fraction(sp.simplify(self.F.subs(self.constants_list)))
        zeros = sp.roots(num, s)
        print("Zeros:")
        for z in zeros:
            print(f"  {z}")

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
    
    def margin(self):
        if self.F is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        w_vals = np.logspace(-2, 2, 1000)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.F.subs(self.constants_list), modules=['numpy'])
        F_vals = F_func(s_vals)

        magnitude = 20 * np.log10(np.abs(F_vals))
        phase = np.angle(F_vals, deg=True)

        gain_crossings = np.where(np.diff(np.sign(magnitude)))[0]
        phase_crossings = np.where(np.diff(np.sign(phase + 180)))[0]

        if gain_crossings.size > 0:
            wcg = w_vals[gain_crossings[0]]
            pm = 180 + phase[gain_crossings[0]]
            print(f"Gain Crossover Frequency (wcg): {wcg:.2f} rad/s")
            print(f"Phase Margin (PM): {pm:.2f} degrees")
        else:
            print("No Gain Crossover Frequency found.")

        if phase_crossings.size > 0:
            wcp = w_vals[phase_crossings[0]]
            gm = -magnitude[phase_crossings[0]]
            print(f"Phase Crossover Frequency (wcp): {wcp:.2f} rad/s")
            print(f"Gain Margin (GM): {gm:.2f} dB")
        else:
            print("No Phase Crossover Frequency found.")