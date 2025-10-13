import sympy as sp
import numpy as np
import classes.defs_sympy as defs_sympy
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

class TransferFunction:
    def __init__(self,global_variables, global_constants):
        self.symbols = []
        self.functions =  []
        self.variables = global_variables
        self.constants = global_constants
        self.local_variables = []
        self.local_constants = {}
        self.input = None
        self.output = None
        self.lhs = None
        self.rhs = None
        self.tf = None
        self.num = None
        self.den = None
        self.zeros = None
        self.poles = None
        self.wcg = None
        self.wcg_found = False
        self.wcp = None
        self.wcp_found = False
        self.gm = None
        self.pm = None
        self.define_input('u')
        self.define_output('y')

    def define_input(self, name: str):
        # if self.input != None:
        #     remove_symbol(self.input, self.symbols)
        self.input = defs_sympy.add_function(name)
        self.functions.append(self.input)

    def define_output(self, name: str):
        # if self.output != None:
        #     remove_symbol(self.output, self.symbols)
        self.output = defs_sympy.add_function(name)
        self.functions.append(self.output)
        # self.output = name

    def define_lhs(self,string_input: str):
        self.lhs = string_input

    def define_rhs(self,string_input: str):
        self.rhs = string_input

    def add_variable(self):
        pass

    def remove_variables(self):
        pass

    def add_constant(self, name: str, value: float, description='None', unit='-'):
        symbol = defs_sympy.add_symbol(name, is_real=True)
        self.symbols.append(symbol)
        self.constants[name] = {
            "value": value,
            "description": description,
            "unit": f"[{unit}]",
            "symbol": symbol}
        
    def remove_constant(self):
        pass

    def get_constant_values(self):
        return {sp.Symbol(name): data["value"] for name, data in self.constants.items()}

    def define_tf_from_string(self, tf: str):
        self.tf = defs_sympy.tf_from_string(defs_sympy.translate_string(tf),self.constants)
        self.num = defs_sympy.get_numerator(self.tf)
        self.den = defs_sympy.get_denominator(self.tf)
        self.lhs, self.rhs = defs_sympy.get_equation(self.tf,input_symbol=self.input,output_symbol=self.output,constants=self.constants)
        self.zeros = self.get_zeros()
        self.poles = self.get_poles()
        self.gm, self.pm, self.wcg, self.wcp, self.wcg_found, self.wcp_found = self.get_margin()

    def define_tf_from_coefs(self,num_coefs=[],den_coefs=[]):
        self.tf = defs_sympy.tf_from_coefs(num_coefs,den_coefs,self.constants)
        self.num = defs_sympy.get_numerator(self.tf)
        self.den = defs_sympy.get_denominator(self.tf)
        self.lhs, self.rhs = defs_sympy.get_equation(self.tf,input_symbol=self.input,output_symbol=self.output,constants=self.constants)
        self.zeros = self.get_zeros()
        self.poles = self.get_poles()
        self.gm, self.pm, self.wcg, self.wcp, self.wcg_found, self.wcp_found = self.get_margin()

    def define_tf_from_equation(self,lhs,rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.tf = defs_sympy.tf_from_equation(lhs,rhs,input_symbol=self.input,output_symbol=self.output,constants=self.constants)
        self.num = defs_sympy.get_numerator(self.tf)
        self.den = defs_sympy.get_denominator(self.tf)
        self.zeros = self.get_zeros()
        self.poles = self.get_poles()
        self.gm, self.pm, self.wcg, self.wcp, self.wcg_found, self.wcp_found = self.get_margin()

## Functions

    def get_poles(self):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        
        den = self.den.subs(self.get_constant_values())
        poles = defs_sympy.roots(den)
        return (poles)

    def get_zeros(self):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        num = self.num.subs(self.get_constant_values())
        zeros = defs_sympy.roots(num)
        return zeros
    
    def get_system_order(self):
        pass

    def get_free_zeros(self):
        pass

    def get_free_poles(self):
        pass

    def determine_w_range(self,print_range=False):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        all_freqs = []
        if self.poles is not None:
            all_freqs += [abs(p.evalf().as_real_imag()[0]) for p in self.poles]
        if self.zeros is not None:
            all_freqs += [abs(z.evalf().as_real_imag()[0]) for z in self.zeros]
        if all_freqs:
            min_freq, max_freq = float(min(all_freqs)), float(max(all_freqs))
            min_freq = 10**(np.floor(np.log10(min_freq)))/10 if min_freq > 0 else 0.1
            max_freq = 10**(np.ceil(np.log10(max_freq)))*10 if max_freq > 0 else 10
            w_range = (min_freq*0.1, max_freq*10)
            if print_range:
                print(f"Auto-determined frequency range: {w_range}")
            return w_range
        else:
            w_range = (0.1, 100)
            if print_range:
                print(f"Default frequency range: {w_range}")
            return w_range
        
    def get_margin(self, w_range=(), num_points=10000):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        gm, pm, wcp, wcg = None, None, None, None
        s, t = defs_sympy.define_st()
        symbols = {'s': s, 't': t}
        for name, const_data in self.constants.items():
            symbols[name] = sp.Symbol(name)
        locals().update(symbols)

        w_range = self.determine_w_range() if not w_range else (0.1, 100)

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.tf.subs(self.get_constant_values()), modules=['numpy'])
        F_vals = F_func(s_vals)

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

## Plots

    def impulse(self, t_range=(0, 10), num_points=1000, delay_time=0):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        impulse = defs_sympy.impulse_function(delay_time)
        Y = self.tf*impulse
        yt = defs_sympy.invL(Y).subs(self.get_constant_values())

        t_vals, yt_vals = defs_sympy.lambdify(yt, t_range, num_points)

        plt.figure()
        plt.plot(t_vals, yt_vals, label='y(t)')
        plt.title('Impulse Response')
        plt.xlabel('Time (s)')
        plt.ylabel('y(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def step(self, t_range=(0, 10), num_points=1000, delay_time=1):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        
        step = defs_sympy.step_function(delay_time)
        Y = self.tf*step
        yt = defs_sympy.invL(Y).subs(self.get_constant_values())

        t_vals, yt_vals = defs_sympy.lambdify(yt, t_range, num_points)

        plt.figure()
        plt.plot(t_vals, yt_vals, label='y(t)')
        plt.title('Step Response')
        plt.xlabel('Time (s)')
        plt.ylabel('f(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def ramp(self, t_range=(0, 10), num_points=1000, delay_time=1):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        ramp = defs_sympy.ramp_function(delay_time)
        Y = self.tf*ramp
        yt = defs_sympy.invL(Y).subs(self.get_constant_values())

        t_vals, yt_vals = defs_sympy.lambdify(yt, t_range, num_points)

        plt.figure()
        plt.plot(t_vals, yt_vals, label='(t)')
        plt.title('Ramp Response')
        plt.xlabel('Time (s)')
        plt.ylabel('y(t)')
        plt.grid(True)
        plt.legend()
        plt.show()


    def bode(self, w_range=(), n_points=10000):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        s, t = defs_sympy.define_st()
        symbols = {'s': s, 't': t}
        for name, const_data in self.constants.items():
            symbols[name] = sp.Symbol(name)
        locals().update(symbols)

        w_range = self.determine_w_range() if not w_range else (0.1, 100)

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), n_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.tf.subs(self.get_constant_values()), modules=['numpy'])
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
        plt.yticks(np.arange(mag_min, mag_max + 20, 20))
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.semilogx(w_vals, phase)
        plt.xlabel('Frequency (rad/s)')
        plt.xlim(w_range)
        plt.ylabel('Phase (degrees)')
        plt.yticks(np.arange(phase_min, phase_max + 45, 45))
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def margin_plot(self, w_range=(), num_points=10000):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        s, t = defs_sympy.define_st()
        symbols = {'s': s, 't': t}
        for name, const_data in self.constants.items():
            symbols[name] = sp.Symbol(name)
        locals().update(symbols)
        
        w_range = self.determine_w_range() if not w_range else (0.1, 100)

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.tf.subs(self.get_constant_values()), modules=['numpy'])
        F_vals = F_func(s_vals)

        magnitude = 20 * np.log10(np.abs(F_vals))
        phase = np.unwrap(np.angle(F_vals)) * (180/np.pi)
        
        mag_max = np.ceil(np.max(magnitude) / 20) * 20
        mag_min = np.floor(np.min(magnitude) / 20) * 20

        phase_max = np.ceil(np.max(phase) / 90) * 90
        phase_min = np.floor(np.min(phase) / 90) * 90


        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.semilogx(w_vals, magnitude)
        plt.title('Bode Plot with Margins')
        plt.ylabel('Magnitude (dB)')
        plt.xlim(w_range)
        plt.yticks(np.arange(mag_min, mag_max, 20))
        plt.grid(True)
        if self.wcp is not None:
            plt.axvline(self.wcp, color='red', linestyle='--')
            plt.axhline(-self.gm, color='red', linestyle='--')
            plt.text(self.wcp, self.gm, f' GM: {self.gm:.2f} dB at {self.wcp:.2f} rad/s', color='red', verticalalignment='bottom')
        if self.wcg is not None:
            plt.axvline(self.wcg, color='green', linestyle='--')

        plt.subplot(2, 1, 2)
        plt.semilogx(w_vals, phase)
        plt.xlabel('Frequency (rad/s)')
        plt.xlim(w_range)
        plt.ylabel('Phase (degrees)')
        plt.yticks(np.arange(phase_min, phase_max, 45))
        plt.grid(True)
        if self.wcp is not None:
            plt.axvline(self.wcp, color='red', linestyle='--')
        if self.wcg is not None:
            plt.axvline(self.wcg, color='green', linestyle='--')
            plt.axhline(-180 + self.pm, color='green', linestyle='--')
            plt.text(self.wcg, -180 + self.pm, f' PM: {self.pm:.2f}° at {self.wcg:.2f} rad/s', color='green', verticalalignment='bottom')

        plt.tight_layout()
        plt.show()
    
    def pzmap(self, x_range=(), y_range=(), num_points=1000):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        s, t = defs_sympy.define_st()
        symbols = {'s': s, 't': t}
        for name, const_data in self.constants.items():
            symbols[name] = sp.Symbol(name)
        locals().update(symbols)

        if x_range == ():
            all_points = []
            all_points += [(0.0, 0.0)]
            if self.poles is not None:
                all_points += [p.evalf().as_real_imag() for p in self.poles]
            if self.zeros is not None:
                all_points += [z.evalf().as_real_imag() for z in self.zeros]
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
        if self.zeros:
            plt.scatter([z.evalf().as_real_imag()[0] for z in self.zeros],
                        [z.evalf().as_real_imag()[1] for z in self.zeros],
                        marker='o', color='blue', label='Zeros')
        if self.poles:
            plt.scatter([p.evalf().as_real_imag()[0] for p in self.poles],
                        [p.evalf().as_real_imag()[1] for p in self.poles],
                        marker='x', color='red', label='Poles')
        zeta_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        for zeta in zeta_vals:
            if zeta >= 1:
                continue
            theta = np.arccos(zeta)
            if np.tan(theta) != 0:
                slope = np.tan(theta)
                x_vals = np.linspace(x_range[0], x_range[1], num_points)
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

    def nyquist(self, w_range=(0.1, 100), num_points=1000):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")
        s, t = defs_sympy.define_st()
        symbols = {'s': s, 't': t}
        for name, const_data in self.constants.items():
            symbols[name] = sp.Symbol(name)
        locals().update(symbols)
        
        w_range = self.determine_w_range() if not w_range else (0.1, 100)

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), num_points)
        s_vals = 1j * w_vals
        F_func = sp.lambdify(s, self.tf.subs(self.get_constant_values()), modules=['numpy'])
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

    ## Prints

    def print_symbols(self):
        print("Symbols:")
        for symbol in self.symbols:
            print(f"  {symbol}, is_real: {symbol.is_real}, is_positive: {symbol.is_positive}")

    def print_variables(self):
        print("Variables:")
        for variable in self.variables:
            print(f"  {variable}")

    def print_constants(self):
        print('Constants:')
        for symbol, data in self.constants.items():
            print(f"  {symbol}: {data['value']} {data['unit']} - {data['description']}")

    def print_input(self):
        print(f"Input function: {self.input}")

    def print_output(self):
        print(f"Output function: {self.output}")

    def print_lhs(self):
        print(f"LHS: {self.lhs}")

    def print_rhs(self):
        print(f"RHS: {self.rhs}")

    def print_equation(self):
        print(f"{self.lhs} = {self.rhs}")

    def print_tf(self):
        if self.tf is None:
            print("Laplace-domain function F(s) is not defined.")
            return
        print(f"Transfer function: G(s) = {sp.simplify(self.tf)}")
    
    def print_numerator(self):
        print(f"Numerator: {self.num}")

    def print_denominator(self):
        print(f"Denominator: {self.den}")

    def print_zeros(self):
        print("Zeros:")
        for z in self.zeros:
            print(f"  {z}")

    def print_poles(self):
        print("Poles:")
        for p in self.poles:
            print(f"  {p}")

    def print_margin(self):
        if self.wcg_found:
            print(f"Gain Crossover Frequency (wcg): {self.wcg:.2f} rad/s")
            print(f"Phase Margin (PM): {self.pm:.2f} degrees")
        else:
            print("No Gain Crossover Frequency found.")

        if self.wcp_found:
            print(f"Phase Crossover Frequency (wcp): {self.wcp:.2f} rad/s")
            print(f"Gain Margin (GM): {self.gm:.2f} dB")
        else:
            print("No Phase Crossover Frequency found.")

    def print_all(self):
        print("==================================================")
        self.print_symbols()
        print('')
        self.print_variables()
        print('')
        self.print_constants()
        print('')
        print("==================================================")
        self.print_input()
        print('')
        self.print_output()
        print('')
        print("==================================================")
        self.print_equation()
        print('')
        print("==================================================")
        self.print_tf()
        print('')
        self.print_numerator()
        print('')
        self.print_denominator()
        print('')
        print("==================================================")
        self.print_zeros()
        print('')
        self.print_poles()
        print("==================================================")
        self.print_margin()
        print('')