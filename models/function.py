import sympy as sp
import numpy as np
from classes.defs_sympy import *

class Function:
    def __init__(self,global_variables, global_constants):
        self.symbols = []
        self.variables = global_variables
        self.constants = global_constants
        self.local_variables = []
        self.local_constants = {}
        self.input = None
        self.output = None
        self.yt = None
        self.tf = None
        self.define_input('u')
        self.define_output('y')

    def define_input(self, name: str):
        if self.input != None:
            remove_symbol(self.input, self.symbols)
        self.input = add_symbol(name, is_real=True)
        self.symbols.append(self.input)

    def define_output(self, name: str):
        if self.output != None:
            remove_symbol(self.output, self.symbols)
        self.output = add_symbol(name, is_real=True)
        self.symbols.append(self.output)

    def add_variable(self):
        pass

    def remove_variables(self):
        pass

    def add_constant(self, name: str, value: float, description='None', unit='-'):
        symbol = add_symbol(name, is_real=True)
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

    def define_yt(self, yt: str):
        self.yt = string_input(yt)
        self.tf = L(yt)
        self.yt = invL(self.tf)

    def define_tf(self, tf: str):
        self.tf = string_input(tf)
        self.yt = invL(self.tf)
        self.tf = L(self.yt)

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
        symbol = self.input
        print(f"Input:  {symbol}, is_real: {symbol.is_real}, is_positive: {symbol.is_positive}")

    def print_output(self):
        symbol = self.output
        print(f"Output  {symbol}, is_real: {symbol.is_real}, is_positive: {symbol.is_positive}")

    def print_tf(self):
        print(f"Transfer function: G(s) = {self.tf}")

    def print_all(self):
        self.print_symbols()
        self.print_variables()
        self.print_constants()
        self.print_tf()



    ## Plots

    def plot_time_response(self, t_range=(0, 10), num_points=1000):
        if self.yt is None:
            raise ValueError("Time-domain function f(t) is not defined.")

        yt = self.yt.subs(self.get_constant_values())
        t_vals, yt_vals = lambdify(yt, t_range, num_points)

        plt.figure()
        plt.plot(t_vals, yt_vals, label='y(t)')
        plt.title('Time Response')
        plt.xlabel('Time (s)')
        plt.ylabel('y(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def impulse(self, t_range=(0, 10), num_points=1000, delay_time=0):
        if self.tf is None:
            raise ValueError("Laplace-domain function F(s) is not defined.")

        impulse = impulse_function(delay_time)
        Y = self.tf*impulse
        yt = invL(Y).subs(self.get_constant_values())

        t_vals, yt_vals = lambdify(yt, t_range, num_points)

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
        
        step = step_function(delay_time)
        Y = self.tf*step
        yt = invL(Y).subs(self.get_constant_values())

        t_vals, yt_vals = lambdify(yt, t_range, num_points)

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

        ramp = ramp_function(delay_time)
        Y = self.tf*ramp
        yt = invL(Y).subs(self.get_constant_values())

        t_vals, yt_vals = lambdify(yt, t_range, num_points)

        plt.figure()
        plt.plot(t_vals, yt_vals, label='(t)')
        plt.title('Ramp Response')
        plt.xlabel('Time (s)')
        plt.ylabel('y(t)')
        plt.grid(True)
        plt.legend()
        plt.show()