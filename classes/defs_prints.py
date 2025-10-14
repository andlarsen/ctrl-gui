import sympy as sp

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

def print_differential_equation(self):
    print(f"Equation:")
    print(f"Symbolic:   {self.differential_equation.symbolic.lhs} = {self.differential_equation.symbolic.rhs}")
    print(f"Numeric:    {self.differential_equation.numeric.lhs} = {self.differential_equation.numeric.rhs}")

def print_tf(self):
    if self.tf.symbolic is None:
        print("Laplace-domain function F(s) is not defined.")
        return
    print(f"Transfer function:")
    print(f"Symbolic:   G(s) = {self.tf.symbolic}")
    print(f"Numeric:    G(s) = {self.tf.numeric}")

def print_numerator(self):
    print(f"Numerator:")
    print(f"Symbolic:   N(s) = {self.numerator.symbolic}")
    print(f"Numeric:    N(s) = {self.numerator.numeric}")

def print_denominator(self):
    print(f"Denominator:")
    print(f"Symbolic:   D(s) = {self.denominator.symbolic}")
    print(f"Numeric:    D(s) = {self.denominator.numeric}")

def print_zeros(self):
    print("Zeros: (symbolic)")
    for z in self.zeros.symbolic:
        print(f"  {sp.simplify(z)}")
    print("Zeros: (numeric)")
    for z in self.zeros.numeric:
        print(f"  {z}")

def print_poles(self):
    print("Poles: (symbolic)")
    for p in self.poles.symbolic:
        print(f"  {sp.simplify(p)}")
    print("Poles: (numeric)")
    for p in self.poles.numeric:
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
    print_symbols(self)
    print('')
    print_variables(self)
    print('')
    print_constants(self)
    print('')
    print("==================================================")
    print_input(self)
    print('')
    print_output(self)
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