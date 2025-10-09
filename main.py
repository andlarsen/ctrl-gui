from classes.control_environment import ControlEnvironment

import sympy as sp
from sympy import *
init_printing(use_latex='mathjax')

ce = ControlEnvironment('test')

def main():
    ce.define_variable('x')
    ce.define_constant('a', value = 1)
    ce.define_constant('M', value = 2)
    ce.define_constant('B', value = 5)
    ce.define_constant('K', value = 100)
    ce.define_f('t**2')
    # ce.define_F('1/(M*s**2 + B*s + K)')
    ce.print()
    pprint(f"diff(f(t)) = {sp.diff(ce.f, ce.get_symbol('t'))}")
    pprint(f"integrate(f(t)) = {sp.integrate(ce.f, ce.get_symbol('t'))}")

    # ce.plot_time_response()
    ce.poles()
    ce.zeros()
    ce.impulse()
    ce.step()
    ce.ramp()
    ce.bode()
    ce.margin()
    ce.nyquist()
    ce.root_locus()

if __name__ == "__main__":
    main()