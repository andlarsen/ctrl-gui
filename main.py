from classes.control_environment import ControlEnvironment

import sympy as sp
from sympy import *
init_printing(use_latex='mathjax')

ce = ControlEnvironment('test')

def main():
    ce.define_variable('x')
    ce.define_constant('a', value = 1)
    ce.define_constant('M', value = 20)
    ce.define_constant('B', value = 5)
    ce.define_constant('K', value = 100)
    ce.define_constant('Kp', value = 300)
    ce.define_constant('Ki', value = 70)
    # ce.define_f('t**2')
    # ce.define_F('1/(M*s**2 + B*s + K)')
    # ce.define_F('11/6*60/(s*(s+1)*(s+10))')
    ce.define_F('12/((s+1)*(s**2+2.8*s+4))')
    # ce.define_F('sqrt(2)/(s+1)')
    # ce.print()
    # pprint(f"diff(f(t)) = {sp.diff(ce.f, ce.get_symbol('t'))}")
    # pprint(f"integrate(f(t)) = {sp.integrate(ce.f, ce.get_symbol('t'))}")

    # ce.plot_time_response()
    # ce.poles()
    # ce.zeros()
    # ce.impulse()
    # ce.step()
    # ce.ramp()
    # ce.bode()
    # ce.margin()
    # ce.nyquist()
    # ce.root_locus()
    # ce.margin_plot()

if __name__ == "__main__":
    main()