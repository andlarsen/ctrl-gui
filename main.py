from classes.control_environment import ControlEnvironment

import sympy as sp
import matplotlib
matplotlib.use('Qt5Agg')

ce = ControlEnvironment('test')

def main():
    ce.define_variable('x')
    ce.define_constant('M', value = 2)
    ce.define_constant('B', value = 5)
    ce.define_constant('K', value = 100)
    # ce.define_f('t*exp(-a*t)')
    ce.define_F('1/(M*s**2 + B*s + K)')
    ce.print()

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