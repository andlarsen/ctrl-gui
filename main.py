from classes.control_environment import ControlEnvironment

ce = ControlEnvironment('test')

def main():

    ce.add_function('eq1')
    ce.functions['eq1'].add_constant('a',1)
    ce.functions['eq1'].add_constant('b',2)
    ce.functions['eq1'].add_constant('c',3)
    ce.functions['eq1'].define_yt('a*t**2+b*t+c')
    # ce.functions['eq1'].define_tf('1/(s+a)')
    ce.functions['eq1'].plot_time_response()
    # ce.functions['eq1'].impulse()
    ce.functions['eq1'].step()
    # ce.functions['eq1'].ramp()
    ce.functions['eq1'].print_all()

    ce.add_function('eq2')
    ce.functions['eq2'].define_tf('12/((s+1)*(s**2+2.8*s+4))')
    ce.functions['eq2'].step()

    # ce.define_variable('x')
    # ce.define_constant('a', value = 1)
    # ce.define_constant('M', value = 20)
    # ce.define_constant('B', value = 5)
    # ce.define_constant('K', value = 100)
    # ce.define_constant('Kp', value = 300)
    # ce.define_constant('Ki', value = 70)
    # # ce.define_f('t**2')
    # ce.define_F('1/(M*s**2 + B*s + K)')
    # ce.define_F('11/6*60/(s*(s+1)*(s+10))')
    # ce.define_F('12/((s+1)*(s**2+2.8*s+4))')
    # ce.define_F('12/((s+1)*(s**2+2.8*s+4))+(s+1)')
    # ce.define_F('sqrt(2)/(s+1)')
    # define an F(s) with zeros and poles
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
    # ce.pzmap()

if __name__ == "__main__":
    main()