from classes.control_environment import ControlEnvironment

ce = ControlEnvironment('test')

def main():

    ce.add_tf('G1')
    ce.tfs['G1'].add_constant('a',1)
    ce.tfs['G1'].add_constant('b',2)
    ce.tfs['G1'].add_constant('c',3)
    ce.tfs['G1'].define_tf('1/(a*s**2+b*s+c)')
    ce.tfs['G1'].define_tf_coefs([1],['a','b','c'])
    ce.tfs['G1'].impulse()
    ce.tfs['G1'].print_all()
    
    # ce.tfs['eq1'].define_tf('1/(s+a)')
    # ce.tfs['eq1'].plot_time_response()
    # ce.tfs['eq1'].impulse()
    # ce.tfs['G1'].step()
    # ce.tfs['eq1'].ramp()
    # ce.tfs['eq1'].print_all()
    # ce.tfs['eq1'].poles(print_poles=True)

    # ce.add_function('eq2')
    # ce.tfs['eq2'].define_tf('(s+12)/(s+1)')
    # ce.tfs['eq2'].define_tf('12/((s+1)*(s**2+2.8*s+4))')
    # ce.tfs['eq2'].step()
    # ce.tfs['eq2'].poles(print_poles=True)
    # ce.tfs['eq2'].zeros(print_zeros=True)

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