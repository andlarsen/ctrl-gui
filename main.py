from classes.control_environment import ControlEnvironment

ce = ControlEnvironment('test')

def main():

    ce.add_tf('G1')
    ce.tfs['G1'].define_input('u')
    ce.tfs['G1'].define_output('y')
    ce.tfs['G1'].add_constant('a',1)
    ce.tfs['G1'].add_constant('b',2)
    ce.tfs['G1'].add_constant('c',3)
    ce.tfs['G1'].add_constant('w_n',10)
    ce.tfs['G1'].add_constant('zeta',0.4)
    # ce.tfs['G1'].define_tf(lhs='ddot(y)+2*zeta*w_n*dot(y)+w_n**2*y',rhs='w_n**2*u')
    ce.tfs['G1'].define_tf('1/(s**2 + 2*w_n*zeta*s + w_n**2)')
    # ce.tfs['G1'].define_tf('1/(a*s**2+b*s+c)')
    # ce.tfs['G1'].define_tf('1/(a*s**2+b*s+c)')
    # ce.tfs['G1'].define_tf(num_coefs=[1],den_coefs=['1','2*w_n*zeta','w_n**2'])
    # ce.tfs['G1'].define_tf(num_coefs=[1],den_coefs=['1','1'])
    # ce.tfs['G1'].define_tf('12/((s+100)*(s**2+2.8*s+4))')
    # ce.tfs['G1'].define_tf('(s + 2) / ( s * (0.1*s + 1) * (0.02*s + 1) )')
    # ce.tfs['G1'].define_tf('(s + 1) / ( s * (0.5*s + 1) * (0.05*s + 1) * (0.01*s + 1) )')
    # ce.tfs['G1'].define_tf('12/((s+1)*(s**2+2.8*s+4))')
    # ce.tfs['G1'].define_tf('11/6*60/(s*(s+1)*(s+10))')
    ce.tfs['G1'].print_all()
    
    # ce.tfs['G1'].impulse()
    # ce.tfs['G1'].step()
    # ce.tfs['G1'].ramp()
    # ce.tfs['G1'].bode()
    # ce.tfs['G1'].margin_plot()
    # ce.tfs['G1'].pzmap()
    # ce.tfs['G1'].nyquist()

    # ce.root_locus()

if __name__ == "__main__":
    main()