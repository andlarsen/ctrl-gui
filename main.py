from classes.control_environment import ControlEnvironment

ce = ControlEnvironment('test')

def main():

    ce.add_tf(name='G1')
    ce.tfs['G1'].define_input('u1')
    ce.tfs['G1'].define_output('y1')
    ce.tfs['G1'].add_constant('w_n',10)
    ce.tfs['G1'].add_constant('zeta',0.4)
    ce.tfs['G1'].define_tf(lhs='ddot(y)+2*zeta*w_n*dot(y)+w_n**2*y',rhs='w_n**2*u')
    ce.tfs['G1'].print_all()
    ce.tfs['G1'].ramp()
    
    ce.add_tf(name='G2')
    ce.tfs['G2'].define_input('u2')
    ce.tfs['G2'].define_output('y2')
    ce.tfs['G2'].add_constant('a',1)
    ce.tfs['G2'].add_constant('b',2)
    ce.tfs['G2'].add_constant('c',3)
    ce.tfs['G2'].define_tf('1/(a*s**2+b*s+c)')
    ce.tfs['G2'].print_all()

    ce.add_tf(name='G3')
    ce.tfs['G3'].define_input('u3')
    ce.tfs['G3'].define_output('y3')
    ce.tfs['G3'].add_constant('w_n',15)
    ce.tfs['G3'].add_constant('zeta',3)
    ce.tfs['G3'].define_tf('w_n**2/(s**2+2*zeta*w_n*s+w_n**2)')
    ce.tfs['G3'].print_all()

    ce.ramp()

    ## TODOS
    # ce.root_locus()

if __name__ == "__main__":
    main()