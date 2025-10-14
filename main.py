from classes.control_system import ControlSystem

cs = ControlSystem('test')

def main():

    cs.add_global_constant('Kg',1)

    cs.add_tf(name='G1')
    cs.components.tfs['G1'].define_input('u1')
    cs.components.tfs['G1'].define_output('y1')
    cs.components.tfs['G1'].add_constant('K',1)
    cs.components.tfs['G1'].add_constant('w_n',10)
    cs.components.tfs['G1'].add_constant('zeta',0.4)
    cs.components.tfs['G1'].define_tf(lhs='ddot(y)+2*zeta*w_n*dot(y)+w_n**2*y',rhs='K*Kg*w_n**2*u')
    cs.components.tfs['G1'].print_all()
    # cs.components.tfs['G1'].bode(w_range=(0.1,1000),sweep_params={'zeta':[0.1,0.5,1]})
    
    cs.add_tf(name='G2')
    cs.components.tfs['G2'].define_input('u2')
    cs.components.tfs['G2'].define_output('y2')
    cs.components.tfs['G2'].add_constant('a',1)
    cs.components.tfs['G2'].add_constant('b',2)
    cs.components.tfs['G2'].add_constant('c',3)
    cs.components.tfs['G2'].define_tf('Kg/(a*s**2+b*s+c)')
    cs.components.tfs['G1'].print_all()
    # cs.components.tfs['G2'].impulse(t_range=(-1,15),delay_time=0,sweep_params={'a':[1, 2, 3]})
    # cs.components.tfs['G2'].step(t_range=(-1,15),delay_time=1,sweep_params={'a':[1, 2, 3]})
    # cs.components.tfs['G2'].ramp(t_range=(-1,15),delay_time=1,sweep_params={'a':[1, 2, 3]})
    cs.components.tfs['G2'].bode(sweep_params={'a':[1, 2, 3]})

    cs.add_tf(name='G3')
    cs.components.tfs['G3'].define_input('u3')
    cs.components.tfs['G3'].define_output('y3')
    cs.components.tfs['G3'].add_constant('w_n',10)
    cs.components.tfs['G3'].add_constant('zeta',0.4)
    cs.components.tfs['G3'].define_tf('Kg*w_n**2/(s**2+2*zeta*w_n*s+w_n**2)')
    cs.components.tfs['G1'].print_all()
    
    # cs.impulse(t_range=(0, 8), sweep_params={'Kg': [1, 2]}) 
    # cs.step(t_range=(0, 8), sweep_params={'Kg': [1, 2]}) 
    # cs.ramp(t_range=(0, 8), sweep_params={'Kg': [1, 2]}) 
    cs.bode(w_range=(0.1, 1000), sweep_params={'Kg': [1, 2]}) 

    ## TODOS
    # cs.root_locus()
    # more components
    # how to connect components?
    # how to simulate?

if __name__ == "__main__":
    main()