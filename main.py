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
    # cs.components.tfs['G1'].define_tf(lhs='ddot(y)+2*zeta*w_n*dot(y)+w_n**2*y',rhs='K*Kg*w_n**2*u')
    # cs.components.tfs['G1'].define_tf(num_coefs=['Kg'],den_coefs=[1,10])
    cs.components.tfs['G1'].define_tf('Kg/(s+1)')
    cs.components.tfs['G1'].impulse()
    cs.components.tfs['G1'].step()
    cs.components.tfs['G1'].ramp()
    cs.components.tfs['G1'].print_all()
    # cs.components.tfs['G1'].bode()
    # cs.components.tfs['G1'].bode(w_range=(0.1,1000),sweep_params={'zeta':[0.1,0.5,1]})
    
    # cs.add_tf(name='G2')
    # cs.components.tfs['G2'].define_input('u2')
    # cs.components.tfs['G2'].define_output('y2')
    # cs.components.tfs['G2'].add_constant('a',1)
    # cs.components.tfs['G2'].add_constant('b',2)
    # cs.components.tfs['G2'].add_constant('c',3)
    # cs.components.tfs['G2'].add_constant('K',2)
    # cs.components.tfs['G2'].define_tf('Kg*K/(a*s**2+b*s+c)')
    # cs.components.tfs['G2'].define_tf(num_coefs=[1],den_coefs=[1,1])
    # cs.components.tfs['G2'].print_all()
    # cs.components.tfs['G2'].impulse(t_range=(-1,15),delay_time=0,sweep_params={'a':[1, 2, 3]})
    # cs.components.tfs['G2'].step(t_range=(-1,15),delay_time=1,sweep_params={'a':[1, 2, 3]})
    # cs.components.tfs['G2'].ramp(t_range=(-1,15),delay_time=1,sweep_params={'a':[1, 2, 3]})
    # cs.components.tfs['G2'].bode(sweep_params={'a':[1, 2, 3]})

    # cs.add_tf(name='Gol')
    # cs.components.tfs['Gol'].open_loop('G1','G2')
    # cs.components.tfs['Gol'].print_all()
    # cs.components.tfs['Gol'].bode()

    cs.add_tf(name='Gp',description='Plant')
    cs.components.tfs['Gp'].define_input('f')
    cs.components.tfs['Gp'].define_output('x')

    cs.components.tfs['Gp'].add_constant(   name        = 'K',
                                            value       = 1,
                                            description = 'Gain',
                                            unit        = '-'       )
    
    cs.components.tfs['Gp'].add_constant(   name        = 'w_n',
                                            value       = 0.1,
                                            description = 'Natural frequency',
                                            unit        = 'rad/s'   )
    
    cs.components.tfs['Gp'].add_constant(   name        = 'zeta',
                                            value       = 1,
                                            description = 'Damping coefficient',
                                            unit        = '-'                )
    
    cs.components.tfs['Gp'].define_tf('Kg*K*w_n**2/(s**2+2*zeta*w_n*s+w_n**2)')
    # cs.components.tfs['Gp'].print_all()
    # cs.components.tfs['Gp'].ramp()

    # cs.impulse(sweep_params={'Kg': [1, 2]})

    # cs.add_tf(name='Gc',description='Controller')
    # cs.components.tfs['Gc'].define_input('u')
    # cs.components.tfs['Gc'].define_output('f')
    # cs.components.tfs['Gc'].add_constant('Kp',1)
    # cs.components.tfs['Gc'].add_constant('Ki',3)
    # cs.components.tfs['Gc'].add_constant('Kd',0)
    # cs.components.tfs['Gc'].define_tf('Kp+1/s*Ki+s*Kd')
    # cs.components.tfs['Gc'].print_all()

    # cs.add_tf(name='Gcl')
    # cs.components.tfs['Gcl'].closed_loop('Gc','Gp')
    # cs.components.tfs['Gcl'].print_all()
    # cs.components.tfs['Gcl'].step()
    # cs.components.tfs['Gcl'].step(sweep_params={"Kp":[0,0.5,1],"Ki":[1,3],"Kd":[0]})

    

    # cs.bode('G1','G2','Gol')

    # cs.impulse(t_range=(0, 8), sweep_params={'Kg': [1, 2]}) 
    # cs.step(t_range=(0, 8), sweep_params={'Kg': [1, 2]}) 
    # cs.ramp(t_range=(0, 8), sweep_params={'Kg': [1, 2]}) 
    # cs.bode(w_range=(0.1, 1000), sweep_params={'Kg': [1, 2]}) 

    ## TODOS
    # Fix sweep_tfs??
    # make get_impulse_response auto logic
    # make get_ramp_response auto logic
    # make get_frequency_reqsponse + auto logic + models
    # cs.root_locus()
    # more components
    # how to connect components?
    # how to simulate?
    # input delay_time and time_range to impulse, steps etc, update_tf() to make new responses, and plot
    # print ramp response info: if no steady state error, print it!
    # change title depending on response type
    

if __name__ == "__main__":
    main()