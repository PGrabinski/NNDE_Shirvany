import naive_nde

for n in range(1, 7):
    for id in range(10):
        print('Starting', n, id)
        naive_nde.train_plot_save(loss_function=naive_nde.loss_well, 
                                    eigen_value_function=naive_nde.eigen_value_well,
                                    analytic_solution=naive_nde.well_analytic,
                                    name='Quantum Potential Well',
                                    dir='Quantum_Potential_Well_Normalized',
                                    n=n, id=id, domain=(0,4.), learning_rate=0.001,
                                    normalize=True
                                )
        print('Done', n,id)
