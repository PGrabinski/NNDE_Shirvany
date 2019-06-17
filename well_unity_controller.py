import naive_nde

for n in range(3, 7):
    for id in range(20):
        print('Starting', n, id)
        naive_nde.train_plot_save(loss_function=naive_nde.loss_well_unity, 
                                    eigen_value_function=naive_nde.eigen_value_well,
                                    analytic_solution=naive_nde.well_analytic,
                                    name='Quantum Potential Well',
                                    dir='Quantum_Potential_Well_Probability_Loss',
                                    n=n, id=id, domain=(0,4.), learning_rate=0.001,
                                    normalize=False, probability_weight=10
                                )
        print('Done', n, id)
