import naive_nde

for n in range(1, 3):
    for id in range(5):
        print('Starting', n, id)
        naive_nde.train_plot_save(loss_function=naive_nde.loss_well, 
                                    eigen_value_function=naive_nde.eigen_value_well,
                                    analytic_solution=naive_nde.well_analytic,
                                    name='Quantum Potential Well',
                                    n=n, id=id, domain=(0,4.), learning_rate=0.001
                                )
        print('Done', n,id)
