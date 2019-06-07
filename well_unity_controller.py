import naive_nde

for n in range(1, 7):
    for id in range(10):
        naive_nde.train_plot_save(loss_function=naive_nde.loss_well_unity, 
                                    eigen_value_function=naive_nde.eigen_value_well,
                                    analytic_solution=naive_nde.well_analytic,
                                    name='Quantum Potential Well',
                                    n=n, id=id, domain=(0,4.)
                                )
        print(f'Done n={n} with id={id}')