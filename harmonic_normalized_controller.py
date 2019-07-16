import naive_nde

for n in range(0, 1):
    for id in range(10):
        print('Starting', n, id)
        naive_nde.train_plot_save(loss_function=naive_nde.loss_harmonic, 
                                    eigen_value_function=naive_nde.eigen_value_harmonic,
                                    analytic_solution=naive_nde.harmonic_analytic,
                                    dir='Harmonic_Oscillator_Normalized',
                                    name='Harmonic Oscillator',
                                    n=n, id=id, domain=(-4,4.), learning_rate=0.001,
                                    normalize=True
                                )
        print('Done', n,id)
