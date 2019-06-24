import naive_nde as nnde
import solver
import time
X_train, X_test = nnde.train_test_domain_a_to_b(0, 4.)
solving_machine = solver.Solver(tolerance=1e-5, delta_E=1e-2, max_iter=10000, max_eigen_counter=1e3, n_h=100)
eigen_value, network = solving_machine.solve(X=X_train,
    conditions=nnde.zero_boundary_conditions(0, 4.), loss_function=nnde.loss_well_unity, learning_rate=0.001)
timestamp = time.time()
network.save_weights(f'./{timestamp}.h5')
with open(f'{timestamp}.dat', "w") as text_file:
    print(f'The eigen value is {eigen_value}', file=text_file)