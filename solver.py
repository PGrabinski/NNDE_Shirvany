class Solver:
  def __init__(self, input_dim=1, hidden_dim=1, tolerance=epsilon, max_iter=max_iterations,
               max_eigen_counter=1e2, delta_E = 1e-1):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.tolerance = tolerance
    self.max_iter = max_iter
    self.max_eig_counter = max_eigen_counter
    self.sn = ShallowNetwork(input_dim, hidden_dim=1)
    self.value_E = 0
    self.delta_E = delta_E
    
  def solve(self, dataset_X, BC_set):
    eigenvalue_counter = 0
    iteration_number = 0
    loss = self.sn.loss_function_all(dataset_X, BC_set, self.value_E)
    while loss > self.tolerance:
      if iteration_number < self.max_iter:
        self.sn.train(dataset_X, BC_set, self.value_E)
        iteration_number += 1
      else:
        self.value_E += self.delta_E
        iteration_number = 0
        if eigenvalue_counter < self.max_eig_counter:
          eigenvalue_counter += 1
        else:
          self.hidden_dim += 1
          self.sn = ShallowNetwork(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
          eigenvalue_counter = 0
      loss = self.sn.loss_function_all(dataset_X, BC_set, self.value_E)
    return self.value_E
