import tensorflow as tf
import numpy as np
import naive_nde as nnde

class Solver:
  def __init__(self, n_i=1., n_h=1., tolerance=1e-8, max_iter=10000,
               max_eigen_counter=1e4, delta_E = 1e-4):
    self.n_i = n_i
    self.n_h = n_h
    self.tolerance = tolerance
    self.max_iter = max_iter
    self.max_eig_counter = max_eigen_counter
    self.sol = nnde.Solution(n_i, n_h=self.n_h)
    self.eigen_value = 0
    self.delta_E = delta_E
    
  def solve(self, X, conditions, loss_function,
            message_frequency=1, learning_rate=0.1,
            optimizer='Adam', verbose=False, boundary_multiplier=1):
    eigenvalue_counter = 0
    loss = 1e6
    while loss > self.tolerance:
      loss = self.sol.train(X=X, conditions=conditions, eigen_value=self.eigen_value,
                       loss_function=loss_function,
                       epochs=self.max_iter, message_frequency=message_frequency, learning_rate=0.01,
                       optimizer_name=optimizer, verbose=verbose)
      print(f'Loss: {loss} for n: {self.eigen_value}')
      if loss < self.tolerance:
        print('break')
        break
      else:
        self.eigen_value += self.delta_E
        if eigenvalue_counter < self.max_eig_counter:
          eigenvalue_counter += 1
          print(f'Rising the eigen value number {eigenvalue_counter}')
        else:
          print('Need more power!')
          self.n_h += 1.
          self.sol = nnde.Solution(n_i=self.n_i, n_h=self.n_h)
          eigenvalue_counter = 0
      loss = 1e6

    print(eigenvalue_counter, loss)
    return self.eigen_value, self.sol