import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermval
from scipy.misc import factorial
import os


class Solution(tf.keras.models.Model):
    def __init__(self, n_i, n_h, n_o=2, activation='sigmoid'):
        super(Solution, self).__init__()

        # Dimension of all the layers
        self.n_i = n_i
        self.n_h = n_h
        self.n_o = n_o

        # Shallow network
        # Hidden layer
        self.hidden_layer = tf.keras.layers.Dense(units=n_h, activation=activation,
                kernel_initializer=tf.initializers.GlorotUniform,
                bias_initializer=tf.initializers.GlorotUniform)
        # Output layer
        self.output_layer = tf.keras.layers.Dense(units=n_o, activation='linear', use_bias=False,
                kernel_initializer=tf.initializers.GlorotUniform,
                bias_initializer=tf.initializers.GlorotUniform)

    def call(self, X):
        # Conversion to a tensor
        X = tf.convert_to_tensor(X)

        # Simple Shallow Network Response
        response = self.hidden_layer(X)
        response = self.output_layer(response)

        response = tf.math.reduce_prod(response, axis=1)

        return response

    def train(self, X, loss_function, epochs, conditions, eigen_value, verbose=True,
                message_frequency=1, learning_rate=0.1, boundary_multiplier=10,
                optimizer_name='Adam'):
        
        # Checking for the right parameters
        if not isinstance(epochs, int) or epochs < 1:
            raise Exception('epochs parameter should be a positive integer.')
        if not isinstance(message_frequency, int) or message_frequency < 1:
            raise Exception(
                    'message_frequency parameter should be a positive integer.')
        
        # Choosing the optimizers
        optimizer = None
        if optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        
        def loss_boundary(network, conditions):
            loss = tf.constant(0., shape=(1,), dtype='float64')
            for condition in conditions:
                X = tf.convert_to_tensor(np.array([condition['value']]).reshape((1,1)))
                boundary_response = tf.reshape(network(X), shape=(-1,))
                boundary_value = condition['function'](X)
                boundary_value = tf.reshape(boundary_value, shape=(-1,))
                loss += (boundary_response - boundary_value) ** 2
            loss = boundary_multiplier*tf.math.reduce_sum(loss)
            return loss

        # Single train step function for the unsupervised equation part
        @tf.function
        def train_step(X, conditions, eigen_value):
            with tf.GradientTape() as tape:
                loss = loss_function(self, X, eigen_value)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(
                        zip(gradients, self.trainable_variables))
            with tf.GradientTape() as tape2:
                loss = loss_boundary(self, conditions)
            gradients = tape2.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(
                        zip(gradients, self.trainable_variables))
        
        # Training for a given number of epochs
        for epoch in range(epochs):
            train_step(X, conditions, eigen_value)
            equation_loss = loss_function(self, X, eigen_value)
            boundary_loss = loss_boundary(self, conditions)
            if verbose and(epoch+1) % message_frequency == 0:
                print(f'Epoch: {epoch+1} Loss equation: \
                    {equation_loss.numpy()} \
                    Loss boundary: {boundary_loss.numpy()}')

def train_test_domain_a_to_b(a=0, b=4, n=200):
    X_train = np.arange(a, b, (b-a)/n) + 1e-8
    X_test = np.arange(a, b, (b-a)/n/10) + 1e-8
    return  X_train.reshape(-1,1), X_test.reshape(-1,1)

def constant(c):
    def func(X):
        return tf.constant(c, dtype='float64', shape=X.shape)
    return func

def loss_well(network, X, eigen_value):
    X = tf.convert_to_tensor(X)
    # Taking the frist (grads) and the second (laplace) derivatives w. r. t. inputs
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape1.watch(X)
            tape2.watch(X)
            response = network(X)
        grads = tape2.gradient(response, X)
    laplace = tape1.gradient(grads, X)
    
    psi = tf.reshape(response, shape=(-1,))  
    nabla = tf.reshape(laplace, shape=(-1,))
    loss = (nabla + eigen_value * psi) ** 2
    
    return tf.math.reduce_mean(loss)

def loss_well_unity(network, X, eigen_value):
    X = tf.convert_to_tensor(X)
    # Taking the frist (grads) and the second (laplace) derivatives w. r. t. inputs
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape1.watch(X)
            tape2.watch(X)
            response = network(X)
        grads = tape2.gradient(response, X)
    laplace = tape1.gradient(grads, X)
    
    psi = tf.reshape(response, shape=(-1,))  
    nabla = tf.reshape(laplace, shape=(-1,))
    loss = (nabla + eigen_value * psi) ** 2
    
    interval = X[1]-X[0]
    probability_unity = tf.math.reduce_sum(response**2) * interval 
    probability_unity = (probability_unity - 1) **2
    loss = tf.math.reduce_mean(loss) + probability_unity
    return loss

def well_analytic(X, n, L, **kwargs):
    Y = np.sin(n*np.pi/L*X)/np.sqrt(2)
    return Y

def zero_boundary_conditions(a, b):
    bcs = [{'variable':0, 'value':a, 'type':'dirichlet',
            'function':constant(0.)},
        {'variable':0, 'value':b, 'type':'dirichlet',
            'function':constant(0.)}]
    return bcs

def eigen_value_well(n, L, **kwargs):
    return (n * np.pi / L)**2

def loss_harmonic(network, X, eigen_value):
    X = tf.convert_to_tensor(X)
  
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape1.watch(X)
            tape2.watch(X)
            response = network(X)
        grads = tape2.gradient(response, X)
    laplace = tape1.gradient(grads, X)
    
    nabla = tf.reshape(laplace, shape=(-1,))
    psi = tf.reshape(response, shape=(-1,))
    x = tf.reshape(X, shape=(-1,))
    
    eigen_value_tensor = tf.constant(eigen_value, shape=(X.shape[0], ), dtype='float64')
    
    loss = tf.square(0.5*nabla + (eigen_value_tensor - 0.5*x** 2) * psi)
    
    return tf.math.reduce_mean(loss)

def loss_harmonic_unity(network, X, eigen_value):
    X = tf.convert_to_tensor(X)
  
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape1.watch(X)
            tape2.watch(X)
            response = network(X)
        grads = tape2.gradient(response, X)
    laplace = tape1.gradient(grads, X)
    
    nabla = tf.reshape(laplace, shape=(-1,))
    psi = tf.reshape(response, shape=(-1,))
    x = tf.reshape(X, shape=(-1,))
    
    eigen_value_tensor = tf.constant(eigen_value, shape=(X.shape[0], ), dtype='float64')
    
    loss = tf.square(0.5*nabla + (eigen_value_tensor - 0.5*x** 2) * psi)
    
    interval = X[1]-X[0]
    probability_unity = tf.math.reduce_sum(response**2) * interval 
    probability_unity = (probability_unity - 1) **2
    loss = tf.math.reduce_mean(loss) + probability_unity
    return loss

def harmonic_analytic(X, n, **kwargs):
    # degree parameter of the polynomial
    c = np.zeros((n+1))
    c[n] = 1
    value = hermval(X, c)
    value *= np.power(np.pi, -0.25)
    value /= np.sqrt(2 ** n * factorial(n))
    value *= np.exp(-0.5*X**2)
    return value

def eigen_value_harmonic(n, **kwargs):
    return n + 0.5

def train_plot_save(loss_function, eigen_value_function, analytic_solution, name, n, id, domain, learning_rate=0.001, n_h=105,
    epochs=100000):
    sol = Solution(n_i=1, n_h=n_h, n_o=2)
    X_train, X_test = train_test_domain_a_to_b(*domain, 200)
    bcs = zero_boundary_conditions(*domain)
    a = domain[0]
    b = domain[1]
    sol.train(X=X_train, conditions=bcs, eigen_value=eigen_value_function(n, L=b-a), loss_function=loss_function,
            epochs=epochs, verbose=False, boundary_multiplier=0.1,
            learning_rate=learning_rate, optimizer_name='Adam')
    y_train = sol(tf.convert_to_tensor(X_train)).numpy()
    train_normalization = (y_train**2).sum()*(b-a)/y_train.shape[0]
    y_train /= train_normalization
    y_test = sol(tf.convert_to_tensor(X_test)).numpy()
    test_normalization = (y_test**2).sum()*(b-a)/y_test.shape[0]
    y_test /= test_normalization
    plt.clf()
    plt.scatter(X_train, y_train, c='r', label='Numerical - Training', marker='x', s=800)
    plt.plot(X_test, y_test, c='xkcd:sky blue', label='Numerical - Test', marker='x', linewidth=5)
    plt.plot(X_test, analytic_solution(X_test, n, L=(b-a)), c='xkcd:goldenrod', label='Analytic', linewidth=5)
    plt.xlabel(r'$x$', fontsize='50')
    plt.xlim((a,b))
    plt.legend(fontsize='40')
    plt.title(f'{name} for n={n}', fontsize='60')
    plt.gcf().set_size_inches(30, 22.5)
    plt.tick_params(axis='both', which='major', labelsize=35)
    path = os.path.join('plots', name)
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, str(n))
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, str(id)), format='pdf')
