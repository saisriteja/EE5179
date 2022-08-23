import numpy as np

################################################
#         Activations
################################################
class Sigmoid():
    def __init__(self, c=1, b=0):
        self.c = c
        self.b = b

    def value(self, x):
        val = 1 + np.exp(-self.c*(x + self.b))
        return 1/val

    def diff(self, x, remove=False):
        y = self.value(x)
        if remove==True:
            y = y[:-1,:]
        val = self.c*y*(1-y)
        return val

class Tanh():
    def __init__(self):
        pass

    def value(self, x):
        num = np.exp(x) - np.exp(-x)
        denom = np.exp(x) + np.exp(-x)
        return num/denom

    def diff(self, x):
        y = self.value(x)
        val = 1 - y**2
        return val

class Relu():
    def __init__(self):
        pass

    def value(self, x):
        val = x
        val[val<0] = 0
        return val

    def diff(self, x):
        val = np.ones(x.shape)
        val[val<=0] = 0
        return val

class Softmax():
    def __init__(self):
        pass

    def value(self, x):
        val = np.exp(x)/np.sum(np.exp(x), axis=0)
        # print("X shape", x.shape, "Val shape", val.shape)
        return val

    def diff(self, x):
        y = self.value(x)
        # print("Y shape:", y.shape)
        # Refernce for checking examples:
        # https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
        mat = np.tile(y, y.shape[0])
        val = np.diag(y.reshape(-1,)) - (mat*mat.T)
        return val




import math
import wandb
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy


map_optimizers = {"Normal":Normal(), "Momentum":Momentum(), "Nesterov":Nesterov(), "AdaGrad":AdaGrad(), "RMSProp":RMSProp(), "Adam":Adam(), "Nadam":Nadam()}
map_losses = {"SquaredError":SquaredError(), "CrossEntropy":CrossEntropy()}
################################################
#         Network
################################################
class NeuralNetwork():
    def __init__(self, layers, batch_size, optimizer, initialization, epochs, t, loss, X_val=None, t_val=None, use_wandb=False, optim_params=None):
        self.layers = layers
        self.batch_size = batch_size
        self.initialization = initialization
        self.epochs = epochs
        self.optimizer = optimizer
        self.t = t
        self.num_batches = math.ceil(self.t.shape[1]/batch_size)
        self.loss_type = loss
        self.loss = map_losses[loss]
        self.use_wandb = use_wandb
        if t_val is not None:
            self.X_val = X_val
            self.layers[0].a_val = X_val
            self.t_val = t_val
        self.param_init(optimizer, optim_params)

    def param_init(self, optimizer, optim_params):
        size_prev = self.layers[0].size
        for layer in self.layers[1:]:
            # layer.W_size = (layer.size, size_prev+1)
            layer.W_size = (layer.size, size_prev)
            size_prev = layer.size
            layer.W_optimizer = deepcopy(map_optimizers[optimizer])
            layer.b_optimizer = deepcopy(map_optimizers[optimizer])
            # Code to set params
            if optim_params:
                layer.W_optimizer.set_params(optim_params)
                layer.b_optimizer.set_params(optim_params)

        if self.initialization == "RandomNormal":
            for layer in self.layers[1:]:
                layer.W = np.random.normal(loc=0, scale=1.0, size = layer.W_size)
                layer.b = np.zeros((layer.W_size[0], 1))

        elif self.initialization == "XavierUniform":
            for layer in self.layers[1:]:
                initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)#, seed=42)
                layer.W = np.array(initializer(shape=layer.W_size))
                layer.b = np.zeros((layer.W_size[0], 1))

        elif self.initialization == "Test":
            for layer in self.layers[1:]:
                layer.W = np.ones(layer.W_size)*0.5
                layer.b = np.zeros((layer.W_size[0], 1))


    def forward_propogation(self):
        for i in range(1, len(self.layers)):
            # print("Layer:", i, self.layers[i].W.shape)
            # Pre-activation
            self.layers[i].h = self.layers[i].W @ self.layers[i-1].a - self.layers[i].b
            # Activation
            self.layers[i].a = self.layers[i].activation.value(self.layers[i].h)
            # Validation
            self.layers[i].h_val = self.layers[i].W @ self.layers[i-1].a_val - self.layers[i].b
            self.layers[i].a_val = self.layers[i].activation.value(self.layers[i].h_val)
        
        if self.loss_type == "CrossEntropy":
            # Final sofmax activation
            self.layers[-1].y = Softmax().value(self.layers[-1].a)
            self.layers[-1].y_val = Softmax().value(self.layers[-1].a_val)
        else:
            self.layers[-1].y = self.layers[-1].a
            self.layers[-1].y_val = self.layers[-1].a_val

    def check_test(self, X_test, t_test):
        self.layers[0].a_test = X_test
        for i in range(1, len(self.layers)):
            self.layers[i].h_test = self.layers[i].W @ self.layers[i-1].a_test - self.layers[i].b
            self.layers[i].a_test = self.layers[i].activation.value(self.layers[i].h_test)

        if self.loss=="CrossEntropy":
            self.layers[-1].y_test = Softmax().value(self.layers[-1].a_test)
        else:
            self.layers[-1].y_test = self.layers[-1].a_test

        loss_test = self.loss.calc_loss(t_test, self.layers[-1].y_test)

        encoder = OneHotEncoder()
        y_tmp = encoder.inverse_transform(self.layers[-1].y_test)
        t_tmp = encoder.inverse_transform(t_test)
        acc_test = np.sum(y_tmp==t_tmp)
        return acc_test, loss_test, y_tmp


    def backward_propogation(self):
        # Initialize variables neesed to keep track of loss
        self.eta_hist = []
        self.loss_hist = []
        self.accuracy_hist = []
        self.loss_hist_val = []
        self.accuracy_hist_val = []
        self.loss = SquaredError()
        flag = 0

        # Perform Backprop
        # for _ in range(self.epochs):
        for ep in tqdm(range(self.epochs)):
            self.eta_hist.append(self.layers[-1].W_optimizer.eta)
            self.loss_hist.append(self.loss.calc_loss(self.t, self.layers[-1].y))
            train_acc, val_acc = self.get_accuracy(validation=True)
            self.accuracy_hist.append(train_acc)
            self.loss_hist_val.append(self.loss.calc_loss(self.t_val, self.layers[-1].y_val))
            self.accuracy_hist_val.append(val_acc)

            if self.use_wandb:
                wandb.log({
                            "step": ep, \
                            "loss:": self.loss_hist[-1]/self.t.shape[1], \
                            "accuracy": self.accuracy_hist[-1]/self.t.shape[1], \
                            "val_loss": self.loss_hist_val[-1]/self.t_val.shape[1], \
                            "val_accuracy": self.accuracy_hist_val[-1]/self.t_val.shape[1]
                        })
            
            for batch in range(self.num_batches):
                # print("\n", "="*50)
                # print("Batch:", batch)
                # X_batch = self.layers[0].input[batch*self.batch_size:(batch+1)*self.batch_size]
                t_batch = self.t[:, batch*self.batch_size:(batch+1)*self.batch_size]
                y_batch = self.layers[-1].y[:, batch*self.batch_size:(batch+1)*self.batch_size]
                self.y_batch = y_batch
                self.t_batch = t_batch

                # Calculate Loss, grad wrt y and softmax for last layer
                # print("t:\n", self.t)
                # print("y:\n", self.layers[-1].y)
                # print(self.loss_hist[-1])
                
                try:
                    if self.loss_hist[-1] > self.loss_hist[-2]:
                        for layer in self.layers[1:]:
                            layer.W_optimizer.set_params({"eta":self.optimizer.eta/2})
                            layer.b_optimizer.set_params({"eta":self.optimizer.eta/2})
                        flag = 1
                except:
                    pass

                if flag == 1:
                    break

                # self.layers[-1].cross_grad = self.loss.diff()
                self.layers[-1].a_grad = self.loss.diff(self.t_batch, self.y_batch)
                self.layers[-1].h_grad = self.layers[-1].a_grad * self.layers[-1].activation.diff(self.layers[-1].h[:, batch*self.batch_size:(batch+1)*self.batch_size])

                self.layers[-1].W_grad = self.layers[-1].h_grad @ self.layers[-2].a[:, batch*self.batch_size:(batch+1)*self.batch_size].T
                self.layers[-1].W_update = self.layers[-1].W_optimizer.get_update(self.layers[-1].W_grad)
                
                self.layers[-1].b_grad = -np.sum(self.layers[-1].h_grad, axis=1).reshape(-1,1)
                self.layers[-1].b_update = self.layers[-1].b_optimizer.get_update(self.layers[-1].b_grad)

                # print("Last Layer")
                # print("a_grad shape:", self.layers[-1].a_grad.shape)
                # print("h_grad shape:", self.layers[-1].h_grad.shape)
                # print("W_grad shape:", self.layers[-1].W_grad.shape)
                # print("W_update shape:", self.layers[-1].W_update.shape)
                # print("W_shape:", self.layers[-1].W.shape)
                # print("a_grad:\n", self.layers[-1].a_grad)
                # print("h_grad:\n", self.layers[-1].h_grad)
                # print("W_grad:\n", self.layers[-1].W_grad)

                assert self.layers[-1].W_update.shape == self.layers[-1].W.shape, "Sizes don't match"


                # Backpropogation for the remaining layers
                for i in range(len(self.layers[:-2]), 0, -1):
                    self.layers[i].a_grad = self.layers[i+1].W.T @ self.layers[i+1].h_grad
                    self.layers[i].h_grad = self.layers[i].a_grad * self.layers[i].activation.diff(self.layers[i].h[:, batch*self.batch_size:(batch+1)*self.batch_size])
                    # print("Layer -", i)
                    # print("a_grad shape:", self.layers[i].a_grad.shape)
                    # print("h_grad shape:", self.layers[i].h_grad.shape)

                    # print("Layer -", i)
                    # print("a_grad:", self.layers[i].a_grad)
                    # print("h_grad:", self.layers[i].h_grad)

                    self.layers[i].b_grad = -np.sum(self.layers[i].h_grad, axis=1).reshape(-1,1)
                    self.layers[i].W_grad = self.layers[i].h_grad @ self.layers[i-1].a[:, batch*self.batch_size:(batch+1)*self.batch_size].T
                    
                    # print("W_grad shape:", self.layers[i].W_grad.shape)
                    # print("W_grad:", self.layers[i].W_grad)
                    # print()
                    self.layers[i].W_update = self.layers[i].W_optimizer.get_update(self.layers[i].W_grad)
                    self.layers[i].b_update = self.layers[i].b_optimizer.get_update(self.layers[i].b_grad)
                    # self.layers[i].b_update = self.layers[i].b_optimizer.get_update(self.layers[i].b_grad)

                # Update the weights
                for _, layer in enumerate(self.layers[1:]):
                    layer.W = layer.W - layer.W_update
                    layer.b = layer.b - layer.b_update
                    # print("Layer -", idx)
                    # print("W:\n", layer.W)
                    # print("h:\n", layer.h)

                    # layer.b = layer.b - self.b_update
                # print("Y:\n", self.layers[-1].y)
                self.forward_propogation()

            if flag == 1:
                break

    def describe(self):
        print("Model with the following layers:")
        for i in self.layers:
            print(i)
        print("Loss:", self.loss)
        print("Epochs:", self.epochs)
        print("Batch Size:", self.batch_size)
        print("Optimizer:", self.optimizer)
        print("Initialization:", self.initialization)

    def get_accuracy(self, validation=False, print_vals=False):
        encoder = OneHotEncoder()
        t_train = encoder.inverse_transform(self.t)
        y_train = encoder.inverse_transform(self.layers[-1].y)
        acc_train = np.sum(t_train==y_train)
        if print_vals:
            print("Train Accuracy:", acc_train)

        if validation:
            t_val = encoder.inverse_transform(self.t_val)
            y_val = encoder.inverse_transform(self.layers[-1].y_val)
            acc_val = np.sum(t_val==y_val)
            if print_vals:
                print("Validation Accuracy:", acc_val)
            return acc_train, acc_val
        return acc_train

import numpy as np

################################################
#         Additional Helper Fucntions
################################################
class OneHotEncoder():
    def __init__(self):
        pass
    
    def fit(self, y, num_classes):
        self.y = y
        self.num_classes = num_classes

    def transform(self):
        transformed = np.zeros((self.num_classes, self.y.size))
        for col,row in enumerate(self.y):
            transformed[row, col] = 1
        return transformed

    def fit_transform(self, y, num_classes):
        self.fit(y, num_classes)
        return self.transform()

    def inverse_transform(self, y):
        # Assumes direct correation between the position and class number
        y_class = np.argmax(y, axis=0)
        return y_class


class MinMaxScaler():
    def __init__(self):
        pass

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        transformed = (X - self.min)/(self.max-self.min)
        return transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


import numpy as np

################################################
#         Initializers
################################################
class RandomNormal():
    def __init__(self, mean = 0.0, stddev = 1.0):
        self.mean = mean
        self.stddev = stddev
    
    def weights_biases(self, n_prev, n_curr):
        W = np.random.normal(loc = self.mean, scale = self.stddev, \
                             size = (n_prev, n_curr))
        b = np.random.normal(loc = self.mean, scale = self.stddev, \
                             size = (n_curr,))
        return W, b
    
class XavierUniform():
    def __init__(self):
        pass
    
    def weights_biases(self, n_prev, n_curr):
        upper_bound = np.sqrt(6.0/(n_prev + n_curr))
        lower_bound = -1*upper_bound
        W = np.random.uniform(low = lower_bound, high = upper_bound, \
                              size = (n_prev, n_curr))
        b = np.zeros((n_curr,), dtype = np.float64)
        return W, b



import numpy as np

map_activations = {"Sigmoid":Sigmoid(), "Tanh":Tanh(), "Relu":Relu(), "Softmax":Softmax()}

################################################
#         Layers
################################################
class Input():
    def __init__(self, data):
        self.name = "Input"
        self.input = data
        self.size = self.input.shape[0]
        # self.input = np.append(data, np.ones((1, data.shape[1])), axis=0)
        # Having the input as the activated output 
        # to be given to the next layer
        self.a = self.input
        self.type = "Input layer"

    def __repr__(self):
        representation = self.type + " - of Size:" + str(self.size)
        return representation

class Dense():
    def __init__(self, size, activation, name, last=False):
        self.name = name
        self.size = size
        self.activation = map_activations[activation]
        self.activation_name = activation
        self.type = "Dense layer"

    def __repr__(self):
        representation = self.type + " - of Size:" + str(self.size) + "; Activation:" + self.activation_name
        return representation


import numpy as np

################################################
#         Loss
################################################
class CrossEntropy():
	def __init__(self):
		pass

	def calc_loss(self, t, y):
		self.t = t
		self.y = y
		loss = -np.sum(np.sum(self.t*np.log(self.y)))
		return loss

	def diff(self):
		grad = -self.t/(self.y)
		return grad

class SquaredError():
	def __init__(self):
		pass

	def calc_loss(self, t, y):
		self.t = t
		self.y = y
		loss = np.sum((t-y)**2)
		return loss

	def diff(self, t_batch, y_batch):
		grad = -(t_batch - y_batch)
		return grad




import numpy as np

################################################
#         Optimizers
################################################
class Normal():
    def __init__(self, eta=0.01):
        self.update = 0
        self.eta = eta

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])

    def get_update(self, grad):
        self.update = self.eta*grad
        return self.update

class Momentum():
    def __init__(self, eta=1e-3, gamma=0.9):
        self.update = 0
        self.eta = eta
        self.gamma = gamma

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])

    def get_update(self, grad):
        self.update = self.gamma*self.update + self.eta*grad
        return self.update

class Nesterov():
    def __init__(self, eta=1e-3, gamma=0.9):
        self.update = 0
        self.eta = eta
        self.gamma = gamma

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        
    def get_update(self, W, grad=None):
        # Have to still work on this
        W_lookahead = W - self.gamma*self.update
        self.update = self.gamma*self.update + self.eta*gradient(W_lookahead) # Need to call gradient function
        W = W - self.update
        return W
        

class AdaGrad():
    def __init__(self, eta=1e-2, eps=1e-7):
        self.v = 0
        self.eta = eta
        self.eps = eps

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
    
    def get_update(self, grad):
        self.v = self.v + grad**2
        return (self.eta/(self.v+self.eps)**0.5)*grad

class RMSProp():
    def __init__(self, beta=0.9, eta = 1e-3, eps = 1e-7):
        self.v = 0
        self.beta = beta
        self.eta = eta
        self.eps = eps

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])

    def get_update(self, grad):
        self.v = self.beta*self.v + (1-self.beta)*(grad**2)
        return (self.eta/(self.v+self.eps)**0.5)*grad

class Adam():
    def __init__(self, beta1=0.9, beta2=0.999, eta=1e-2, eps=1e-8):
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.eps = eps
        self.iter = 1

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])

    def get_update(self, grad):
        # try:
        #     print("Size of M:", self.m.shape)
        # except:
        #     pass
        # print("Size of term 2:", grad.shape)
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        m_cap = self.m/(1-self.beta1**self.iter)
        v_cap = self.v/(1-self.beta2**self.iter)        
        self.iter += 1
        return (self.eta/(v_cap+self.eps)**0.5)*m_cap

class Nadam():
    # Reference: https://ruder.io/optimizing-gradient-descent/index.html#nadam
    def __init__(self, beta1=0.9, beta2=0.999, eta=1e-3, eps=1e-7):
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.eps = eps
        self.iter = 1

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
    
    def get_update(self, grad):
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        m_cap = self.m/(1-self.beta1**self.iter)
        v_cap = self.v/(1-self.beta2**self.iter) 
        update = self.beta1*m_cap + ((1-self.beta1)/(1-self.beta1**self.iter))*grad
        self.iter += 1
        return (self.eta/(v_cap+self.eps)**0.5)*update





