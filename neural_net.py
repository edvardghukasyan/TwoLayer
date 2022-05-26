import numpy as np

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        param input_size: integer, number of features of the input
        param hidden_size: integer, arbitrary number of parameters
        param output_size: integer, number of classes

        Define simple two layer neural network with relu activation function.
        '''
        self.params = {}

        std = 1e-4
        self.params['W1'] = std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def loss(self, X, y, reg=0.0):
        '''
        param X: numpy.array, input features
        param y: numpy.array, input labels
        param reg: float, regularization value


        Return:
        param loss: Define loss with data loss and regularization loss
        param grads: Gradients for weights and biases
        '''

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        loss = None
        grads = {}

        hidden_layer = np.maximum(0,X.dot(W1) + b1)
        scores  = hidden_layer.dot(W2) + b2        

        scores_shifted_exp = np.exp(scores - np.max(scores, axis=1).reshape(scores.shape[0],1))
        sum_of_exps = np.sum(scores_shifted_exp, axis=1).reshape(scores_shifted_exp.shape[0],1)
        probs = scores_shifted_exp/sum_of_exps
        
        y_pred = np.argmax(scores, axis=1)

        loss = -np.mean(np.log(probs[np.arange(probs.shape[0]), y])) + reg * (np.sum(W2*W2) + np.sum(W1*W1))
        softmax_grad = probs
        softmax_grad[np.arange(len(y)),y] -= 1
        softmax_grad /= len(y)

        grads['b2'] = np.sum(softmax_grad, axis=0)
        grads['W2'] = hidden_layer.T.dot(softmax_grad) + 2*reg*W2
        hidden_grad = softmax_grad.dot(W2.T)
        hidden_grad[hidden_layer == 0] = 0
        grads['b1'] = np.sum(hidden_grad, axis=0)
        grads['W1'] = X.T.dot(hidden_grad) + 2*reg*W1

        return loss, grads


    def train(self, X_train, y_train, X_val, y_val, learning_rate=1e-3, batch_size=4, num_iters=100):
        '''
        param X_train: numpy.array, trainset features 
        param y_train: numpy.array, trainset labels
        param X_val: numpy.array, valset features
        param y_val: numpy.array, valset labels
        param learning_rate: float, learning rate should be used to updated grads
        param batch_size: float, batch size is the number of images should be used in single iteration
        param num_iters: int, number of iterations you want to train your model
        '''

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        indexes = np.arange(X_train.shape[0])
        for it in range(num_iters):
            X_batch, y_batch = None, None
            batch_idx = np.random.choice(indexes, batch_size)
            X_batch = X_train[batch_idx, :]
            y_batch = y_train[batch_idx]
            reg = 0.2
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            if num_iters == 55000:
                learning_rate = 0.01*learning_rate
           
            self.params['W1'] -= learning_rate*grad['W1']
            self.params['W2'] -= learning_rate*grad['W2']
            self.params['b1'] -= learning_rate*grad['b1']
            self.params['b2'] -= learning_rate*grad['b2']

            if (it+1) % 100 == 0:
                print(f'Iteration {it+1} / {num_iters} : {loss}')
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
            

        return {'loss_history': loss_history, 'train_acc_history': train_acc_history, 'val_acc_history': val_acc_history}

    def predict(self, X):
        '''
        param X: numpy.array, input features matrix
        return y_pred: Predicted values
        '''
        y_pred = None

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        hidden_layer = np.maximum(0,X.dot(W1) + b1)
        scores  = hidden_layer.dot(W2) + b2
        y_pred = np.argmax(scores, axis=1)

        return y_pred
