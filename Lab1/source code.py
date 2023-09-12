import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

np.random.seed(0)

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def sigmoid(a, derivative=False):
    if not derivative:
        return 1.0 / (1.0 + np.exp(-a))
    else:
        return np.multiply(a, 1.0 - a)

def relu(a, derivative=False):
    if not derivative:
        return np.maximum(0.0, a)
    else:
        return np.heaviside(a, 0.001)

def tanh(a, derivative=False):
    if not derivative:
        return np.tanh(a)
    else:
        return 1.0 - a**2
    

def no_activation(a, derivative=False):
    if not derivative:
        return a
    else:
        return 1

def mse(a, b, derivative=False):
    if not derivative:
        return (a - b) **2.0
    else:
        return 2.0 * (a - b)


def show_result(x, y, pred_y):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
'''def learning_curve(loss, epoch, data_file):
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = []
    epochs = np.arange(1 ,epoch + 1)
    data.append((epochs, loss))
    learning = [0.01, 0.1, 0.9, 0.09, 0.15, 0.2, 0.17, 0.159, 0.14]
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    for i, (x, y) in enumerate(data):
        plt.plot(x, y, label=f'learning rate={learning[i]})
    epochs = np.arange(1 ,epoch + 1)
    plt.plot(epochs, loss)
    plt.title('Learning_curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    '''

def learning_curve(loss, epoch):
    epochs = np.arange(1 ,epoch + 1)
    plt.plot(epochs, loss)
    plt.title('Learning_curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#定義全連接層
class FCLayer:
    def __init__(self, input_nodes, output_nodes, activation = "sigmoid"):
        self.weight = np.random.normal(0, 1, (input_nodes, output_nodes))
        self.activation = activation
        self.F = np.zeros((output_nodes, 1)) #input和weight相乘後的矩陣
        self.Z = np.zeros((output_nodes, 1)) #F經過activation後的矩陣
        self.dC_dF = np.zeros((output_nodes, 1)) #C是loss function
        self.gradient = np.zeros((output_nodes, input_nodes))
        self.movement = np.zeros((input_nodes, output_nodes))
        self.movement_hat = np.zeros((input_nodes, output_nodes))
        self.v = np.zeros((input_nodes, output_nodes))
        self.v_hat = np.zeros((input_nodes, output_nodes))
        self.t = 1
        self.sum_square_gradient = np.zeros((output_nodes, input_nodes))
        if activation == "sigmoid":
            self.activation = sigmoid
        elif activation == "relu":
            self.activation = relu
        elif activation == "tanh":
            self.activation = tanh
        else:
            self.activation = no_activation
    


class NeuralNet:
    def __init__(self, input_dim = 2, hidden1_dim = 3, hidden2_dim = 3, output_dim = 1
                 , num_weightlayers = 3, activation = 'sigmoid', learning_rate = 0.01, optimizer = 'gd'):
        self.num_weightlayers = num_weightlayers
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.activation = activation

        #設定input層
        self.layers = [FCLayer(input_dim, hidden1_dim, activation)]

        #設定中間層
        self.layers.append(FCLayer(hidden1_dim, hidden2_dim, activation))

        #設定output層
        self.layers.append(FCLayer(hidden2_dim, output_dim, activation))
    
    def forward_pass(self, x):
        z = x
        for i in range(self.num_weightlayers):
            f = np.matmul(z, self.layers[i].weight)
            self.layers[i].F = f
            z = self.layers[i].activation(f)
            self.layers[i].Z = z
        return z
    
    def backward_pass(self, y, y_hat):
        self.layers[2].dC_dF = mse(y, y_hat, derivative=True) * self.layers[2].activation(self.layers[2].Z, derivative=True)
        for i in range(1, -1, -1):
            self.layers[i].dC_dF = self.layers[i].activation(self.layers[i].Z, derivative = True) * np.matmul(self.layers[i+1].weight, self.layers[i+1].dC_dF )
    
    def compute_grad(self, x):
        self.layers[0].gradient = x * self.layers[0].dC_dF.reshape(-1, 1)
        for i in range(1, 3, 1):
            self.layers[i].gradient = self.layers[i-1].Z * self.layers[i].dC_dF.reshape(-1, 1)
    def optimize(self):
        if self.optimizer == 'gd':
            for i in range(3):
                self.layers[i].weight = self.layers[i].weight + (-self.learning_rate * self.layers[i].gradient.T)
        elif self.optimizer == 'momentum':
            for i in range(3):
                self.layers[i].movement = (0.9 * self.layers[i].movement) +  (-self.learning_rate * self.layers[i].gradient.T)
                self.layers[i].weight = self.layers[i].weight + self.layers[i].movement
        elif self.optimizer == 'adagrad':
            for i in range(3):
                self.layers[i].sum_square_gradient += np.square(self.layers[i].gradient)
                self.layers[i].weight = self.layers[i].weight + ((-self.learning_rate * self.layers[i].gradient.T) / (np.sqrt(self.layers[i].sum_square_gradient).T +1e-8))
        elif self.optimizer == 'adam':
            for i in range(3):
                self.layers[i].movement = (0.9 * self.layers[i].movement) +  (0.1 * self.layers[i].gradient.T)
                self.layers[i].movement_hat = self.layers[i].movement / (1 - (0.9 ** self.layers[i].t))
                if self.layers[i].t > 1 :
                    self.layers[i].v = 0.999 * self.layers[i].v + 0.001 * np.square(self.layers[i].gradient).T
                else :
                    self.layers[i].v = np.square(self.layers[i].gradient).T 
                self.layers[i].v_hat = self.layers[i].v / (1 - (0.999 ** self.layers[i].t))
                self.layers[i].weight = self.layers[i].weight - (self.learning_rate * self.layers[i].movement_hat / (np.sqrt(self.layers[i].v_hat) + 1e-8))
                self.layers[i].t += 1
    
    
def main():
    inputs, labels  = generate_linear(100)
    #inputs, labels  = generate_XOR_easy()
    epoch = 10000
    model = NeuralNet(2, 3, 3, 1, 3, 'sigmoid', 0.01, 'gd')
    total_loss = []
    # training
    for i in range(1, epoch+1):
        loss = 0
        for x, y in zip(inputs, labels): 
            output = model.forward_pass(x)
            loss = loss + mse(output, y).item()
            model.backward_pass(output, y)
            model.compute_grad(x)
            model.optimize()
        total_loss.append(loss / len(inputs))    
        if i % 1000 == 0:
            print('epoch', i , 'loss :' , loss / len(inputs))
    loss = 0
    iters = 1
    correct = 0
    pred_y = []
    for x, y in zip(inputs, labels): 
        output = model.forward_pass(x)
        loss = loss + mse(output, y).item()
        pred_y.append(np.round(output).item())
        print(f'Iter{iters} |  Ground truth:{y.item()} |  prediction:{output.item()}')
        if np.round(output) == y:
            correct = correct + 1
        iters = iters + 1
    accuracy = (correct / len(labels)) * 100
    print(f'loss={loss / len(inputs)} accuracy={accuracy}%')
    show_result(inputs, labels, pred_y)
    learning_curve(total_loss, epoch)
    #data_file = 'data.pkl_xor_opt(7)'
    #learning_curve(total_loss, epoch, data_file)


if __name__ == '__main__':
    main()





    





                
        



        



        


        



        