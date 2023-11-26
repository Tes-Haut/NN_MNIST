import numpy as np
import requests
from io import BytesIO
from gzip import GzipFile
import matplotlib.pyplot as plt
import pickle


def load_mnist():
    url_image_train = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_label_train = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'

    url_image_test = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_label_test = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    urls = [url_image_train, url_label_train, url_image_test, url_label_test]

    files = []
    for url in urls:
        response = requests.get(url)
        with GzipFile(fileobj=BytesIO(response.content), mode='rb') as f:
            content = f.read()
            files.append(content)

    # Training Img
    trainImg = np.frombuffer(files[0], dtype=np.uint8, offset=16).reshape(-1, 28, 28) / 255
    trainImg = np.reshape(trainImg, (trainImg.shape[0], trainImg.shape[1] * trainImg.shape[2]))
    # Training Label
    trainLabel = np.eye(10)[np.frombuffer(files[1], dtype=np.uint8, offset=8)]
    # Test Img
    testImg = np.frombuffer(files[2], dtype=np.uint8, offset=16).reshape(-1, 28, 28) / 255
    testImg = np.reshape(testImg, (testImg.shape[0], testImg.shape[1] * testImg.shape[2]))
    # Test Label
    testLabel = np.eye(10)[np.frombuffer(files[3], dtype=np.uint8, offset=8)]

    return trainImg, trainLabel, testImg, testLabel


def softmax(x):
    A = np.exp(x) / sum(np.exp(x))
    return A


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):  # O(h_pre) * (1 - O(h_pre))
    return np.exp(-x) / (1 + np.exp(-x)) ** 2

def derivative_hidden(hidden): # h * (1 - h)
    return hidden * (1 - hidden)


def mse_derivative(o, l):
    return (2 / len(o)) * (o - l)


def forwardPropagation(w1, b1, w2, b2, i):
    h = sigmoid(b1 + w1 @ i)
    o = sigmoid(b2 + w2 @ h)

    return h, o


def cost(output, label):  # Mean Squared Error (MSE)
    e = 1 / len(output) * np.sum((output - label) ** 2, axis=0)
    return e


def backPropagation(hidden, output, label, w_h_o, b_h_o, w_i_h, b_i_h, alpha, img):
    o_delta = output - label  # Cost function derivative (MSE) -> pas de 1/n car flm -> + lisible, + maj poids
    w_h_o += -alpha * o_delta @ hidden.T
    b_h_o += -alpha * o_delta

    h_delta = w_h_o.T @ o_delta * derivative_hidden(hidden)  # Active function derivative (Sigmoid)
    w_i_h += -alpha * h_delta @ img.T
    b_i_h += -alpha * h_delta


def save_parameters(w_i_h, b_i_h, w_h_o, b_h_o, filename="parameters.pkl"):
    parameters = {
        "w_i_h": w_i_h,
        "b_i_h": b_i_h,
        "w_h_o": w_h_o,
        "b_h_o": b_h_o
    }
    with open(filename, 'wb') as file:
        pickle.dump(parameters, file)


def load_parameters(filename="parameters.pkl"):
    with open(filename, 'rb') as file:
        parameters = pickle.load(file)
    return parameters["w_i_h"], parameters["b_i_h"], parameters["w_h_o"], parameters["b_h_o"]

# Load MNIST
trainImg, trainLabel, testImg, testLabel = load_mnist()

w_i_h = np.random.uniform(-0.5, 0.5, (28, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 28))
b_i_h = np.zeros((28, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.025
epochs = 0

Load = input("Train or Load ? T/L ")
if Load == "L":
    w_i_h, b_i_h, w_h_o, b_h_o = load_parameters()
else:
    epochs = int(input("How many epochs ? "))
    print(f"Start training for {epochs} epochs")

for epoch in range(epochs):
    correctOutput = 0
    totalOutput = 0
    for img, label in zip(trainImg, trainLabel):
        img.shape += (1,)  # Vector to matrix for calcul
        label.shape += (1,)

        hidden, output = forwardPropagation(w_i_h, b_i_h, w_h_o, b_h_o, img)

        error = cost(output, label)

        correctOutput += int(np.argmax(output) == np.argmax(label))
        totalOutput += 1

        backPropagation(hidden, output, label, w_h_o, b_h_o, w_i_h, b_i_h, learn_rate, img)
    print(f"CorrectOutput : {correctOutput} / {totalOutput} | Accuracy : {(correctOutput/totalOutput)*100}% | Epoch : {epoch + 1}")

if Load == "T":
    Save = input("End Training. Do you want to save parameters ? Y/N ")
    if Save == "Y":
        save_parameters(w_i_h, b_i_h, w_h_o, b_h_o)

while True:
    index = int(input("Enter a number (0 - 999): "))
    img = testImg[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    _, output = forwardPropagation(w_i_h, b_i_h, w_h_o, b_h_o, img)

    plt.title(f"Prediction : {output.argmax()}")
    plt.show()
