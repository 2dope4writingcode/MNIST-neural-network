from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

images, labels = get_mnist()
wih = np.random.uniform(-0.5, 0.5, (20, 784))
who = np.random.uniform(-0.5, 0.5, (10, 20))
bih = np.zeros((20, 1))
bho = np.zeros((10,1))

learn_rate = 0.01
nr_correct = 0
epochs = 50

for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)

        h_pre = bih + wih @ img
        h = 1 / (1 + np.exp(-h_pre))

        o_pre = bho + who @ h
        o = 1 / (1 + np.exp(-o_pre))

        e = 1 / len(o) * np.sum((o - l) ** 2, axis = 0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        delta_o = o - l
        who += -learn_rate * delta_o @ np.transpose(h)
        bho += -learn_rate * delta_o

        delta_h = np.transpose(who) @ delta_o * (h * (1 - h))
        wih += -learn_rate * delta_h @ np.transpose(img)
        bih += -learn_rate * delta_h

    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

while True:
    index = int(input("Enter a number (0 - 59999):"))
    img = images[index]
    plt.imshow(img.reshape(28,28), cmap="Greys")

    img.shape += (1,)

    h_pre = bih + wih @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))

    o_pre = bho + who @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Activated neuron is {o.argmax()}")
    plt.show()



