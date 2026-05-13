from nn import MLP
n = MLP(2, [8, 8, 1])

import random

random.seed(42)

xs = []
ys = []

for _ in range(100):
    x = random.uniform(-1.5, 1.5)
    y = random.uniform(-1.5, 1.5)

    xs.append([x, y])

    if x*x + y*y < 1.0:
        ys.append(1.0)
    else:
        ys.append(-1.0)


for k in range(500):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)) / len(xs)

    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    lr = 0.03

    for p in n.parameters():
        p.data += -lr * p.grad

    if k % 50 == 0:
        print(k, loss.data)


correct = 0

for x, ygt in zip(xs, ys):
    yout = n(x).data
    pred = 1.0 if yout > 0 else -1.0

    if pred == ygt:
        correct += 1

print("accuracy:", correct / len(xs))