
import matplotlib.pyplot as plt

# Simulated accuracy values across rounds
rounds = [1, 2, 3, 4, 5]
accuracy = [0.85, 0.85, 0.85, 0.85, 0.85]

plt.plot(rounds, accuracy, marker='o')
plt.title("Accuracy vs Training Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.grid()

plt.show()

import matplotlib.pyplot as plt

rounds = [1, 2, 3, 4, 5]
loss = [0.35, 0.34, 0.34, 0.33, 0.33]

plt.plot(rounds, loss, marker='o')
plt.title("Loss vs Training Rounds")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.grid()

plt.show()

import matplotlib.pyplot as plt

rounds = [1, 2, 3, 4, 5]
values = [0.80, 0.83, 0.84, 0.85, 0.85]

plt.plot(rounds, values, marker='o')
plt.title("Federated Learning Convergence")
plt.xlabel("Rounds")
plt.ylabel("Model Performance")
plt.grid()

plt.show()

import matplotlib.pyplot as plt

models = ["Centralized", "Federated"]
accuracy = [0.90, 0.85]

plt.bar(models, accuracy)
plt.title("Centralized vs Federated Learning Accuracy")
plt.xlabel("Model Type")
plt.ylabel("Accuracy")

plt.show()

