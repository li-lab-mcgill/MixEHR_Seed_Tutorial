import matplotlib.pyplot as plt

# Given values
values = [-18928.88123435822, -10100.90205618582, -4265.862627919032, -3317.6474367274723, -3119.9120292870484, -3066.0972879218384, -3055.641014568567, -3053.965448990082, -3053.8296066828652, -3053.757824674462]
epochs = list(range(1, len(values) + 1))
plt.figure(figsize=(8, 5))
plt.plot(epochs, values, marker='o')
plt.title('SVI ELBO')
# plt.xlabel('Epoch')
plt.ylabel('value')
plt.grid(True)
plt.show()
