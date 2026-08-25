import ten
import matplotlib.pyplot as plt

x = ten.random.rand_norm([1000])

print("Plot histogram")
ten.hist(x)
plt.show()

print("Low level histogram")
h = ten.histogram()
h.fit(x)
h.plot(color = "red", fill = "red")
plt.show()

