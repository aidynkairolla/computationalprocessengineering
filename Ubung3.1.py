import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

y_sin = np.sin(x)
y_cos = np.cos(x*x)

plt.plot(x, y_sin, label='sin(x)')
plt.plot(x, y_cos, label='cos(x^2)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(x) and cos(x^2)')
plt.show()
