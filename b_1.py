import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
data = np.array([
  [0.1, -0.57169266822719167],
  [0.3, -0.32694832826410258],
  [0.5,  0.15476157111017930],
  [0.7,  0.84623831164925778],
  [0.9,  1.1367486533816560],
  [1.1,  0.51868551715289568],
  [1.3,  0.86873878660264336],
  [1.5,  1.1652758836729025],
  [1.7,  0.93710248962036347],
  [1.9,  0.92183741670636998]
])

t = data[:, 0]
y = data[:, 1]

# Cálculo dos coeficientes do ajuste linear (y = a*x + b)
n = len(t)    # Tamanho da entrada
t_mean = np.mean(t)
y_mean = np.mean(y)

# Calcula a inclinação (a)
num = np.sum((t - t_mean) * (y - y_mean))
den = np.sum((t - t_mean) ** 2)
a = num / den

# Calcula (b)
b = y_mean - a * t_mean

# Predição dos valores ajustados
y_fit = a * t + b

# Erro total (soma dos quadrados dos resíduos)
erro_total = np.sum((y - y_fit) ** 2)

print(f"Ajuste linear: y = {a:.6f} x + {b:.6f}")
print(f"Erro total do ajuste: {erro_total:.6f}")

plt.scatter(t, y, color='blue', label='Pontos experimentais')
plt.plot(t, y_fit, color='red', label='Ajuste linear')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Ajuste linear dos dados')
plt.legend()
plt.grid(True)
plt.show()