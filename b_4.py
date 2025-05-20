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

graus = range(1, 11) # Plotando todos os erros de grau 1 a 10
erros = []

for grau in graus:
    X = np.zeros((len(t), grau + 1))
    for i in range(grau + 1):
        X[:, i] = t ** i
    coef = np.linalg.solve(X.T @ X, X.T @ y)
    y_fit = X @ coef
    erro_total = np.sum((y - y_fit) ** 2)
    erros.append(erro_total)

plt.figure(figsize=(8, 5))
plt.plot(graus, erros, marker='o')
plt.xlabel('Grau do polinômio')
plt.ylabel('Erro total (soma dos quadrados)')
plt.title('Erro do ajuste em função do grau do polinômio interpolador')
plt.grid(True)
plt.show()