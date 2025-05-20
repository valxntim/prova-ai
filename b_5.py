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

# Ponto de teste
new_data = [0.16, -0.452298074]
t_test = new_data[0]
y_test = new_data[1]

graus = range(1, 11)
erros_teste = []

for grau in graus:
    # Ajusta o polinômio aos dados originais
    X = np.zeros((len(data), grau + 1))
    for i in range(grau + 1):
        X[:, i] = data[:, 0] ** i
    coef = np.linalg.solve(X.T @ X, X.T @ data[:, 1])
    # Avalia o polinômio no ponto de teste
    y_pred = sum(coef[i] * t_test**i for i in range(grau + 1))
    erro = abs(y_pred - y_test)
    print(erro)
    erros_teste.append(erro)

plt.figure(figsize=(8, 5))
plt.plot(graus, erros_teste, marker='o', color='orange')
plt.xlabel('Grau do polinômio')
plt.ylabel('Erro absoluto no ponto de teste')
plt.title('Erro no ponto de teste em função do grau do polinômio')
plt.grid(True)
plt.show()