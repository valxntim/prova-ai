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

graus = [2, 3, 4, 5]
cores = ['red', 'green', 'orange', 'purple']

plt.figure(figsize=(10, 6))
plt.scatter(t, y, color='blue', label='Pontos experimentais')

t_plot = np.linspace(min(t), max(t), 200)

for grau, cor in zip(graus, cores):
    X = np.zeros((len(t), grau + 1))
    # Monta a matriz de Vandermonde
    # Cada coluna i contém t^i
    for i in range(grau + 1):
        X[:, i] = t ** i

    # Resolve o sistema normal dos mínimos quadrados
    coef = np.linalg.solve(X.T @ X, X.T @ y)
    # Calcula os valores ajustados para os pontos originais
    y_fit = X @ coef

    # Monta a matriz para o gráfico suave
    X_plot = np.zeros((len(t_plot), grau + 1))
    for i in range(grau + 1):
        X_plot[:, i] = t_plot ** i
    y_plot = X_plot @ coef

    # Erro total
    erro_total = np.sum((y - y_fit) ** 2)
    # Mostra os coeficientes e erro
    print(f"Polinômio grau {grau}: coef = {coef}, erro total = {erro_total:.6f}")
    # Plota o ajuste
    plt.plot(t_plot, y_plot, color=cor, label=f'Grau {grau}')

plt.xlabel('t')
plt.ylabel('y')
plt.title('Ajuste polinomial dos dados')
plt.legend()
plt.grid(True)
plt.show()