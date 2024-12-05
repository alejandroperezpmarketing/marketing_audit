import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Crear datos de ejemplo
X, y = make_classification(
    n_samples=500, 
    n_features=5, 
    n_informative=3, 
    n_redundant=0, 
    n_classes=3, 
    n_clusters_per_class=1, 
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluar diferentes valores de k
k_values = range(1, 21)
errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)  # Tasa de error
    errors.append(error)

# Crear el gráfico del codo
plt.figure(figsize=(10, 6))
plt.plot(k_values, errors, marker='o', linestyle='-', color='b', label="Tasa de error")
plt.title("Método del Codo para determinar el valor óptimo de k")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("Tasa de error")
plt.xticks(k_values)
plt.axvline(x=errors.index(min(errors)) + 1, color='r', linestyle='--', label="Valor óptimo de k")
plt.legend()
plt.grid()
plt.show()
