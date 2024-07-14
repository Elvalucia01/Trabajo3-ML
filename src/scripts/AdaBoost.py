import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings

# Ignorar advertencias
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load():
    colNames = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']
    data = pd.read_csv('yeast_no_outliers.csv')

    # Eliminar la columna sequence_name
    data = data.drop(columns=['sequence_name'])

    # Normalización de datos
    scaler = MinMaxScaler()
    data[colNames[:-1]] = scaler.fit_transform(data[colNames[:-1]])

    # Convertir etiquetas de clase a números
    le = LabelEncoder()
    data['class'] = le.fit_transform(data['class'])

    X = data.drop('class', axis=1)
    y = data['class']

    # Balancear el dataset usando SMOTE
    smote = SMOTE(k_neighbors=min(y.value_counts().min() - 1, 5), random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Graficar la distribución de las clases después de SMOTE
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_res)
    plt.title('Distribución de Clases Después de SMOTE')
    plt.xlabel('Clase')
    plt.ylabel('Frecuencia')
    plt.show()

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Definir el modelo y el espacio de búsqueda para los hiperparámetros
    model = AdaBoostClassifier()
    param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluar el modelo usando validación cruzada
    scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"AdaBoost Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

    # Evaluar el desempeño del modelo
    y_pred = best_model.predict(X_test)
    print(f"AdaBoost Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Graficar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusión para AdaBoost')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plotFeatureImportances(model, colN):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Importancia de las características")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(colN)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()

def main():
    load()

if __name__ == "__main__":
    main()
