# Paula Sophia Santoyo Arteaga
# A01745312
# 11-Sept-2023
# Uso de framework de aprendizaje máquina para la implementación de una solución
# ------------------------------------------------------------------------------


# Importar librerias
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar dataset de wine de scikit-learn
def load_data():
    # Almacena el dataset de vinos
    wine = load_wine()
    # Almacena los datos de los atributos
    x = wine.data
    print(f'Datos de los atributos\n{x}')
    # Almacena los datos de la clasificación del vino
    y = wine.target
    print(f'Datos de la clasificación del vino\n{y}')
    # Almacena los nombres de la clasificacion de vinos (clase 0, 1 o 2)
    target_names = wine.target_names
    return x, y, target_names


def train_model(x, y, max_depth=None):
    # Dividir los datos en 2: entrenamiento y testing (temp)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, 
                                            test_size=0.2, random_state=42)
    # Dividir los datos para testing y validación
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, 
                                            test_size=0.5, random_state=42)
    # Crear modelo de árbol de decisión con profundidad y 
    # semilla aleatoria personalizada
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    # Entrenar el modelo con los datos de entrenamiento (x_train y y_train)
    clf.fit(x_train, y_train)
    # Regresa los datos de entrenamiento, validación, testing 
    # y el modelo entrenado
    return x_train, x_val, x_test, y_train, y_val, y_test, clf


def evaluate_model(max_depth, clf, x, y, data_type):
    # Realiza predicciones de los datos de la variable x (train o test)
    y_pred = clf.predict(x)
    # Calcula la precisión del modelo
    accuracy = accuracy_score(y, y_pred)
    # Crea el informe de clasificación de los datos proporcionados
    classification = classification_report(y, y_pred, 
                        zero_division=0, target_names=target_names)
    # Crea la matriz de confusión
    confusion = confusion_matrix(y, y_pred)

    # Muestra el valor de precisión y el informe de clasificación del modelo
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{classification}')
    # Manda llamar la función que crea la gráfica de la matriz de confusión
    c_matrix(max_depth, data_type, confusion)


def c_matrix(max_depth, data_type, matrix):
    # Genera la gráfica de la matriz de confusión
    plt.figure(figsize=(5, 3))
    sns.set(font_scale=1)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="RdPu", 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix {data_type} Run {max_depth}')
    plt.show()


def prediction(clf):
    # Datos para hacer una predicción 
    new_wine = [[13.24, 2.59, 2.87, 21.0, 118.0, 2.8, 2.69, 
                 0.39, 1.82, 4.32, 1.04, 2.93, 735.0]]
    # Usar el modelo entrenado para hacer la predicción con los datos de arriba
    predictions = clf.predict(new_wine)
    # Muestra la predicción del vino ingresado
    print('\n~ PREDICCIÓN ~')
    print(f'Atributos del vino: {new_wine}')
    print(f'Predicción para el vino: {target_names[predictions[0]]}')


if __name__ == "__main__":
    # Llama la función para obtener los datos
    x, y, target_names = load_data()
    
    # Ejecuta el entrenamiento, validación y test de los datos 3 veces
    for max_depth in range(1, 4):
        # Obtiene los datos para entrenamiento, validación, test y el modelo entrenado
        x_train, x_val, x_test, y_train, y_val, y_test, clf = train_model(x, 
                                                                    y, max_depth)
        # Muestra la ejecución que está corriendo como la profundidad máxima con la que 
        # se esta entrenando
        print(f'\n*****   RUN {max_depth}   *****\nMax Depth: {max_depth}\n')
        print('                      ~ TRAINING DATA ~\n')
        # Se evalua el modelo con los datos de entrenamiento
        evaluate_model(max_depth, clf, x_train, y_train, "Training")
        print('\n                      ~ TEST DATA ~')
        # Se evaulua el modelo con los datos de testing
        evaluate_model(max_depth, clf, x_test, y_test, "Test")
        print('-' * 40)
    
    # Llama la función que hace la predicción con los datos de un vino nuevo
    prediction(clf)