# Solución de EPDs con métodos de alta fidelidad asistidos por redes neuronales

Repositorio de experimentos numéricos para la solución de ecuaciones en derivadas parciales (EPDs) mediante métodos iterativos, métodos multigrid y técnicas de precondicionamiento. El objetivo es estudiar cómo la estructura algebraica del problema y los precondicionadores influyen en la convergencia, el tiempo de ejecución y la precisión de la solución.

## Contenido del proyecto

| Archivo | Descripción |
|---|---|
| [`NotebookProyInvestigacion.ipynb`](NotebookProyInvestigacion.ipynb) | Comparación de Steepest Descent y Conjugate Gradients para sistemas simétricos definidos positivos. |
| [`GNP,_GMRES,_AMG.ipynb`](GNP,_GMRES,_AMG.ipynb) | Comparación de GMRES sin precondicionamiento, GMRES+AMG y GMRES+GNP en el Laplaciano 1D. |
| [`tiempoGNP.ipynb`](tiempoGNP.ipynb) | Evaluación del tiempo de ejecución de GMRES y GMRES+AMG para diferentes tamaños del sistema, junto con pruebas de GNP. |
| [`Smoothers.ipynb`](Smoothers.ipynb) | Estudio de suavizadores y de su efecto en la convergencia del método multigrid Ruge–Stüben. |
| [`experimentos_ruge_stuben.py`](experimentos_ruge_stuben.py) | Experimentos reproducibles sobre la formación y transferencia entre niveles de una jerarquía Ruge–Stüben. |

## 1. Comparación de Steepest Descent y Conjugate Gradients

**Archivo:** [`NotebookProyInvestigacion.ipynb`](NotebookProyInvestigacion.ipynb)

Este notebook implementa desde cero dos métodos clásicos para resolver sistemas lineales

```text
A x = b
```

cuando `A` es una matriz simétrica definida positiva (SPD):

- **Steepest Descent (B1):** actualiza la solución en la dirección del residual. Es estable, pero puede requerir muchas iteraciones cuando la matriz está mal condicionada.
- **Conjugate Gradients (B2):** construye direcciones conjugadas respecto a `A` y normalmente alcanza la solución con muchas menos iteraciones.

### Experimentos incluidos

1. Generación de matrices SPD aleatorias, tridiagonales de diferencias finitas, diagonales, de Hilbert y con espectro controlado.
2. Cálculo de autovalores, autovalor mínimo, autovalor máximo y número de condición espectral:
   `kappa(A) = lambda_max / lambda_min`.
3. Comparación de la norma del residual, el error en norma de energía, el número de iteraciones y el error final.
4. Comparación en una matriz moderadamente condicionada (`kappa` aproximado de 50).
5. Comparación en una matriz de Hilbert, que representa un caso numéricamente mal condicionado.
6. Estudio de la relación entre el número de condición y la cantidad de iteraciones.

**Conclusión esperada:** al aumentar el número de condición, Steepest Descent se vuelve más lento; Conjugate Gradients aprovecha mejor la distribución espectral y converge en menos iteraciones.

## 2. GMRES con AMG y GNP

**Archivo:** [`GNP,_GMRES,_AMG.ipynb`](GNP,_GMRES,_AMG.ipynb)

Este experimento resuelve el Laplaciano unidimensional mediante matrices tridiagonales dispersas de tamaños `n = 1024`, `4096` y `16384`. Se comparan:

- **GMRES:** método iterativo general para sistemas lineales, utilizado sin precondicionamiento.
- **GMRES + AMG:** GMRES utilizando un precondicionador construido con Algebraic Multigrid (AMG) de PyAMG.
- **GMRES + GNP:** GMRES utilizando el precondicionador basado en redes neuronales GNP del proyecto externo integrado en el notebook.

Para cada tamaño se registran el tiempo de ejecución, el número de iteraciones y el residual relativo:

```text
||b - A x|| / ||b||
```

El notebook también incluye comandos para instalar y ejecutar el código de GNP, además de una tabla comparativa de tiempos y residuales.

**Resultado documentado en el notebook:** AMG reduce de forma importante el tiempo y el residual frente a GMRES directo. En los casos registrados, GMRES+GNP presenta un costo mayor y residuales superiores, especialmente para `n = 16384`.

> Nota: las celdas de entrenamiento de GNP contienen actualmente un marcador (`código que entrena GNP`); por ello, el entrenamiento debe configurarse antes de considerarlo un flujo completamente automatizado.

## 3. Medición de tiempos para GMRES y AMG

**Archivo:** [`tiempoGNP.ipynb`](tiempoGNP.ipynb)

Este notebook repite el problema del Laplaciano 1D y concentra el análisis en el costo computacional. Para cada tamaño de matriz:

1. Construye una matriz dispersa tridiagonal.
2. Define una solución conocida `x_real` y calcula `b = A x_real`.
3. Ejecuta GMRES sin precondicionamiento.
4. Construye un solucionador Smoothed Aggregation AMG.
5. Ejecuta GMRES utilizando AMG como precondicionador.
6. Organiza los tiempos, iteraciones y residuales en una tabla de pandas.

También conserva las pruebas con GMRES+GNP para contrastar sus resultados con AMG. Su propósito es determinar cómo escala cada alternativa al crecer el número de incógnitas.

## 4. Influencia de los suavizadores en Ruge–Stüben

**Archivo:** [`Smoothers.ipynb`](Smoothers.ipynb)

Un suavizador reduce principalmente los errores de alta frecuencia durante las etapas de relajación del método multigrid. El notebook:

1. Lista los suavizadores disponibles en PyAMG y revisa las configuraciones de Ruge–Stüben y Smoothed Aggregation.
2. Construye el sistema disperso del problema de Poisson 2D sobre una malla `64 x 64`, con una matriz de `4096 x 4096`.
3. Compara configuraciones de suavización previa (`presmoother`) y posterior (`postsmoother`):
   - Gauss–Seidel simétrico.
   - Jacobi relajado.
   - SOR.
   - Sin suavizador.
4. Mide el tiempo de construcción, el tiempo de solución, las iteraciones y el residual relativo.
5. Grafica la convergencia en escala logarítmica.
6. Evalúa combinaciones como `GS/GS`, `Jacobi/Jacobi`, `Jacobi/GS`, `GS/Jacobi` y configuraciones sin suavizador.

**Interpretación:** Gauss–Seidel simétrico y SOR muestran una reducción rápida del residual; Jacobi converge de manera más progresiva. La selección independiente del suavizador previo y posterior permite estudiar el equilibrio entre costo por iteración y velocidad de convergencia.

## 5. Formación de la jerarquía Ruge–Stüben

**Archivo:** [`experimentos_ruge_stuben.py`](experimentos_ruge_stuben.py)

Este script estudia internamente cómo PyAMG construye una jerarquía multigrid para problemas de Poisson. Utiliza matrices dispersas y conserva la información de cada nivel (`keep=True`).

Se puede ejecutar todo el conjunto con:

```bash
python experimentos_ruge_stuben.py
```

También se puede ejecutar un experimento individual:

```bash
python experimentos_ruge_stuben.py 1
```

### Experimento 1: tamaño de los niveles

Construye jerarquías para Poisson 1D con `n = 8, 12, 16, 20` y muestra cuántas incógnitas tiene cada nivel. Sirve para observar la reducción progresiva del problema fino al problema grueso.

### Experimento 2: separación C/F y fuerza de conexión

Muestra la matriz de fuerza de conexión y clasifica los puntos en:

- **Puntos C:** variables seleccionadas para el nivel grueso.
- **Puntos F:** variables que permanecen en el nivel fino y se interpolan desde los puntos C.

Sirve para entender cómo Ruge–Stüben selecciona las variables que formarán la jerarquía.

### Experimento 3: operadores de transferencia y producto de Galerkin

Examina los operadores de prolongación `P`, restricción `R` y la matriz del siguiente nivel. Verifica numéricamente:

```text
R = P^T
A_(k+1) = R A_k P
```

También comprueba que las filas de `P` asociadas a puntos C forman la identidad y que la matriz coarse conserva la simetría.

### Experimento 4: proyección del vector derecho

Parte de un vector conocido `b_k = [1, 2, ..., n]` y calcula el vector del siguiente nivel mediante:

```text
b_(k+1) = R_k b_k = P_k^T b_k
```

Sirve para visualizar cómo se transfieren las cargas o el lado derecho del sistema entre niveles.

### Experimento 5: variación de parámetros

Compara cinco configuraciones:

- Configuración base.
- Umbral de fuerza `theta = 0.5`.
- Umbral de fuerza `theta = 0.9`.
- Interpolación directa.
- Ruge–Stüben con segunda pasada.

Para cada configuración reporta el tamaño de los niveles, el número de elementos no nulos, la complejidad del operador y los puntos C. El objetivo es estudiar cómo las decisiones de construcción afectan el costo y la estructura de la jerarquía.

### Experimento 6: Poisson 2D

Repite el análisis para mallas `4x2`, `4x3`, `4x4` y `5x4`. Compara la jerarquía obtenida en dos dimensiones e identifica los puntos C y F de cada nivel.

## Instalación

Se requiere Python 3.9 o posterior y las siguientes bibliotecas:

```bash
python -m pip install numpy scipy pandas matplotlib pyamg jupyter
```

Para ejecutar los notebooks:

```bash
jupyter notebook
```

El notebook de GNP además requiere instalar y ejecutar el proyecto externo que se referencia dentro de sus celdas.

## Flujo general de trabajo

1. Generar el sistema disperso a partir de un problema de Poisson o de una matriz SPD.
2. Analizar sus propiedades espectrales y su número de condición.
3. Resolver con un método iterativo base.
4. Incorporar AMG, suavizadores o GNP como precondicionamiento.
5. Comparar iteraciones, tiempos de ejecución y residuales.
6. Analizar la relación entre la fidelidad de la solución y el costo computacional.

## Estado del proyecto

El repositorio contiene experimentos reproducibles y notebooks de investigación. Los resultados dependen del hardware, las versiones de Python/PyAMG y las tolerancias utilizadas. Los valores incluidos en las tablas de los notebooks deben interpretarse como resultados de referencia, no como benchmarks universales.
