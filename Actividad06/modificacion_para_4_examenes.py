import random
import numpy as np
import pandas as pd

# ====================== LECTURA DE DATOS ======================
df = pd.read_csv('../notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

# ====================== CONFIGURACIÓN GENÉTICA ======================
NUM_ALUMNOS = 39
NUM_EXAMENES = 4  # Cambio importante: ahora tenemos 4 exámenes
GENES_POR_ALUMNO = NUM_EXAMENES  # 4 bits por alumno
LONGITUD_CROMOSOMA = NUM_ALUMNOS * GENES_POR_ALUMNO

# ====================== FUNCIONES ======================
def crear_cromosoma():
    cromosoma = []
    for _ in range(NUM_ALUMNOS):
        examen = random.randint(0, NUM_EXAMENES - 1)
        genes = [0] * NUM_EXAMENES
        genes[examen] = 1
        cromosoma.extend(genes)
    return cromosoma

def decodificar_cromosoma(cromosoma):
    asignaciones = {ex: [] for ex in ['A', 'B', 'C', 'D']}
    examenes = list(asignaciones.keys())
    for i in range(NUM_ALUMNOS):
        idx = i * GENES_POR_ALUMNO
        for j in range(NUM_EXAMENES):
            if cromosoma[idx + j] == 1:
                asignaciones[examenes[j]].append(i)
                break
    return asignaciones

def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    tamaños = sorted([len(asignaciones[ex]) for ex in asignaciones])
    
    # Distribución más equitativa con 39 alumnos en 4 grupos: [9, 10, 10, 10]
    if tamaños != [9, 10, 10, 10]:
        return -1000

    promedios = []
    for examen in asignaciones:
        notas_examen = [notas[i] for i in asignaciones[examen]]
        promedios.append(np.mean(notas_examen))

    desviacion = np.std(promedios)
    return -desviacion  # Menor desviación es mejor

def mutacion(cromosoma):
    crom_mutado = cromosoma.copy()
    alumno1, alumno2 = random.sample(range(NUM_ALUMNOS), 2)
    idx1, idx2 = alumno1 * GENES_POR_ALUMNO, alumno2 * GENES_POR_ALUMNO

    ex1 = cromosoma[idx1:idx1 + GENES_POR_ALUMNO].index(1)
    ex2 = cromosoma[idx2:idx2 + GENES_POR_ALUMNO].index(1)

    if ex1 != ex2:
        crom_mutado[idx1:idx1 + GENES_POR_ALUMNO] = [0] * GENES_POR_ALUMNO
        crom_mutado[idx1 + ex2] = 1
        crom_mutado[idx2:idx2 + GENES_POR_ALUMNO] = [0] * GENES_POR_ALUMNO
        crom_mutado[idx2 + ex1] = 1

    return crom_mutado

def algoritmo_genetico(generaciones=100, tam_poblacion=50):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]

    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        nueva_poblacion = [fs[0] for fs in fitness_scores[:int(tam_poblacion * 0.2)]]

        while len(nueva_poblacion) < tam_poblacion:
            padre = random.choice(nueva_poblacion)
            hijo = mutacion(padre)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

        if gen % 20 == 0:
            print(f"Generación {gen}: Mejor fitness = {fitness_scores[0][1]:.4f}")

    return fitness_scores[0][0]

# ====================== EJECUCIÓN PRINCIPAL ======================
print("REPRESENTACIÓN BINARIA - 4 EXÁMENES")
print("Problema: Distribuir 39 alumnos en 4 exámenes (A, B, C, D)")
print("Cromosoma: 156 bits (39 alumnos × 4 bits cada uno)")
print("Gen: [0,0,1,0] significa alumno asignado al examen C\n")

mejor_solucion = algoritmo_genetico()
asignaciones_finales = decodificar_cromosoma(mejor_solucion)

print("\nDistribución final:")
promedios = []
for examen in ['A', 'B', 'C', 'D']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    promedios.append(promedio)
    print(f"Examen {examen}: {len(indices)} alumnos, promedio = {promedio:.2f}")
    print(f"  Ejemplos: {[alumnos[i] for i in indices[:5]]}...")

print(f"\nDesviación estándar entre promedios: {np.std(promedios):.4f}")

