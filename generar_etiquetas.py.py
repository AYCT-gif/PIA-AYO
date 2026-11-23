import os
import math
import time
import random
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

def calcular_distancia_matriz(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def costo_tour(tour, dist_matrix):
    return sum(dist_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def two_opt_local_search(tour, dist_matrix):
    best_tour = tour[:]
    best_cost = costo_tour(best_tour, dist_matrix)
    improvements = 0
    n = len(tour)
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_cost = costo_tour(new_tour, dist_matrix)
                if new_cost < best_cost - 1e-6:
                    best_tour = new_tour
                    best_cost = new_cost
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
    return best_tour, best_cost, improvements

def simulated_annealing(dist_matrix, schedule='exp', t0=100.0, tf=0.1, alpha=0.995, beta=0.1):
    n = len(dist_matrix)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_cost = costo_tour(current_tour, dist_matrix)
    best_tour, best_cost = current_tour[:], current_cost

    t = t0
    while t > tf:
        i, j = sorted(random.sample(range(n), 2))
        neighbor_tour = current_tour[:i] + current_tour[i:j+1][::-1] + current_tour[j+1:]
        neighbor_cost = costo_tour(neighbor_tour, dist_matrix)

        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / t):
            current_tour = neighbor_tour
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost

        if schedule == 'exp':
            t *= alpha
        elif schedule == 'linear':
            t -= beta

    return best_tour, best_cost

def mst_approximation_tsp(dist_matrix):
    mst = minimum_spanning_tree(dist_matrix).toarray()
    n = len(dist_matrix)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0 or mst[j, i] > 0:
                adj[i].append(j)

    visited = []
    stack = [0]
    seen = set()
    while stack:
        u = stack.pop()
        if u not in seen:
            seen.add(u)
            visited.append(u)
            for v in reversed(adj[u]):
                if v not in seen:
                    stack.append(v)

    cost = costo_tour(visited, dist_matrix)
    return visited, cost


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "hh_tsp_dataset", "data")
INSTANCES_DIR = os.path.join(DATA_DIR, "instances")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SA_PARAMS = {
    "schedule": "exp",
    "t0": 100.0,
    "tf": 0.1,
    "alpha": 0.995
}

def cargar_coordenadas(instance_id):
    ruta = os.path.join(INSTANCES_DIR, f"{instance_id}.csv")
    df_coords = pd.read_csv(ruta)
    return df_coords.iloc[:, :2].to_numpy(dtype=float)

def etiquetar_split(csv_nombre, seed_base=1899555):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_nombre))
    c_2opt_list, c_sa_list, c_mst_list, best_list = [], [], [], []

    t_inicio = time.perf_counter()
    for idx, row in df.iterrows():
        inst_id = row["instance_id"]
        coords = cargar_coordenadas(inst_id)
        dist_matrix = calcular_distancia_matriz(coords)
        n = len(coords)

        semilla = seed_base + int(inst_id.split("_")[1])
        random.seed(semilla)
        np.random.seed(semilla)

        rand_tour = list(range(n))
        random.shuffle(rand_tour)
        _, cost_2opt, _ = two_opt_local_search(rand_tour, dist_matrix)

        _, cost_sa = simulated_annealing(dist_matrix, **SA_PARAMS)

        tour_mst, _ = mst_approximation_tsp(dist_matrix)
        _, cost_mst, _ = two_opt_local_search(tour_mst, dist_matrix)

        costos = {
            "2opt_hill_climbing": round(cost_2opt, 8),
            "sa_2opt": round(cost_sa, 8),
            "mst_2opt": round(cost_mst, 8)
        }
        best_alg = min(costos, key=costos.get)

        c_2opt_list.append(cost_2opt)
        c_sa_list.append(cost_sa)
        c_mst_list.append(cost_mst)
        best_list.append(best_alg)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"  [{csv_nombre}] Procesadas {idx + 1}/{len(df)} instancias ({time.perf_counter() - t_inicio:.1f}s)")

    df["cost_2opt_hill_climbing"] = c_2opt_list
    df["cost_sa_2opt"] = c_sa_list
    df["cost_mst_2opt"] = c_mst_list
    df["best_algorithm"] = best_list
    return df

if __name__ == "__main__":
    print("=== GENERANDO ETIQUETAS ===")
    print("Parámetros SA utilizados:", SA_PARAMS)

    df_train = etiquetar_split("train.csv")
    df_test = etiquetar_split("test.csv")

    df_train.to_csv(os.path.join(OUTPUT_DIR, "train_labeled.csv"), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "test_labeled.csv"), index=False)

    df_total = pd.concat([df_train, df_test], ignore_index=True)
    print("\n=== BALANCE DE CLASES TOTAL (500 INSTANCIAS) ===")
    conteo = df_total["best_algorithm"].value_counts()
    porcentaje = df_total["best_algorithm"].value_counts(normalize=True) * 100
    resumen_clases = pd.DataFrame({"Instancias": conteo, "Porcentaje (%)": porcentaje.round(2)})
    print(resumen_clases)
    print("\nArchivos guardados en 'outputs/train_labeled.csv' y 'outputs/test_labeled.csv'.")