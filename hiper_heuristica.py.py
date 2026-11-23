import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

df_train = pd.read_csv(os.path.join(OUTPUT_DIR, "train_labeled.csv"))
df_test = pd.read_csv(os.path.join(OUTPUT_DIR, "test_labeled.csv"))

df_all = pd.concat([df_train, df_test], ignore_index=True)
print("=== 1. BALANCE DE CLASES (500 INSTANCIAS) ===")
for alg, cnt in df_all["best_algorithm"].value_counts().items():
    print(f"  {alg:<20}: {cnt:3d} ({cnt / len(df_all) * 100:.2f}%)")

clase_mayoritaria = df_train["best_algorithm"].value_counts().idxmax()
y_train = df_train["best_algorithm"]
y_test = df_test["best_algorithm"]

y_pred_baseline = np.full(len(y_test), clase_mayoritaria)
acc_baseline = accuracy_score(y_test, y_pred_baseline)

print(f"\nClase mayoritaria en train: '{clase_mayoritaria}'")
print(f"Accuracy Baseline Trivial en test: {acc_baseline * 100:.2f}%")

MAPA_COLUMNAS_COSTO = {
    "2opt_hill_climbing": "cost_2opt_hill_climbing",
    "sa_2opt": "cost_sa_2opt",
    "mst_2opt": "cost_mst_2opt"
}

def calcular_regret(df_eval, predicciones):
    costos_elegidos = []
    costos_optimos = df_eval[["cost_2opt_hill_climbing", "cost_sa_2opt", "cost_mst_2opt"]].min(axis=1).to_numpy()

    for idx, alg_pred in enumerate(predicciones):
        col = MAPA_COLUMNAS_COSTO[alg_pred]
        costos_elegidos.append(df_eval.iloc[idx][col])

    costos_elegidos = np.array(costos_elegidos)
    regrets = costos_elegidos - costos_optimos
    regret_relativo = (regrets / costos_optimos) * 100
    return regrets.mean(), regrets.max(), regret_relativo.mean()

reg_base_mean, reg_base_max, reg_base_rel = calcular_regret(df_test, y_pred_baseline)
print(f"Regret Promedio Baseline Trivial:  {reg_base_mean:.5f} (Relativo: {reg_base_rel:.3f}%)")

num_features = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
cat_features = ["distribution"]
features = num_features + cat_features

X_train = df_train[features]
X_test = df_test[features]

preprocesador = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop=None, sparse_output=False), cat_features)
    ]
)

modelos = {
    "Arbol_Decision (max_depth=4)": Pipeline([
        ("prep", preprocesador),
        ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42))
    ]),
    "KNN (k=7, distance)": Pipeline([
        ("prep", preprocesador),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
    ]),
    "Random_Forest (n=100, depth=5)": Pipeline([
        ("prep", preprocesador),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
}

clases_ordenadas = sorted(df_all["best_algorithm"].unique())

print("\n=== EVALUACIÓN EN CONJUNTO DE PRUEBA ===")
resultados_modelos = {}

for nombre, pipe in modelos.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    reg_mean, reg_max, reg_rel = calcular_regret(df_test, y_pred)
    resultados_modelos[nombre] = (pipe, y_pred, acc, reg_mean)

    print(f"\n--- {nombre} ---")
    print(f"Accuracy:        {acc * 100:.2f}% (vs. Baseline: {acc_baseline * 100:.2f}%)")
    print(f"Regret Promedio: {reg_mean:.5f} (vs. Baseline: {reg_base_mean:.5f}) | Máx: {reg_max:.5f} | Relativo: {reg_rel:.3f}%")
    print("\nReporte por clase:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=clases_ordenadas)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_ordenadas)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.xticks(rotation=15)
    plt.tight_layout()
    nombre_archivo = nombre.split()[0].lower()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{nombre_archivo}.png"), dpi=300)
    plt.close()

dt_pipe = resultados_modelos["Arbol_Decision (max_depth=4)"][0]
ohe_cols = list(dt_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_features))
nombres_features_final = num_features + ohe_cols
importancias = dt_pipe.named_steps["clf"].feature_importances_

df_imp = pd.DataFrame({
    "Caracteristica": nombres_features_final,
    "Importancia": importancias
}).sort_values(by="Importancia", ascending=False)

print("\n=== IMPORTANCIA DE CARACTERÍSTICAS (ÁRBOL DE DECISIÓN) ===")
print(df_imp.to_string(index=False))

print("\nReglas del Árbol de Decisión:")
print(export_text(dt_pipe.named_steps["clf"], feature_names=nombres_features_final))

plt.figure(figsize=(8, 4.5))
plt.barh(df_imp["Caracteristica"][::-1], df_imp["Importancia"][::-1], color="teal")
plt.xlabel("Importancia (Gini)")
plt.title("Importancia de Características para Predecir el Algoritmo Ganador")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "importancia_caracteristicas.png"), dpi=300)
plt.close()

print("\n=== PATRÓN: ALGORITMO GANADOR POR DISTRIBUCIÓN (%) ===")
tabla_dist = pd.crosstab(df_all["distribution"], df_all["best_algorithm"], normalize="index") * 100
print(tabla_dist.round(2))

print("\n=== PATRÓN: ALGORITMO GANADOR POR TAMAÑO n (%) ===")
tabla_n = pd.crosstab(df_all["n"], df_all["best_algorithm"], normalize="index") * 100
print(tabla_n.round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
tabla_dist.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set2", edgecolor="black")
axes[0].set_title("Algoritmo Ganador por Distribución Espacial")
axes[0].set_ylabel("Porcentaje de Instancias (%)")
axes[0].set_xlabel("Distribución")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(title="Algoritmo")

tabla_n.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2", edgecolor="black")
axes[1].set_title("Algoritmo Ganador por Tamaño de Problema (n)")
axes[1].set_ylabel("Porcentaje de Instancias (%)")
axes[1].set_xlabel("Número de Ciudades (n)")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Algoritmo")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "patrones_distribucion_tamano.png"), dpi=300)
plt.close()

print("\nGráficas exportadas en 'outputs/': cm_*.png, importancia_caracteristicas.png y patrones_distribucion_tamano.png.")