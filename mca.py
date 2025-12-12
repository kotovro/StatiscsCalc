import numpy as np
import pandas as pd

# -------------------------------------------------------
# 1. Загрузка данных
# -------------------------------------------------------
def load_data(filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(filename)
    else:
        raise ValueError("Поддерживаются только CSV и XLSX.")
    return df


# -------------------------------------------------------
# 2. Индикаторная матрица
# -------------------------------------------------------
def indicator_matrix(df):
    return pd.get_dummies(df).astype(float)


# -------------------------------------------------------
# 3. Burt-матрица
# -------------------------------------------------------
def burt_matrix(Z):
    return Z.T @ Z


# -------------------------------------------------------
# 4. χ²-нормировка (как в STATISTICA)
# -------------------------------------------------------
def normalized_matrix(B):
    total = B.values.sum()

    # массы категорий
    masses = B.sum(axis=1) / total

    # частоты
    P = B / total

    # центрирование
    S = P - np.outer(masses, masses)

    # M^(-1/2) с защитой от деления на 0
    Minv_sqrt = np.diag(1.0 / np.sqrt(np.where(masses > 0, masses, 1)))

    S_norm = Minv_sqrt @ S @ Minv_sqrt

    return S_norm, masses, Minv_sqrt


# -------------------------------------------------------
# 5. Собственное разложение
# -------------------------------------------------------
def eigen_decomposition(S_norm):
    eigvals, eigvecs = np.linalg.eigh(S_norm)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


# -------------------------------------------------------
# 6. Координаты категорий
# -------------------------------------------------------
def coordinates(eigvals, eigvecs, Minv_sqrt):
    return Minv_sqrt @ eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))


# -------------------------------------------------------
# 7. Сохранение
# -------------------------------------------------------
def save_results(eigvals, coords, categories):
    pd.DataFrame({"eigenvalue": eigvals}).to_csv("mca_eigenvalues.csv", index=False)
    pd.DataFrame(coords, index=categories).to_csv("mca_coordinates.csv")

    print("Файлы сохранены:")
    print(" → mca_eigenvalues.csv")
    print(" → mca_coordinates.csv")


# -------------------------------------------------------
# 8. Главная функция
# -------------------------------------------------------
def MCA_from_file(filename):
    df = load_data(filename)

    Z = indicator_matrix(df)
    B = burt_matrix(Z)

    S_norm, masses, Minv_sqrt = normalized_matrix(B)
    eigvals, eigvecs = eigen_decomposition(S_norm)
    coords = coordinates(eigvals, eigvecs, Minv_sqrt)

    save_results(eigvals, coords, Z.columns)


# -------------------------------------------------------
# Запуск
# -------------------------------------------------------
if __name__ == "__main__":
    MCA_from_file("input_example.csv")
