import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.worker import Worker
from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.heuristic_gaussian_vs_exponential import (
    HeuristicGaussianVsExponential,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider


def construct_algorithm(threshold, learning_sample_size):
    likelihood = HeuristicGaussianVsExponential()
    return BayesianOnline(
        learning_sample_size=learning_sample_size,
        likelihood=likelihood,
        hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
        detector=ThresholdDetector(threshold=threshold),
        localizer=ArgmaxLocalizer(),
    ), likelihood


# Загрузка данных
data = pd.read_csv("").to_numpy()
data_provider = ListUnivariateProvider(data=list(data))

# Загрузка ожидаемых точек изменений
expected_change_points = pd.read_csv("").to_numpy()

# Создание и запуск алгоритма
algorithm, likelihood = construct_algorithm(threshold=0.00001, learning_sample_size=20)
cpd_core = OnlineCpdCore(algorithm=algorithm, data_provider=data_provider)
worker = Worker(online_cpd=cpd_core)

cp_results, cpd_time = worker.run()

localized_change_points = [result.change_point for result in cp_results]
detection_times = [result.change_point + result.delay for result in cp_results]

print("Expected change points:", expected_change_points)
print("Localized change points:", localized_change_points)
print("Detection times:", detection_times)


def calculate_metrics(expected_cps, localized_cps, detection_times, window=25):
    """
    Рассчитывает метрики детектирования точек изменений.

    Параметры:
    expected_cps -- список ожидаемых точек изменений
    localized_cps -- список обнаруженных точек изменений (локализация)
    detection_times -- список времен детектирования
    window -- размер окрестности для сопоставления точек

    Возвращает:
    metrics -- словарь с метриками TP, FP, FN и задержками
    """
    # Преобразуем в массивы NumPy для удобства
    expected = np.array(expected_cps).flatten()
    localized = np.array(localized_cps)
    detections = np.array(detection_times)

    # Инициализация результатов
    TP = 0
    FP = 0
    FN = 0
    delays = []
    matched_expected = []  # Сопоставленные ожидаемые точки
    matched_localized = []  # Сопоставленные обнаруженные точки

    # Создаем копии для отслеживания необработанных точек
    remaining_expected = expected.copy()
    remaining_localized = localized.copy()
    remaining_detections = detections.copy()

    # Проход по всем ожидаемым точкам изменений
    for exp in expected:
        # Находим все обнаруженные точки в окрестности текущей ожидаемой
        in_window_mask = (remaining_localized >= exp - window) & (remaining_localized <= exp + window)
        candidates = remaining_detections[in_window_mask]

        if len(candidates) > 0:
            # Выбираем кандидата с минимальной задержкой
            best_idx = np.argmin(candidates - exp)
            best_detection = candidates[best_idx]

            # Проверяем условие: время детектирования должно быть после ожидаемой точки
            if best_detection >= exp:
                # Рассчитываем задержку
                delay = best_detection - exp
                delays.append(int(delay))

                # Увеличиваем счетчик TP
                TP += 1

                # Удаляем сопоставленную точку
                candidate_idx = np.where(remaining_detections == best_detection)[0][0]
                remaining_localized = np.delete(remaining_localized, candidate_idx)
                remaining_detections = np.delete(remaining_detections, candidate_idx)

                # Сохраняем для отладки/визуализации
                matched_expected.append(exp)
                matched_localized.append(
                    remaining_localized[candidate_idx] if candidate_idx < len(remaining_localized) else -1
                )
                continue

        # Если не нашли подходящих кандидатов
        FN += 1

    # Все оставшиеся обнаруженные точки считаем FP
    FP = len(remaining_localized)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "delays": delays,
        "precision": TP / (TP + FP) if TP + FP > 0 else 0,
        "recall": TP / (TP + FN) if TP + FN > 0 else 0,
        "f-score": 2 * TP / (2 * TP + FP + FN),
        "matched_expected": list(map(int, matched_expected)),
        "matched_localized": list(map(int, matched_localized)),
        "unmatched_expected": list(map(int, list(set(expected) - set(matched_expected)))),
        "unmatched_localized": list(map(int, list(remaining_localized))),
    }


# Пример использования
metrics = calculate_metrics(expected_change_points, localized_change_points, detection_times, window=25)

print(f"TP: {metrics['TP']}")
print(f"FP: {metrics['FP']}")
print(f"FN: {metrics['FN']}")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"Delays: {metrics['delays']}")
print(f"F-score: {metrics['f-score']}")
print(f"Время: {cpd_time}")
print(f"Не сопоставленные размеченные: {metrics['unmatched_expected']}")
print(f"Ошибочно локализованные: {metrics['unmatched_localized']}")


# Визуализация
plt.figure(figsize=(16, 9), dpi=100)

# 1. Основной график данных
plt.plot(data, color="royalblue", alpha=0.7, linewidth=0.8, label="Данные")

# 2. Ожидаемые точки изменений
for cp in expected_change_points:
    plt.axvline(x=cp, color="green", linestyle="-", linewidth=1.5, alpha=0.7, label="Ожидаемые разладки")

# 3. Обнаруженные точки изменений (оценка алгоритма)
for cp in localized_change_points:
    plt.axvline(x=cp, color="purple", linestyle="--", linewidth=1.5, alpha=0.9, label="Обнаруженные разладки")

# 4. Времена детектирования
# for dt in detection_times:
#     plt.axvline(x=dt, color='red', linestyle=':', linewidth=1.8, alpha=0.9, label='Время детектирования')

# Улучшение отображения для большого объема данных
plt.title("Локализация разладок во временном ряде", fontsize=16)
plt.xlabel("Временной индекс", fontsize=14)
plt.ylabel("Значение", fontsize=14)
plt.grid(True, alpha=0.3)

# Обработка легенды без дубликатов
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=12, loc="upper right")

# Добавление шкалы для улучшения навигации
plt.axvspan(0, len(data), facecolor="none", edgecolor="gray", alpha=0.1)

# Интерактивные элементы для навигации
plt.tight_layout()

# Дополнительный зум на первые 1000 точек для детального просмотра
plt.figure(figsize=(16, 6), dpi=100)
zoom_range = 1000  # Первые 1000 точек
plt.plot(data[:zoom_range], color="royalblue", alpha=0.7, linewidth=1.0)

# Отметки в увеличенной области
for cp in expected_change_points:
    if cp < zoom_range:
        plt.axvline(x=cp, color="green", linestyle="-", linewidth=1.5, alpha=0.7)

for cp in localized_change_points:
    if cp < zoom_range:
        plt.axvline(x=cp, color="purple", linestyle="--", linewidth=1.5, alpha=0.9)

for dt in detection_times:
    if dt < zoom_range:
        plt.axvline(x=dt, color="red", linestyle=":", linewidth=1.8, alpha=0.9)

plt.title(f"Детализированный вид (первые {zoom_range} точек)", fontsize=14)
plt.xlabel("Временной индекс", fontsize=12)
plt.ylabel("Значение", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
