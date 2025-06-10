import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.stats import expon, norm

# Настройка стиля для презентации
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    }
)

# Создаем компактную фигуру
fig, ax = plt.subplots(figsize=(10, 3.5))
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)  # Уменьшаем отступы

# Убираем оси
ax.axis("off")

# Устанавливаем границы для компактного размещения
ax.set_xlim(0.1, 0.9)
ax.set_ylim(0.05, 0.95)

# Рисуем нормальное распределение (левая часть)
x_left = np.linspace(0.15, 0.45, 200)
pdf_normal = norm.pdf(x_left, loc=0.3, scale=0.05)
pdf_normal = pdf_normal / np.max(pdf_normal) * 0.25  # Нормализуем и масштабируем
ax.plot(x_left, pdf_normal + 0.6, "b-", lw=2)
ax.text(0.3, 0.88, r"Гауссовская модель $\mathcal{N}(\mu,\sigma^2)$", ha="center", fontsize=14)

# Рисуем экспоненциальное распределение (правая часть)
x_right = np.linspace(0.55, 0.85, 200)
pdf_expon = expon.pdf(x_right, loc=0.55, scale=0.07)
pdf_expon = pdf_expon / np.max(pdf_expon) * 0.25  # Нормализуем и масштабируем
ax.plot(x_right, pdf_expon + 0.6, "r-", lw=2)
ax.text(0.7, 0.88, r"Экспоненциальная модель $\mathrm{Exp}(\lambda)$", ha="center", fontsize=14)

# Настройка стрелок
arrow_props = dict(arrowstyle="->", color="gray", lw=1.8, mutation_scale=25)

# Стрелки от нормального распределения
arrow1 = FancyArrowPatch((0.25, 0.6), (0.15, 0.2), **arrow_props)
arrow2 = FancyArrowPatch((0.3, 0.6), (0.3, 0.2), **arrow_props)
arrow3 = FancyArrowPatch((0.35, 0.6), (0.45, 0.2), **arrow_props)
ax.add_patch(arrow1)
ax.add_patch(arrow2)
ax.add_patch(arrow3)

# Стрелки от экспоненциального распределения
arrow4 = FancyArrowPatch((0.65, 0.6), (0.55, 0.2), **arrow_props)
arrow5 = FancyArrowPatch((0.75, 0.6), (0.85, 0.2), **arrow_props)
ax.add_patch(arrow4)
ax.add_patch(arrow5)

# Подписи распределений снизу с математическими символами
ax.text(
    0.15,
    0.15,
    r"$\mathcal{B}(\alpha,\beta)$",
    ha="center",
    fontsize=16,
    bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
)
ax.text(
    0.3,
    0.15,
    r"$\mathcal{N}(\mu,\sigma^2)$",
    ha="center",
    fontsize=16,
    bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
)
ax.text(0.45, 0.15, r"$U(a,b)$", ha="center", fontsize=16, bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"))
ax.text(
    0.55, 0.15, r"$Exp(\lambda)$", ha="center", fontsize=16, bbox=dict(facecolor="white", alpha=0.0, edgecolor="none")
)
ax.text(
    0.85, 0.15, r"$W(k,\lambda)$", ha="center", fontsize=16, bbox=dict(facecolor="white", alpha=0.0, edgecolor="none")
)

# Добавляем общее название схемы
ax.text(0.5, 0.97, "Моделирование распределений", ha="center", fontsize=16, weight="bold")

# Сохраняем в PDF
plt.savefig("distributions_scheme.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close()
