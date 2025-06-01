import numpy as np
import matplotlib.pyplot as plt

# Константы
g = 9.81  # Ускорение свободного падения (м/с²)

def calculate_min_radius(v, mu):
    """
    Рассчитывает минимальный радиус поворота, чтобы избежать проскальзывания колёс.
    
    Аргументы:
    v  -- линейная скорость робота (м/с)
    mu -- коэффициент трения между колёсами и поверхностью
    
    Возвращает:
    R_min -- минимальный радиус поворота (м)
    """
    if mu <= 0:
        raise ValueError("Коэффициент трения должен быть положительным")
    
    R_min = v**2 / (mu * g)
    return R_min

def plot_radius_vs_speed(mu_values, speed_range=(0.1, 2.0), num_points=100):
    """
    Строит график зависимости минимального радиуса поворота от скорости для разных коэффициентов трения.
    
    Аргументы:
    mu_values    -- список коэффициентов трения для построения графика
    speed_range  -- диапазон скоростей (min, max)
    num_points   -- количество точек для построения графика
    """
    speeds = np.linspace(speed_range[0], speed_range[1], num_points)
    
    plt.figure(figsize=(10, 6))
    
    for mu in mu_values:
        radii = [calculate_min_radius(v, mu) for v in speeds]
        plt.plot(speeds, radii, label=f'μ = {mu}')
    
    plt.title('Минимальный радиус поворота в зависимости от скорости')
    plt.xlabel('Скорость (м/с)')
    plt.ylabel('Минимальный радиус поворота (м)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Пример использования
if __name__ == "__main__":
    # Параметры
    speed = 1.0           # Скорость робота (м/с)
    mu = 0.6              # Коэффициент трения (резина по бетону)
    
    # Расчет минимального радиуса
    try:
        R_min = calculate_min_radius(speed, mu)
        print(f"Минимальный радиус поворота (R_min): {R_min:.3f} м")
        print(f"Радиус поворота должен быть не меньше этой величины, чтобы избежать проскальзывания.")
    except ValueError as e:
        print("Ошибка:", str(e))
    
    # Построение графика зависимости радиуса от скорости
    mu_values = [0.3, 0.5, 0.7]  # разные коэффициенты трения
    plot_radius_vs_speed(mu_values)