import numpy as np
import matplotlib.pyplot as plt

# Параметры робота
r = 0.12       # радиус колеса, м
L = 0.85       # колесная база, м
m = 400        # масса робота, кг
mu = 0.02      # коэффициент трения качения
I = 64         # момент инерции, кг·м²
g = 9.81       # ускорение свободного падения, м/с²

# Силы и моменты (пример)
F_traction = 500   # сила тяги, Н
M_rotation = 100   # крутящий момент при повороте, Н·м

# Временной шаг и длительность моделирования
dt = 0.01
t_end = 10
time = np.arange(0, t_end, dt)

# === КИНЕМАТИКА ===
def differential_drive_kinematics(omega_L, omega_R):
    v = r / 2 * (omega_L + omega_R)
    omega = (r / L) * (omega_R - omega_L)
    return v, omega

# === ДИНАМИКА ===
def calculate_forces_and_torques():
    F_friction = mu * m * g
    M_friction = mu * m * g * L / 2
    a = (F_traction - F_friction) / m
    alpha = (M_rotation - M_friction) / I
    return F_friction, M_friction, a, alpha

# === МОДЕЛИРОВАНИЕ ===
def simulate_motion(omega_L, omega_R):
    x = 0
    y = 0
    theta = 0

    trajectory_x = []
    trajectory_y = []

    for t in time:
        v, omega = differential_drive_kinematics(omega_L, omega_R)

        # Обновляем координаты и ориентацию
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        trajectory_x.append(x)
        trajectory_y.append(y)

    return trajectory_x, trajectory_y

# === ОСНОВНОЙ БЛОК ===
if __name__ == "__main__":
    # Задаем угловые скорости колес (рад/с)
    omega_L = 10  # рад/с
    omega_R = 10  # рад/с

    # Рассчитываем кинематику
    v, omega = differential_drive_kinematics(omega_L, omega_R)
    print(f"Линейная скорость: {v:.3f} м/с")
    print(f"Угловая скорость: {omega:.3f} рад/с")

    # Рассчитываем динамику
    F_fric, M_fric, a, alpha = calculate_forces_and_torques()
    print(f"\nСила трения: {F_fric:.2f} Н")
    print(f"Момент трения: {M_fric:.2f} Н·м")
    print(f"Линейное ускорение: {a:.3f} м/с²")
    print(f"Угловое ускорение: {alpha:.3f} рад/с²")

    # Моделируем движение
    traj_x, traj_y = simulate_motion(omega_L, omega_R)

    # График траектории
    plt.figure(figsize=(8, 6))
    plt.plot(traj_x, traj_y, label="Траектория робота")
    plt.title("Траектория движения робота")
    plt.xlabel("X, м")
    plt.ylabel("Y, м")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()