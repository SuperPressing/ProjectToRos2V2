import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import ast

# === ПАРАМЕТРЫ РОБОТА ===
r = 0.05       # радиус колеса, м
L = 0.35       # колёсная база, м
m = 5        # масса робота, кг
mu = 0.02      # коэффициент трения качения
I = 64         # момент инерции, кг·м²
g = 9.81       # ускорение свободного падения, м/с²

# === ВХОДНОЙ ПУТЬ (x, y) ===
file_path = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/trac.txt'
with open(file_path, 'r') as file:
    data_str = file.read().strip()

# Преобразуем строку в список кортежей
data_list = ast.literal_eval(data_str)

# Преобразуем в numpy массив
path = np.array(data_list, dtype=np.float64)

# === ПАРАМЕТРЫ ДВИЖЕНИЯ ===
v_max = 5     # максимальная линейная скорость, м/с
a = 0.5        # ускорение, м/с²

# === РАСЧЁТ ДЛИНЫ ПУТИ ===
distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
total_length = np.sum(distances)

# === РАСЧЁТ ВРЕМЕННЫХ ФАЗ ===
t_accel = v_max / a
s_accel = 0.5 * a * t_accel**2

if 2 * s_accel > total_length:
    # Треугольный профиль (не достигаем v_max)
    t_accel = np.sqrt(total_length / a)
    t_constant = 0
    t_total = 2 * t_accel
else:
    # Трапециидальный профиль
    s_constant = total_length - 2 * s_accel
    t_constant = s_constant / v_max
    t_total = 2 * t_accel + t_constant

# === ГЕНЕРАЦИЯ ВРЕМЕННОГО ШАГА ===
dt = 0.1
t_eval = np.arange(0, t_total, dt)

# === ГЕНЕРАЦИЯ ПРОФИЛЯ СКОРОСТИ (трапециидальный) ===
v_profile = []
for t in t_eval:
    if t <= t_accel:
        v = a * t
    elif t <= t_accel + t_constant:
        v = v_max
    else:
        v = a * (t_total - t)
    v_profile.append(v)

# === РАСЧЁТ ЛИНЕЙНОГО УСКОРЕНИЯ ===
a_profile = np.gradient(v_profile, dt)

# === ИНТЕРПОЛЯЦИЯ ПУТИ ПО ДЛИНЕ ДУГИ ===
cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

x_path = interp1d(cumulative_distances, path[:, 0], kind='linear', fill_value="extrapolate")
y_path = interp1d(cumulative_distances, path[:, 1], kind='linear', fill_value="extrapolate")

# === РАСЧЁТ ТРАЕКТОРИИ ПО ПРОЙДЕННОМУ ПУТИ ===
s_profile = np.cumsum(v_profile) * dt
trajectory_x = []
trajectory_y = []

for s in s_profile:
    trajectory_x.append(x_path(s))
    trajectory_y.append(y_path(s))

# === РАСЧЁТ НАПРАВЛЕНИЯ РОБОТА И УГЛОВОЙ СКОРОСТИ ===
dx = np.gradient(trajectory_x, dt)
dy = np.gradient(trajectory_y, dt)
theta = np.arctan2(dy, dx)
omega_profile = np.gradient(theta, dt)

# === КИНЕМАТИКА → УГЛОВЫЕ СКОРОСТИ КОЛЁС ===
def inverse_kinematics(v, omega):
    omega_L = (v - (L / 2) * omega) / r
    omega_R = (v + (L / 2) * omega) / r
    return omega_L, omega_R

omega_L_profile = []
omega_R_profile = []

for v, omega in zip(v_profile, omega_profile):
    wl, wr = inverse_kinematics(v, omega)
    omega_L_profile.append(wl)
    omega_R_profile.append(wr)

# === ДИНАМИКА → СИЛЫ И МОМЕНТЫ ===
F_traction = m * a_profile + mu * m * g
M_rotation = I * np.gradient(omega_profile, dt) + mu * m * g * L / 2

# === ВИЗУАЛИЗАЦИЯ ===
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

# 1. Траектория
axs[0, 0].plot(path[:, 0], path[:, 1], 'ro-', label='Путь')
axs[0, 0].plot(trajectory_x, trajectory_y, 'b--', label='Траектория')
axs[0, 0].set_title("Траектория движения")
axs[0, 0].grid(True)
axs[0, 0].legend()
axs[0, 0].axis('equal')

# Добавляем стрелочки ориентации
step = max(1, len(trajectory_x) // 20)
arrow_length = 0.3
for i in range(0, len(trajectory_x), step):
    x = trajectory_x[i]
    y = trajectory_y[i]
    angle = theta[i]
    dx_arrow = arrow_length * np.cos(angle)
    dy_arrow = arrow_length * np.sin(angle)
    axs[0, 0].arrow(x, y, dx_arrow, dy_arrow,
                    head_width=0.05, length_includes_head=True, color='blue')

# 2. Скорость
axs[0, 1].plot(t_eval, v_profile, 'g-', label='v(t)')
axs[0, 1].set_title("Линейная скорость")
axs[0, 1].grid(True)
axs[0, 1].legend()

# 3. Ускорение
axs[1, 0].plot(t_eval, a_profile, 'm-', label='a(t)')
axs[1, 0].set_title("Линейное ускорение")
axs[1, 0].grid(True)
axs[1, 0].legend()

# 4. Ориентация робота
axs[1, 1].plot(t_eval, theta, 'c-', label='θ(t)')
axs[1, 1].set_title("Ориентация робота в пространстве (радианы)")
axs[1, 1].grid(True)
axs[1, 1].legend()

# 5. Угловые скорости колёс
axs[2, 0].plot(t_eval, omega_L_profile, 'b', label='ω_L')
axs[2, 0].plot(t_eval, omega_R_profile, 'r', label='ω_R')
axs[2, 0].set_title("Угловые скорости колёс")
axs[2, 0].grid(True)
axs[2, 0].legend()

# 6. Сила тяги
axs[2, 1].plot(t_eval, F_traction, 'purple', label='F_тяги(t)')
axs[2, 1].set_title("Необходимая сила тяги")
axs[2, 1].grid(True)
axs[2, 1].legend()

trajectory_dict = []
for i in range(len(t_eval)):
    point = {
        'x': float(trajectory_x[i]),
        'y': float(trajectory_y[i]),
        'theta': float(theta[i]),
        't': float(t_eval[i]),
        'v': float(v_profile[i]),
        'w': float(omega_profile[i])
    }
    trajectory_dict.append(point)

# === СОХРАНЕНИЕ В ФАЙЛ ===
output_file = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/trajectory_output.txt'
with open(output_file, 'w') as f:
    f.write(str(trajectory_dict))

print(f"Траектория сохранена в файл: {output_file}")

plt.tight_layout()
plt.show()