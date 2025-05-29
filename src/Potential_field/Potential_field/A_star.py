import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import heapq
import math

# ——————— Функция построения дуги окружности ———————
def build_circular_arc(p_start, p_end, center, R_pixels, num_points=20):
    """
    Строит дугу окружности между двумя точками вокруг заданного центра
    :param p_start: начальная точка (x, y)
    :param p_end: конечная точка (x, y)
    :param center: центр окружности (x, y)
    :param R_pixels: радиус окружности (в пикселях)
    :param num_points: количество точек на дуге
    :return: список точек [(x, y), ...]
    """
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    center = np.array(center)

    angle_start = math.atan2(p_start[1] - center[1], p_start[0] - center[0])
    angle_end = math.atan2(p_end[1] - center[1], p_end[0] - center[0])

    # Определяем направление дуги
    delta_angle = angle_end - angle_start
    if abs(delta_angle) > math.pi:
        if delta_angle < 0:
            angle_end += 2 * math.pi
        else:
            angle_start += 2 * math.pi

    angles = np.linspace(angle_start, angle_end, num_points)
    arc_points = []

    for theta in angles:
        x = center[0] + R_pixels * math.cos(theta)
        y = center[1] + R_pixels * math.sin(theta)
        arc_points.append((int(x), int(y)))

    return arc_points


class AStarPlanner(Node):

    def __init__(self, map_name, start_x, start_y, goal_x, goal_y):
        super().__init__('a_star_planner_node')
        self.get_logger().info("A* Planner запущен")

        self.path_pub = self.create_publisher(Path, '/potential_path', 10)

        self.map_name = map_name
        self.start_x = int(start_x)
        self.start_y = int(start_y)
        self.goal_x = int(goal_x)
        self.goal_y = int(goal_y)

        self.get_logger().info(f'Загрузка карты из {self.map_name}')
        self.get_logger().info(f'Старт: ({self.start_x}, {self.start_y}), Цель: ({self.goal_x}, {self.goal_y})')

        self.load_map()
        self.plan_path()

    def load_map(self):
        try:
            with open(f"{self.map_name}.yaml", "r") as f:
                self.metadata = yaml.safe_load(f)

            img = Image.open(f"{self.map_name}.pgm").convert("L")
            self.width, self.height = img.size
            self.resolution = self.metadata['resolution']
            self.origin = self.metadata['origin']

            pixels = list(img.getdata())
            self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

            for y in range(self.height):
                for x in range(self.width):
                    idx = x + y * self.width
                    if pixels[idx] == 0:
                        self.grid[y][x] = 1  # стена
                    elif pixels[idx] > 0:
                        self.grid[y][x] = 0  # свободно
                    else:
                        self.grid[y][x] = 1  # неизвестно = стена

            self.get_logger().info(f'Карта загружена: {self.width}x{self.height}')

        except Exception as e:
            self.get_logger().error(f'Ошибка загрузки карты: {e}')
            raise

    def plan_path(self):
        start = (self.start_x, self.start_y)
        goal = (self.goal_x, self.goal_y)

        if not self.is_valid_point(start):
            self.get_logger().error("Стартовая точка вне карты или в стене")
            return

        if not self.is_valid_point(goal):
            self.get_logger().error("Целевая точка вне карты или в стене")
            return

        path = self.a_star(start, goal)

        if path:
            self.get_logger().info('Путь найден!')

            # Динамическое сглаживание с контролем отклонения
            R_min_start = 1.0  # Начальный минимальный радиус (в метрах)
            max_deviation_threshold = 0.5  # Максимальное отклонение (в метрах)
            smoothed_path = self.smooth_path_with_dynamic_radius(path, R_min_start, max_deviation_threshold)

            # Публикация и отображение
            self.publish_path(smoothed_path)
            self.plot_paths_before_after(path, smoothed_path)

        else:
            self.get_logger().error('Путь не найден!')

    def is_valid_point(self, point):
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] == 0

    def a_star(self, start, goal):
        def heuristic(a, b):
            dx = abs(b[0] - a[0])
            dy = abs(b[1] - a[1])
            return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)

        neighbors = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        move_cost = {
            (0, 1): 1,
            (1, 0): 1,
            (0, -1): 1,
            (-1, 0): 1,
            (1, 1): math.sqrt(2),
            (1, -1): math.sqrt(2),
            (-1, 1): math.sqrt(2),
            (-1, -1): math.sqrt(2)
        }

        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid_point(neighbor):
                    continue

                tentative_g_score = g_score[current] + move_cost[(dx, dy)]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # Путь не найден

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def angle_between(self, v1, v2):
        """Возвращает угол в градусах между двумя векторами"""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return float('inf')  # Вектор слишком маленький

        unit_v1 = v1 / norm_v1
        unit_v2 = v2 / norm_v2
        dot_product = np.dot(unit_v1, unit_v2)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        return np.degrees(np.arccos(dot_product))

    def find_strong_nonlinear_points(self, path, threshold_angle=150, step=3):
        """
        Находит индексы точек с резкими поворотами
        :param path: список координат [(x, y), ...]
        :param threshold_angle: порог угла (в градусах)
        :param step: шаг для анализа
        :return: список индексов
        """
        nonlinear_indices = []
        n = len(path)

        for i in range(step, n - step):
            p_prev = np.array(path[i - step])
            p_curr = np.array(path[i])
            p_next = np.array(path[i + step])

            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue

            angle = self.angle_between(v1, v2)

            if angle > threshold_angle:
                nonlinear_indices.append(i)
                self.get_logger().info(f'Резкий поворот в точке {i}, угол = {angle:.1f}°')

        return sorted(nonlinear_indices)

    def simplify_nonlinear_points(self, nonlinear_indices, min_gap=5):
        """
        Упрощает список точек поворотов, объединяя близкие
        :param nonlinear_indices: список индексов с резкими поворотами
        :param min_gap: минимальное расстояние между группами
        :return: упрощённый список
        """
        if not nonlinear_indices:
            return []

        simplified = []
        group = [nonlinear_indices[0]]

        for idx in nonlinear_indices[1:]:
            if abs(idx - group[-1]) <= min_gap:
                group.append(idx)
            else:
                simplified.append(int(np.mean(group)))
                group = [idx]

        simplified.append(int(np.mean(group)))
        return simplified

    def get_points_around_bend(self, path, bend_index, distance_meters=1.0):
        """
        Возвращает точки за 1 м до и после точки поворота
        :param path: список координат
        :param bend_index: индекс точки поворота
        :param distance_meters: расстояние от точки поворота для анализа
        :return: (точка_до, точка_после)
        """
        if bend_index >= len(path) or bend_index < 0:
            return None, None

        distance_pixels = distance_meters / self.resolution
        n = len(path)

        i_before = bend_index
        dist = 0.0
        while i_before > 0 and dist < distance_pixels:
            i_before -= 1
            dist += math.hypot(
                path[i_before + 1][0] - path[i_before][0],
                path[i_before + 1][1] - path[i_before][1]
            )

        i_after = bend_index
        dist = 0.0
        while i_after < n - 1 and dist < distance_pixels:
            i_after += 1
            dist += math.hypot(
                path[i_after][0] - path[i_after - 1][0],
                path[i_after][1] - path[i_after - 1][1]
            )

        if i_before < 0 or i_after >= len(path):
            return None, None

        return path[i_before], path[i_after]

    def smooth_path_with_min_radius(self, path, simplified_indices, R_min_meters=1.0):
        """
        Заменяет участки вокруг острых поворотов на дуги окружности
        :param path: оригинальный путь
        :param simplified_indices: точки с резкими поворотами
        :param R_min_meters: минимальный радиус (в метрах)
        :return: обновлённая траектория
        """
        smoothed_path = list(path)
        offset = 0
        R_min_pixels = R_min_meters / self.resolution

        for idx in sorted(simplified_indices):
            adjusted_idx = idx + offset

            before_point, after_point = self.get_points_around_bend(
                smoothed_path, adjusted_idx, distance_meters=1.0
            )

            if not before_point or not after_point:
                continue

            p_before = np.array(before_point)
            p_after = np.array(after_point)
            p_curr = np.array(smoothed_path[adjusted_idx])

            # Центр окружности — середина между before и after
            center = (p_before + p_after) / 2

            # Если радиус слишком мал, корректируем центр
            current_radius = np.linalg.norm(p_curr - center)
            if current_radius < R_min_pixels:
                direction = (p_curr - center) / (current_radius + 1e-8)
                center = p_curr - direction * R_min_pixels

            # Строим дугу окружности
            arc_points = build_circular_arc(p_before, p_after, center, R_min_pixels)

            # Удаляем старый участок и вставляем дугу
            try:
                i_start = smoothed_path.index(before_point)
                i_end = smoothed_path.index(after_point)
                if i_start >= i_end:
                    continue
                del smoothed_path[i_start:i_end+1]
                smoothed_path[i_start:i_start] = arc_points
                offset += len(arc_points) - (i_end - i_start + 1)
            except ValueError:
                self.get_logger().warn(f'Не удалось найти точки {before_point} или {after_point} в пути')

        return smoothed_path

    def calculate_max_deviation(self, original_path, smoothed_path):
        """
        Вычисляет максимальное отклонение между двумя траекториями (в метрах)
        """
        max_deviation = 0.0
        smoothed_array = np.array(smoothed_path)

        for x, y in original_path:
            point = np.array((x, y))
            distances = np.linalg.norm(smoothed_array - point, axis=1)
            if len(distances) > 0:
                min_distance = np.min(distances)
                max_deviation = max(max_deviation, min_distance)

        return max_deviation * self.resolution

    def smooth_path_with_dynamic_radius(self, path, initial_R_min=1.0, max_deviation_threshold=0.5, max_iterations=100):
        """
        Сглаживает путь, динамически уменьшая радиус, если отклонение слишком велико
        :param path: оригинальный путь
        :param initial_R_min: начальный радиус (в метрах)
        :param max_deviation_threshold: порог отклонения (в метрах)
        :param max_iterations: лимит итераций
        :return: сглаженная траектория
        """
        current_path = list(path)
        R_min = initial_R_min
        iteration = 0

        while iteration < max_iterations:
            # Найти точки с резкими поворотами
            nonlinear_indices = self.find_strong_nonlinear_points(current_path, threshold_angle=20, step=3)
            simplified_indices = self.simplify_nonlinear_points(nonlinear_indices, min_gap=5)

            if not simplified_indices:
                self.get_logger().info(f'Сглаживание завершено за {iteration} итераций')
                break

            self.get_logger().info(f'Итерация {iteration + 1}: найдено {len(simplified_indices)} точек перегиба')

            # Применить сглаживание с текущим радиусом
            new_path = self.smooth_path_with_min_radius(current_path, simplified_indices)

            # Проверить отклонение
            max_deviation = self.calculate_max_deviation(path, new_path)
            self.get_logger().info(f'Максимальное отклонение: {max_deviation:.2f} м')

            # Если отклонение допустимо — выйти
            if max_deviation <= max_deviation_threshold:
                self.get_logger().info('Отклонение в пределах порога. Сглаживание завершено.')
                return new_path

            # Иначе уменьшить радиус и попробовать снова
            R_min = max(R_min - 0.1, 0.2)
            current_path = new_path
            iteration += 1

        self.get_logger().warn(f'Достигнут лимит итераций ({max_iterations}), сглаживание не завершено')
        return current_path

    def publish_path(self, path):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for x, y in path:
            pose = PoseStamped()
            pose.header = Header()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x * self.resolution + self.origin[0]
            pose.pose.position.y = y * self.resolution + self.origin[1]
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.get_logger().info('Путь опубликован на /potential_path')

    def plot_paths_before_after(self, original_path, smoothed_path):
        plt.figure(figsize=(12, 8))
        orig = np.array(original_path)
        smooth = np.array(smoothed_path)

        plt.plot(orig[:, 0], orig[:, 1], 'r--', label='Оригинал')
        plt.plot(smooth[:, 0], smooth[:, 1], 'g-', linewidth=2, label='Сглаженная')
        plt.plot(self.start_x, self.start_y, 'go', markersize=10, label='Начало')
        plt.plot(self.goal_x, self.goal_y, 'ro', markersize=10, label='Цель')

        plt.legend()
        plt.title('Путь до и после сглаживания (минимальный радиус = 1 м)')
        plt.axis('equal')
        plt.grid(True)
        plt.show()


def main(args=None):
    rclpy.init(args=args)

    map_name = 'new_map'     # имя карты без расширения
    start_x = 20             # стартовая X-координата (в пикселях)
    start_y = 10             # стартовая Y-координата
    goal_x = 300             # целевая X-координата
    goal_y = 150             # целевая Y-координата

    node = AStarPlanner(map_name, start_x, start_y, goal_x, goal_y)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()