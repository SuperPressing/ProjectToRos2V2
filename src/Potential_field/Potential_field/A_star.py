import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import heapq  # для приоритетной очереди в A*
import math


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

            # Найти точки с сильной нелинейностью
            nonlinear_indices = self.find_strong_nonlinear_points(path, threshold_angle=150, step=3)

            # Упростить список: оставить только ключевые повороты
            simplified_indices = self.simplify_nonlinear_points(nonlinear_indices, min_gap=5)

            # Для каждой точки поворота найти точки за 1 м до и после
            for idx in simplified_indices:
                before_point, after_point = self.get_points_around_bend(path, idx, distance_meters=1.0)
                if before_point and after_point:
                    self.get_logger().info(
                        f'За 1 метр: {before_point}, Поворот: {path[idx]}, После 1 метра: {after_point}'
                    )

            # Отобразить траекторию с отметкой упрощённых поворотов
            self.publish_path(path)
            self.plot_path(path, simplified_indices)
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
            (0, 1), (1, 0), (0, -1), (-1, 0),  # основные направления
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # диагонали
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
            return float('inf')  # Вектор слишком маленький — не определён угол

        unit_v1 = v1 / norm_v1
        unit_v2 = v2 / norm_v2

        dot_product = np.dot(unit_v1, unit_v2)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        return np.degrees(np.arccos(dot_product))

    def find_strong_nonlinear_points(self, path, threshold_angle=150, step=3):
        """
        Находит точки с резким изменением направления
        :param path: список координат [(x, y), ...]
        :param threshold_angle: порог угла (в градусах)
        :param step: шаг для анализа
        :return: список индексов с резкими поворотами
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

            if angle > 180-threshold_angle:
                nonlinear_indices.append(i)
                self.get_logger().info(f'Резкий поворот в точке {i}, угол = {angle:.1f}°')

        return sorted(nonlinear_indices)

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

    def plot_path(self, path, nonlinear_indices=None):
        plt.figure(figsize=(10, 8))
        grid_vis = np.copy(self.grid)

        plt.imshow(grid_vis, cmap='gray', origin='lower', alpha=0.7)

        path_np = np.array(path)
        plt.plot(path_np[:, 0], path_np[:, 1], 'r-', linewidth=2, label='A* путь')

        if nonlinear_indices:
            bad_points = np.array([path[i] for i in nonlinear_indices])
            plt.scatter(bad_points[:, 0], bad_points[:, 1], color='orange', s=60, label='Резкие повороты')

        plt.plot(self.start_x, self.start_y, 'go', markersize=10, label='Начало')
        plt.plot(self.goal_x, self.goal_y, 'ro', markersize=10, label='Цель')
        plt.legend()
        plt.title('A* планировщик пути')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def simplify_nonlinear_points(self, nonlinear_indices, min_gap=5):
        """
        Упрощает список индексов, объединяя рядом стоящие точки в группы
        и возвращая только средний индекс для каждой группы.

        :param nonlinear_indices: список индексов с резкими поворотами
        :param min_gap: минимальное расстояние между группами (по индексу)
        :return: упрощённый список индексов
        """
        if not nonlinear_indices:
            return []

        simplified = []
        group = [nonlinear_indices[0]]

        for idx in nonlinear_indices[1:]:
            if abs(idx - group[-1]) <= min_gap:
                group.append(idx)
            else:
                # Сохраняем средний индекс группы
                simplified.append(int(np.mean(group)))
                group = [idx]

        # Добавляем последнюю группу
        simplified.append(int(np.mean(group)))

        self.get_logger().info(f'Сокращено с {len(nonlinear_indices)} до {len(simplified)} точек')
        return simplified

    def get_points_around_bend(self, path, bend_index, distance_meters=1.0):
        """
        Возвращает две точки: за 1 метр до и через 1 метр после точки поворота
        :param path: список координат [(x, y), ...]
        :param bend_index: индекс точки поворота
        :param distance_meters: расстояние в метрах (например, 1 м)
        :return: кортеж (точка_до, точка_после)
        """

        # Переводим расстояние в пиксели
        distance_pixels = distance_meters / self.resolution

        n = len(path)

        # Ищем точку до поворота (~1 метр назад)
        i_before = bend_index
        dist = 0.0
        while i_before > 0 and dist < distance_pixels:
            i_before -= 1
            dist += math.hypot(
                path[i_before + 1][0] - path[i_before][0],
                path[i_before + 1][1] - path[i_before][1]
            )

        # Ищем точку после поворота (~1 метр вперёд)
        i_after = bend_index
        dist = 0.0
        while i_after < n - 1 and dist < distance_pixels:
            i_after += 1
            dist += math.hypot(
                path[i_after][0] - path[i_after - 1][0],
                path[i_after][1] - path[i_after - 1][1]
            )

        point_before = path[i_before] if i_before >= 0 else None
        point_after = path[i_after] if i_after < len(path) else None

        return point_before, point_after

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