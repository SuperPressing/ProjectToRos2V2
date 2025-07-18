import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Float32MultiArray
import numpy as np
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import heapq
import math
import cv2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def build_circular_arc(p_start, p_end, center, R_pixels, num_points=20):

    p_start = np.array(p_start)
    p_end = np.array(p_end)
    center = np.array(center)

    angle_start = math.atan2(p_start[1] - center[1], p_start[0] - center[0])
    angle_end = math.atan2(p_end[1] - center[1], p_end[0] - center[0])

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

    def __init__(self, map_name, start_x, start_y, goal_x, goal_y, upscale_factor=10):
        super().__init__('a_star_planner_node')
        self.map_name = map_name
        self.get_logger().info("A* Planner запущен")
        self.upscale_factor = upscale_factor  
        self.map_name = map_name
        self.start_x = int(start_x)
        self.start_y = int(start_y)
        self.goal_x = int(goal_x)
        self.goal_y = int(goal_y)
        self.load_map()
        self.path_pub = self.create_publisher(Path, '/potential_path', 10)

        self.array_pub = self.create_publisher(Float32MultiArray, '/path_coordinates', 10)
        self.start_sub = self.create_subscription(Path, '/start_pose', self.start_callback, 10)
        
        self.get_logger().info(f'Загрузка карты из {self.map_name}')
        self.get_logger().info(f'Старт: ({self.start_x}, {self.start_y}), Цель: ({self.goal_x}, {self.goal_y})')
        self.plan_path()

 

    def start_callback(self, msg):
        self.get_logger().info('Получен старт')
        self.start_received = True
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
            grid = np.zeros((self.height, self.width), dtype=np.uint8)
            for y in range(self.height):
                for x in range(self.width):
                    idx = x + y * self.width
                    if pixels[idx] == 0:
                        grid[y][x] = 1 
                    elif pixels[idx] > 0:
                        grid[y][x] = 0  
                    else:
                        grid[y][x] = 1  

            self.grid = self.expand_obstacles(grid, distance_meters=0.2)

            self.get_logger().info(f"Исходный размер: {self.width}x{self.height}")
            grid = self.upscale_map(grid)
            self.get_logger().info(f"Новый размер: {self.width}x{self.height}, разрешение: {self.resolution} м/пиксель")

            self.start_x = int(self.start_x * self.upscale_factor)
            self.start_y = int(self.start_y * self.upscale_factor)
            self.goal_x = int(self.goal_x * self.upscale_factor)
            self.goal_y = int(self.goal_y * self.upscale_factor)
            self.get_logger().info(f"Новые координаты: Старт({self.start_x}, {self.start_y}), Цель({self.goal_x}, {self.goal_y})")

            self.grid = self.expand_obstacles(grid, distance_meters=0.2)
            self.get_logger().info(f'Карта загружена и расширена: {self.width}x{self.height}')
            
        except Exception as e:
            self.get_logger().error(f'Ошибка загрузки карты: {e}')
            raise

    def upscale_map(self, grid):
        """Увеличивает разрешение карты с интерполяцией"""
        height, width = grid.shape
        
        new_width = width * self.upscale_factor
        new_height = height * self.upscale_factor
        upscaled = cv2.resize(
            grid.astype(np.float32), 
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        self.width = new_width
        self.height = new_height
        self.resolution /= self.upscale_factor
        
        binary_upscaled = np.zeros_like(upscaled, dtype=np.uint8)
        binary_upscaled[upscaled > 0.5] = 1 
        
        return binary_upscaled
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

            R_min_start = 0.2 
            max_deviation_threshold = 0.5  
            smoothed_path = path
            output_file = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/trac.txt'
            with open(output_file, 'w') as f:
                f.write(str(smoothed_path))
            print(smoothed_path)
            # Публикация и отображение
            msg = Path()
            msg.header.stamp = self.get_clock().now().to_msg() 
            msg.header.frame_id = "map" 
            for x, y in smoothed_path:
                pose = PoseStamped()
                pose.header.frame_id = "map"

                pose.pose.position.x = self.origin[0]
                pose.pose.position.y = self.origin[1]
                pose.pose.orientation.w = 1.0
                msg.poses.append(pose)
            self.publish_path(smoothed_path)
            self.publish_path_array(smoothed_path)
            self.plot_paths_before_after(path, smoothed_path)

        else:
            self.get_logger().error('Путь не найден!')

    def is_valid_point(self, point):
        x, y = point
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        if self.grid[y][x] != 0:
            return False

        check_radius = 1  # пикселей
        for dy in range(-check_radius, check_radius + 1):
            for dx in range(-check_radius, check_radius + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx] != 0:
                        return False
        return True

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

        return []  

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

            
            center = (p_before + p_after) / 2

            current_radius = np.linalg.norm(p_curr - center)
            if current_radius < R_min_pixels:
                direction = (p_curr - center) / (current_radius + 1e-8)
                center = p_curr - direction * R_min_pixels

           
            arc_points = build_circular_arc(p_before, p_after, center, R_min_pixels)

            
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
       
        current_path = list(path)
        R_min = initial_R_min
        iteration = 0

       
        while iteration < max_iterations:
            nonlinear_indices = self.find_strong_nonlinear_points(current_path, threshold_angle=20, step=3)
            simplified_indices = self.simplify_nonlinear_points(nonlinear_indices, min_gap=5)

            if not simplified_indices:
                self.get_logger().info(f'Сглаживание завершено за {iteration} итераций')
                break

            self.get_logger().info(f'Итерация {iteration + 1}: найдено {len(simplified_indices)} точек перегиба')

            new_path = self.smooth_path_with_min_radius(current_path, simplified_indices, R_min)

            max_deviation = self.calculate_max_deviation(path, new_path)
            self.get_logger().info(f'Максимальное отклонение: {max_deviation:.2f} м')

            if max_deviation <= max_deviation_threshold:
                self.get_logger().info('Отклонение в пределах порога. Сглаживание завершено.')
                return new_path

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
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.get_logger().info('Путь опубликован на /potential_path')

    # Новая функция для публикации массива координат
    def publish_path_array(self, path):
        # Создаем массив: [x0, y0, x1, y1, ...]
        data = []
        for point in path:
            # Преобразуем в мировые координаты (метры)
            x_world = point[0] * self.resolution + self.origin[0]
            y_world = point[1] * self.resolution + self.origin[1]
            data.append(float(x_world))
            data.append(float(y_world))

        msg = Float32MultiArray()
        msg.data = data
        self.array_pub.publish(msg)
        self.get_logger().info(f'Опубликован массив координат пути, всего {len(data)} чисел')

    def expand_obstacles(self, grid, distance_meters=0.2):

        height, width = grid.shape
        expanded_grid = grid.copy()
        distance_pixels = int(distance_meters / self.resolution)

       
        for y in range(height):
            for x in range(width):
                if grid[y][x] == 1:  
                    for dy in range(-distance_pixels, distance_pixels + 1):
                        for dx in range(-distance_pixels, distance_pixels + 1):
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                expanded_grid[ny][nx] = 1 
        return expanded_grid
    
    def plot_paths_before_after(self, original_path, smoothed_path):
        try:
            img = Image.open(f"{self.map_name}.pgm").convert("L")
            map_img = np.array(img)
            
            plt.figure(figsize=(12, 8))
            
            
            plt.imshow(map_img, cmap='gray', origin='lower', 
                      extent=[0, self.width/self.upscale_factor, 
                              0, self.height/self.upscale_factor])
            
            orig = np.array(original_path) / self.upscale_factor
            smooth = np.array(smoothed_path) / self.upscale_factor
            
            
            plt.plot(orig[:, 0], orig[:, 1], 'r--', label='Original Path')
            plt.plot(smooth[:, 0], smooth[:, 1], 'g-', linewidth=2, label='Smoothed Path')
            
            
            plt.plot(self.start_x/self.upscale_factor, 
                     self.start_y/self.upscale_factor, 
                     'go', markersize=10, label='Start')
            plt.plot(self.goal_x/self.upscale_factor, 
                     self.goal_y/self.upscale_factor, 
                     'ro', markersize=10, label='Goal')
            
            plt.legend()
            plt.title('Path Before and After Smoothing Overlaid on Map')
            plt.xlabel('X (pixels)')
            plt.ylabel('Y (pixels)')
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            self.get_logger().error(f'Failed to plot map: {e}')
            
            plt.figure(figsize=(12, 8))
            orig = np.array(original_path)
            smooth = np.array(smoothed_path)

            plt.plot(orig[:, 0], orig[:, 1], 'r--', label='Original')
            plt.plot(smooth[:, 0], smooth[:, 1], 'g-', linewidth=2, label='Smoothed')
            plt.plot(self.start_x, self.start_y, 'go', markersize=10, label='Start')
            plt.plot(self.goal_x, self.goal_y, 'ro', markersize=10, label='Goal')

            plt.legend()
            plt.title('Path Before and After Smoothing (Min Radius = 1 m)')
            plt.axis('equal')
            plt.grid(True)
            plt.show()


def main(args=None):
    rclpy.init(args=args)

    map_name = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/new_map'
    start_x = 250
    start_y = 150
    goal_x = 50
    goal_y = 100
    upscale_factor = 1 

    node = AStarPlanner(map_name, start_x, start_y, goal_x, goal_y, upscale_factor)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()