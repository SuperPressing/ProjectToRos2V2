#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import heapq
import matplotlib.pyplot as plt
import yaml
import os

class PathReplanner(Node):
    def __init__(self):
        super().__init__('path_replanner')
        
        # Параметры алгоритма
        self.step_back = 10      # Отступ назад от точки пересечения (пиксели)
        self.step_forward = 10   # Отступ вперед от точки пересечения (пиксели)
        self.safety_margin = 5   # Безопасное расстояние от препятствий (пиксели)
        self.enable_visualization = True
        
        # Текущие данные
        self.original_path = None
        self.obstacle_points = []
        self.grid_data = None
        self.grid_width = 0
        self.grid_height = 0
        
        # Подписки
        self.create_subscription(
            Path,
            '/potential_path',
            self.path_callback,
            10
        )
        
        self.create_subscription(
            Path,
            '/obstacle_points',
            self.obstacle_callback,
            10
        )
        
        # Публикатор нового пути
        self.path_pub = self.create_publisher(Path, '/potential_path_test', 10)
        
        # Загрузка карты
        self.load_map_from_file('/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/update_map.pgm')
        
        # Визуализация
        if self.enable_visualization:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.title("Path Replanning Visualization")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.grid(True)

    def load_map_from_file(self, pgm_file):
        """Загрузка карты из PGM файла"""
        try:
            base_name = os.path.splitext(pgm_file)[0]
            yaml_file = base_name + '.yaml'
            
            with open(yaml_file, 'r') as f:
                map_metadata = yaml.safe_load(f)
            
            with open(pgm_file, 'rb') as f:
                # Чтение заголовка PGM
                header = f.readline().decode().strip()
                if header != 'P5':
                    raise ValueError("Unsupported PGM format")
                
                # Пропуск комментариев
                line = f.readline().decode().strip()
                while line.startswith('#'):
                    line = f.readline().decode().strip()
                
                # Получение размеров
                self.grid_width, self.grid_height = map(int, line.split())
                max_val = int(f.readline().decode().strip())
                
                # Чтение данных
                img_data = f.read()
                if len(img_data) != self.grid_width * self.grid_height:
                    raise ValueError("Map data size mismatch")
                
                # Преобразование в occupancy grid (0-100)
                self.grid_data = [100 - int(pixel/255.0*100) for pixel in img_data]
                
            self.get_logger().info(f"Карта загружена: {self.grid_width}x{self.grid_height} px")
            
        except Exception as e:
            self.get_logger().error(f"Ошибка загрузки карты: {str(e)}")
            raise

    def path_callback(self, msg):
        """Обработка получения исходного пути"""
        if not msg.poses:
            self.get_logger().warn("Получен пустой путь")
            return
            
        self.original_path = msg
        self.get_logger().info(f"Получен новый путь из {len(msg.poses)} точек")
        
        # Если уже есть препятствия - выполняем перепланировку
        if self.obstacle_points:
            self.replan_and_publish()

    def obstacle_callback(self, msg):
        """Обработка новых точек препятствий"""
        if not msg.poses:
            self.get_logger().warn("Получены пустые препятствия")
            return
            
        # Добавляем новые точки препятствий
        for pose in msg.poses:
            point = int(pose.pose.position.x), int(pose.pose.position.y)
            if point not in self.obstacle_points:
                self.obstacle_points.append(point)
        
        self.get_logger().info(f"Получено {len(msg.poses)} новых препятствий. Всего: {len(self.obstacle_points)}")
        
        # Если есть путь - выполняем перепланировку
        if self.original_path:
            self.replan_and_publish()

    def is_cell_free(self, x, y):
        """Проверка, свободна ли ячейка карты"""
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.grid_data[y * self.grid_width + x] <= 50
        return False

    def find_closest_point_index(self, obstacle_point):
        """Находит ближайшую точку пути к препятствию"""
        min_dist = float('inf')
        closest_idx = 0
        ox, oy = obstacle_point
        
        for i, pose in enumerate(self.original_path.poses):
            px = int(pose.pose.position.x)
            py = int(pose.pose.position.y)
            dist = (px - ox)**2 + (py - oy)**2
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx

    def a_star_search(self, start, goal):
        """Алгоритм A* для поиска пути"""
        if not self.is_cell_free(*start) or not self.is_cell_free(*goal):
            return None

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        directions = [(1,0), (-1,0), (0,1), (0,-1), 
                     (1,1), (1,-1), (-1,1), (-1,-1)]
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
                
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_cell_free(*neighbor):
                    continue
                    
                tentative_g = g_score[current] + np.hypot(dx, dy)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return None

    def replan_and_publish(self):
        """Основная функция перепланировки"""
        if not self.original_path or not self.obstacle_points:
            return
            
        # Создаем копию оригинального пути
        new_path = Path()
        new_path.header = self.original_path.header
        new_path.header.stamp = self.get_clock().now().to_msg()
        new_path.poses = list(self.original_path.poses)
        
        # Обрабатываем каждое препятствие
        for obstacle in self.obstacle_points:
            inter_idx = self.find_closest_point_index(obstacle)
            
            # Определяем сегмент для перепланировки
            start_idx = max(0, inter_idx - self.step_back)
            end_idx = min(len(new_path.poses) - 1, inter_idx + self.step_forward)
            
            # Начальная и конечная точки
            start_point = (int(new_path.poses[start_idx].pose.position.x), 
                          int(new_path.poses[start_idx].pose.position.y))
            end_point = (int(new_path.poses[end_idx].pose.position.x), 
                         int(new_path.poses[end_idx].pose.position.y))
            
            # Поиск нового пути
            path_pixels = self.a_star_search(start_point, end_point)
            if path_pixels:
                # Создаем новый сегмент
                new_segment = []
                for px, py in path_pixels:
                    pose = PoseStamped()
                    pose.header = new_path.header
                    pose.pose.position.x = float(px)
                    pose.pose.position.y = float(py)
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    new_segment.append(pose)
                
                # Заменяем сегмент в пути
                new_path.poses = new_path.poses[:start_idx] + new_segment + new_path.poses[end_idx+1:]
        
        # Публикация нового пути
        self.path_pub.publish(new_path)
        self.save_path_to_file(new_path, '/home/neo/Documents/replanned_path.txt')
        self.get_logger().info("Опубликован перепланированный путь")
        
        # Визуализация
        if self.enable_visualization:
            self.visualize_path(new_path)

    def visualize_path(self, path):
        """Визуализация пути и препятствий с корректной ориентацией карты"""
        try:
            self.ax.clear()
            
            # Отображение карты (переворачиваем по оси Y)
            if self.grid_data:
                grid = np.array(self.grid_data).reshape(self.grid_height, self.grid_width)
                occupied = np.where(grid > 50)
                # Инвертируем Y-координаты
                self.ax.scatter(occupied[1], self.grid_height - occupied[0] - 1, 
                            c='gray', s=1, alpha=0.5, label='Препятствия')
            
            # Оригинальный путь (переворачиваем Y)
            if self.original_path.poses:
                orig_x = [p.pose.position.x for p in self.original_path.poses]
                orig_y = [p.pose.position.y for p in self.original_path.poses]
                self.ax.plot(orig_x, orig_y, 'b-', linewidth=2, label='Исходный путь')
            
            # Новый путь (переворачиваем Y)
            if path.poses:
                new_x = [p.pose.position.x for p in path.poses]
                new_y = [p.pose.position.y for p in path.poses]
                self.ax.plot(new_x, new_y, 'g--', linewidth=3, label='Новый путь')
            
            # Препятствия (переворачиваем Y)
            for ox, oy in self.obstacle_points:
                self.ax.scatter(ox, oy, 
                            c='red', marker='x', s=100, label='Препятствия')
            
            # Настройки графика
            self.ax.set_title("Перепланировка пути (корректная ориентация)")
            self.ax.set_xlim(0, self.grid_width)
            self.ax.set_ylim(0, self.grid_height)
            
            # Убираем дублирующиеся легенды
            handles, labels = self.ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys())
            
            self.ax.grid(True)
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            self.get_logger().error(f"Ошибка визуализации: {str(e)}")

    def save_path_to_file(self, path_msg, filename='replanned_path.txt'):
        """Сохраняет координаты точек из Path в текстовый файл"""
        try:
            with open(filename, 'w') as f:
                for i, pose_stamped in enumerate(path_msg.poses):
                    position = pose_stamped.pose.position
                    line = f"Point {i}: x={position.x}, y={position.y}, z={position.z}\n"
                    f.write(line)
            self.get_logger().info(f"Путь сохранён в файл: {filename}")
        except Exception as e:
            self.get_logger().error(f"Ошибка при сохранении файла: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = PathReplanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.enable_visualization:
            plt.ioff()
            plt.show(block=True)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()