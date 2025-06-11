#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PointStamped, Point
import numpy as np
import heapq
import matplotlib.pyplot as plt
import yaml
import os

class PathReplanner(Node):
    def __init__(self):
        super().__init__('path_replanner')
        
        # Параметры
        self.step_back = 10    # Отступ назад от точки пересечения (в пикселях)
        self.step_forward = 15 # Отступ вперед от точки пересечения (в пикселях)
        self.safety_margin = 5 # Безопасное расстояние от препятствий (в пикселях)
        self.enable_visualization = True
        
        # Подписки
        self.sub_path = self.create_subscription(
            Path,
            '/potential_path',
            self.path_callback,
            10
        )
        self.sub_obstacle = self.create_subscription(
            PointStamped,
            '/obstacle_points',
            self.obstacle_callback,
            10
        )
        
        # Публикатор
        self.pub_path = self.create_publisher(Path, '/replanned_path', 10)
        
        # Переменные для данных
        self.current_path = None
        self.obstacle_point = None  # Теперь в координатах карты (x, y)
        self.occupancy_grid = None
        self.grid_data = None
        self.grid_width = 0
        self.grid_height = 0

        # Загружаем карту
        self.load_map_from_file('/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/update_map.pgm')
        
        # Инициализация графики
        if self.enable_visualization:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.title("Path Visualization (Map Coordinates)")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.grid(True)

    def load_map_from_file(self, pgm_file):
        """Загружает карту из PGM файла"""
        try:
            base_name = os.path.splitext(pgm_file)[0]
            yaml_file = base_name + '.yaml'
            
            with open(yaml_file, 'r') as f:
                map_metadata = yaml.safe_load(f)
            
            with open(pgm_file, 'rb') as f:
                # Чтение заголовков PGM
                header = f.readline().decode().strip()
                if header != 'P5':
                    raise ValueError("Unsupported PGM format")
                
                # Пропуск комментариев
                line = f.readline().decode().strip()
                while line.startswith('#'):
                    line = f.readline().decode().strip()
                
                # Размеры карты
                self.grid_width, self.grid_height = map(int, line.split())
                max_val = int(f.readline().decode().strip())
                
                # Чтение данных
                img_data = f.read()
                if len(img_data) != self.grid_width * self.grid_height:
                    raise ValueError("Map data size mismatch")
                
                # Преобразование в occupancy data (0-100)
                self.grid_data = [100 - int(pixel/255.0*100) for pixel in img_data]
                
            self.get_logger().info(f"Карта загружена: {self.grid_width}x{self.grid_height} px")
            
        except Exception as e:
            self.get_logger().error(f"Ошибка загрузки карты: {str(e)}")
            raise

    def obstacle_callback(self, msg):
        """Обработка точки препятствия (уже в координатах карты)"""
        # Предполагаем, что точка уже в координатах карты
        self.obstacle_point = (int(msg.point.x), int(msg.point.y))
        self.get_logger().info(f"Получена точка препятствия: {self.obstacle_point}")

    def path_callback(self, msg):
        if self.obstacle_point is None or self.grid_data is None:
            if self.obstacle_point is None:
                self.get_logger().info("Не найдена точка пересечения")
            elif self.grid_data is None:
                self.get_logger().info("Преобразование в occupancy data")
            self.get_logger().info("Ожидание данных...")
            return
            
        self.current_path = msg
        original_length = len(self.current_path.poses)
        new_path = self.replan_path()
        
        # Визуализация только если путь изменился
        if new_path is not None and len(new_path.poses) != original_length:
            self.get_logger().info(f"Путь изменен: было {original_length}, стало {len(new_path.poses)} точек")
            if self.enable_visualization:
                self.visualize_path(new_path)
        else:
            self.get_logger().info("Перепланировка не требуется")

    def is_cell_free(self, x, y):
        """Проверка, свободна ли ячейка карты"""
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.grid_data[y * self.grid_width + x] <= 50  # <=50 - свободно
        return False

    def a_star_search(self, start, goal):
        """Поиск пути A* в координатах карты"""
        if not self.is_cell_free(*start) or not self.is_cell_free(*goal):
            return None

        # Эвристика (манхэттенское расстояние)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # 8-связная область
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
                # Восстановление пути
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

    def find_closest_point_index(self):
        """Находит ближайшую точку пути к препятствию"""
        if not self.current_path.poses:
            return None
            
        min_dist = float('inf')
        closest_idx = 0
        ox, oy = self.obstacle_point
        
        for i, pose in enumerate(self.current_path.poses):
            px, py = int(pose.pose.position.x), int(pose.pose.position.y)
            dist = (px - ox)**2 + (py - oy)**2
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx

    def replan_path(self):
        """Перепланировка пути в координатах карты"""
        if not self.current_path.poses:
            return None
            
        inter_idx = self.find_closest_point_index()
        if inter_idx is None:
            return self.current_path
            
        poses = self.current_path.poses
        
        # Определяем сегмент для перепланировки
        start_idx = max(0, inter_idx - self.step_back)
        end_idx = min(len(poses) - 1, inter_idx + self.step_forward)
        
        # Начальная и конечная точки (уже в координатах карты)
        start_point = (int(poses[start_idx].pose.position.x), 
                       int(poses[start_idx].pose.position.y))
        end_point = (int(poses[end_idx].pose.position.x), 
                     int(poses[end_idx].pose.position.y))
        
        # Поиск нового пути
        path_pixels = self.a_star_search(start_point, end_point)
        if not path_pixels:
            return self.current_path
            
        # Создаем новый путь
        new_path = Path()
        new_path.header = self.current_path.header
        new_path.header.stamp = self.get_clock().now().to_msg()
        
        # Часть до точки отступления
        new_path.poses = poses[:start_idx]
        
        # Добавляем новый сегмент
        for px, py in path_pixels:
            pose = PoseStamped()
            pose.header = new_path.header
            pose.pose.position.x = float(px)
            pose.pose.position.y = float(py)
            pose.pose.position.z = 0.0
            pose.pose.orientation = poses[start_idx].pose.orientation
            new_path.poses.append(pose)
        
        # Часть после нового сегмента
        new_path.poses += poses[end_idx:]
        
        self.pub_path.publish(new_path)
        self.obstacle_point = None  # Сброс после обработки
        
        return new_path

    def visualize_path(self, new_path):
        """Визуализация в координатах карты"""
        try:
            if not plt.fignum_exists(self.fig.number):
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
                plt.ion()
            
            self.ax.clear()
            
            # Отображение карты
            if self.grid_data:
                # Преобразуем в 2D массив для визуализации
                grid = np.array(self.grid_data).reshape(self.grid_height, self.grid_width)
                occupied = np.where(grid > 50)
                self.ax.scatter(
                    occupied[1],  # X - столбцы
                    occupied[0],  # Y - строки
                    c='gray', s=1, alpha=0.5, label='Obstacles'
                )
            
            # Оригинальный путь
            if self.current_path.poses:
                orig_x = [p.pose.position.x for p in self.current_path.poses]
                orig_y = [p.pose.position.y for p in self.current_path.poses]
                self.ax.plot(orig_x, orig_y, 'b-', linewidth=2, label='Original Path')
            
            # Новый путь
            if new_path.poses:
                new_x = [p.pose.position.x for p in new_path.poses]
                new_y = [p.pose.position.y for p in new_path.poses]
                self.ax.plot(new_x, new_y, 'g--', linewidth=3, label='Replanned Path')
            
            # Точка препятствия
            if self.obstacle_point:
                self.ax.scatter(
                    self.obstacle_point[0], self.obstacle_point[1],
                    c='red', marker='x', s=100, label='Obstacle'
                )
            
            # Настройки графика
            self.ax.set_title("Path in Map Coordinates")
            self.ax.set_xlim(0, self.grid_width)
            self.ax.set_ylim(0, self.grid_height)
            self.ax.legend()
            self.ax.grid(True)
            
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            self.get_logger().error(f"Visualization error: {str(e)}")

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