import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import numpy as np
import math
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import ast
import subprocess
import json
class MDPReplanner(Node):
    def __init__(self):
        super().__init__('mdp_replanner_node')
        self.get_logger().info("MDP Replanner запущен")

        # Подписки
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)

        # Исходная и сглаженная траектории
        self.original_path = []
        self.current_position = (0, 0)
        self.dynamic_grid = None
        self.grid = None
        self.resolution = 0.05
        self.origin = (0.0, 0.0)
        self.width = 0
        self.height = 0
        self.markov = False
        # Путь к файлу для сохранения
        self.output_file = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/trac.txt'

        # Загрузка карты и пути
        self.load_map()
        self.load_path_from_file()

    def load_path_from_file(self):
        """Загружает путь из файла trac.txt"""
        file_path = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/trac.txt'
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            if not content:
                raise ValueError("Файл trac.txt пуст")
            self.original_path = ast.literal_eval(content)
            self.get_logger().info(f'Загружен путь с {len(self.original_path)} точками')
            self.plot_paths()
        except FileNotFoundError:
            self.get_logger().error(f'Файл {file_path} не найден')
        except (SyntaxError, ValueError) as e:
            self.get_logger().error(f'Ошибка парсинга файла trac.txt: {e}')

    def path_callback(self, msg):
        self.original_path = []
        for pose in msg.poses:
            x = int((pose.pose.position.x - self.origin[0]) / self.resolution)
            y = int((pose.pose.position.y - self.origin[1]) / self.resolution)
            self.original_path.append((x, y))
        self.get_logger().info(f'Получена траектория с {len(self.original_path)} точками')
        self.plot_paths()

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # self.current_x = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        # self.current_y = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        self.current_x = x*20.9+250
        self.current_y = y*20.9+150
        self.current_position = (x, y)

    def lidar_callback(self, msg):
        if not self.original_path:
            return
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_inc = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max
        robot_x, robot_y = self.current_position
        angles = np.linspace(angle_min, angle_max, len(ranges))

        for i, r in enumerate(ranges):
            if r < range_max and r > range_min:
                obs_x = robot_x + int(r * np.cos(angles[i]) / self.resolution)
                obs_y = robot_y + int(r * np.sin(angles[i]) / self.resolution)
                if 0 <= obs_x < self.width and 0 <= obs_y < self.height:
                    self.dynamic_grid[obs_y][obs_x] = 1  # помечаем как препятствие
        cloc = 0
        # Проверяем, сталкивается ли робот с препятствием
        for point in self.original_path:
            x, y = point
            if self.dynamic_grid[y][x] == 1:
                cloc+=1
            if cloc>20:
                self.get_logger().warn(f'Препятствие на траектории в точке ({x}, {y})')
                self.original_path = self.mdp_replan(self.original_path, point)
                self.plot_paths()
                self.save_path_to_file(self.original_path)  # ✅ Сохраняем путь в файл
                with open('/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/Flag.json', 'w') as f:
                    json.dump('True', f)
                break
            else:
                with open('Flag.json', 'w') as f:
                    json.dump('False', f)

    def load_map(self):
        """Загрузка карты (пример, можно заменить на /map callback)"""
        map_name = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/new_map' 
        try:
            with open(f'{map_name}.yaml', 'r') as f:
                self.metadata = yaml.safe_load(f)
            img = Image.open(f'{map_name}.pgm').convert('L')
            self.width, self.height = img.size
            self.resolution = self.metadata['resolution']
            self.origin = tuple(self.metadata['origin'])
            pixels = list(img.getdata())
            self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
            for y in range(self.height):
                for x in range(self.width):
                    idx = x + y * self.width
                    self.grid[y][x] = 0 if pixels[idx] > 0 else 1
            self.dynamic_grid = np.copy(self.grid)
        except Exception as e:
            self.get_logger().error(f'Ошибка загрузки карты: {e}')
            raise


    def mdp_replan(self, path, obstacle_point):
        self.get_logger().info('Запуск MDP перепланирования...')
        start = self.current_position
        goal = path[-1]
        states = self.build_states(path, obstacle_point)
        transitions = self.build_transitions(states)
        rewards = self.build_rewards(states, goal, path)
        policy = self.solve_mdp(states, transitions, rewards)
        new_path = self.path_from_policy(start, policy, goal)
        return new_path

    def build_states(self, path, obstacle_point):
        buffer = 5
        states = set(path)
        for x, y in path:
            for dx in range(-buffer, buffer + 1):
                for dy in range(-buffer, buffer + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 0:
                        states.add((nx, ny))
        return list(states)

    def build_transitions(self, states):
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        transitions = defaultdict(dict)
        for state in states:
            x, y = state
            for a in actions:
                next_states = []
                for dx, dy in [(a[0], a[1]), (a[1], a[0]), (-a[1], -a[0])]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in states and self.dynamic_grid[ny][nx] == 0:
                        prob = 0.8 if (dx, dy) == a else 0.1
                        next_states.append((prob, (nx, ny)))
                transitions[state][a] = next_states
        return transitions

    def build_rewards(self, states, goal, original_path):
        rewards = {}
        for s in states:
            x, y = s
            if s == goal:
                rewards[s] = 1000
            elif self.dynamic_grid[y][x] == 1:
                rewards[s] = -1000
            else:
                rewards[s] = -1
                if s in original_path:
                    rewards[s] += 10
        return rewards

    def solve_mdp(self, states, transitions, rewards, gamma=0.99, epsilon=1e-6):
        V = {s: 0.0 for s in states}
        policy = {s: (0, 0) for s in states}
        while True:
            delta = 0
            for s in states:
                if s == states[-1]:  # если цель
                    continue
                max_value = float('-inf')
                best_action = None
                for a in transitions[s]:
                    total = sum(prob * V[next_s] for prob, next_s in transitions[s][a])
                    value = rewards[s] + gamma * total
                    if value > max_value:
                        max_value = value
                        best_action = a
                delta = max(delta, abs(V[s] - max_value))
                V[s] = max_value
                policy[s] = best_action
            if delta < epsilon:
                break
        return policy

    def path_from_policy(self, start, policy, goal):
        path = [start]
        current = start
        max_steps = 1000
        for _ in range(max_steps):
            if current == goal:
                break
            if current not in policy:
                self.get_logger().warn('Политика не определена для текущей точки')
                break
            action = policy[current]
            next_state = (current[0] + action[0], current[1] + action[1])
            if next_state not in policy or self.dynamic_grid[next_state[1]][next_state[0]] == 1:
                self.get_logger().warn('Обход препятствия невозможен')
                break
            path.append(next_state)
            current = next_state
        return path

    def save_path_to_file(self, path):
        """Сохраняет путь в файл в формате списка кортежей"""
        try:
            with open(self.output_file, 'w') as f:
                f.write(str(path))
            self.get_logger().info(f'Новый путь сохранён в {self.output_file}')
        except Exception as e:
            self.get_logger().error(f'Ошибка сохранения пути в файл: {e}')
        subprocess.run(['python3', '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/Traectory.py'])


    def plot_paths(self):
        if not self.original_path:
            return

        plt.figure(figsize=(12, 8))
        grid_vis = np.copy(self.dynamic_grid)
        plt.imshow(grid_vis, cmap='gray', origin='lower', alpha=0.5)
        orig = np.array(self.original_path)
        plt.plot(orig[:, 0], orig[:, 1], 'r--', label='Оригинал')
        plt.plot(orig[:, 0], orig[:, 1], 'go', markersize=3)
        plt.plot(orig[:, 0], orig[:, 1], 'g-', linewidth=1)
        plt.plot(orig[0, 0], orig[0, 1], 'go', markersize=10, label='Начало')
        plt.plot(orig[-1, 0], orig[-1, 1], 'ro', markersize=10, label='Цель')
        plt.legend()
        plt.title('Путь после MDP перепланирования')
        plt.axis('equal')
        plt.grid(True)
        rclpy.shutdown()
        # plt.show()

    def run(self):
        rclpy.spin(self)
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MDPReplanner()
    node.run()
    

if __name__ == '__main__':
    main()

