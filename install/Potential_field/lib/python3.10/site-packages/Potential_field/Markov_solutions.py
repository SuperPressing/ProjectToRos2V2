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


class MDPReplanner(Node):
    def __init__(self):
        super().__init__('mdp_replanner_node')
        self.get_logger().info("MDP Replanner запущен")

        # Подписки
        self.path_sub = self.create_subscription(Path, '/path', self.path_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)

        # Публикация нового пути
        self.replanned_path_pub = self.create_publisher(Path, '/replanned_path', 10)

        # Исходная и сглаженная траектории
        self.original_path = []
        self.current_position = (0, 0)
        self.dynamic_grid = None
        self.grid = None
        self.resolution = 0.05
        self.origin = (0.0, 0.0)
        self.width = 0
        self.height = 0

    def path_callback(self, msg):
        self.original_path = [(int((pose.pose.position.x - self.origin[0]) / self.resolution),
                              int((pose.pose.position.y - self.origin[1]) / self.resolution)]
        for pose in msg.poses:
            x = int((pose.pose.position.x - self.origin[0]) / self.resolution)
            y = int((pose.pose.position.y - self.origin[1]) / self.resolution)
            self.original_path.append((x, y))
        self.get_logger().info(f'Получена траектория с {len(self.original_path)} точками')
        self.plot_paths()

    def pose_callback(self, msg):
        x = int((msg.pose.pose.position.x - self.origin[0]) / self.resolution)
        y = int((msg.pose.pose.position.y - self.origin[1]) / self.resolution)
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

        # Проверяем, сталкивается ли робот с препятствием
        for point in self.original_path:
            x, y = point
            if self.dynamic_grid[y][x] == 1:
                self.get_logger().warn(f'Препятствие на траектории в точке ({x}, {y})')
                self.original_path = self.mdp_replan(self.original_path, point)
                self.plot_paths()
                break

    def load_map(self):
        """Загрузка карты (пример, можно заменить на /map callback)"""
        try:
            with open('new_map.yaml', 'r') as f:
                self.metadata = yaml.safe_load(f)
            img = Image.open('new_map.pgm').convert('L')
            self.width, self.height = img.size
            self.resolution = self.metadata['resolution']
            self.origin = self.metadata['origin']
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
        """
        Строит новую траекторию с обходом препятствия и возвратом на исходную траекторию
        :param path: оригинальная траектория
        :param obstacle_point: точка препятствия (x, y)
        :return: новая траектория
        """
        self.get_logger().info('Запуск MDP перепланирования...')

        # Используем текущее положение и цель
        start = self.current_position
        goal = path[-1]

        # Создаём MDP модель
        states = self.build_states(path, obstacle_point)
        transitions = self.build_transitions(states)
        rewards = self.build_rewards(states, goal, path)
        policy = self.solve_mdp(states, transitions, rewards)

        # Восстанавливаем путь
        new_path = self.path_from_policy(start, policy, goal)
        return new_path

    def build_states(self, path, obstacle_point):
        """Создаёт список состояний (x, y) вокруг траектории"""
        buffer = 5  # точки вокруг траектории, которые учитываются
        states = set(path)
        for x, y in path:
            for dx in range(-buffer, buffer + 1):
                for dy in range(-buffer, buffer + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[ny][nx] == 0:
                            states.add((nx, ny))
        return list(states)

    def build_transitions(self, states):
        """Создаёт переходы между состояниями с шумом"""
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
        """
        Строит функцию награды:
        - +1000 за достижение цели
        - -1 за движение
        - -1000 за столкновение с препятствием
        - +10 за возврат на оригинальную траекторию
        """
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
                    rewards[s] += 10  # Поощрение за возврат на путь
        return rewards

    def solve_mdp(self, states, transitions, rewards, gamma=0.99, epsilon=1e-6):
        """
        Решает MDP через Value Iteration
        """
        V = {s: 0.0 for s in states}
        policy = {s: (0, 0) for s in states}

        while True:
            delta = 0
            for s in states:
                if s == goal:
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
        """Восстанавливает путь из политики MDP"""
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

    def publish_path(self, path):
        """Публикует путь как Path сообщение"""
        msg = Path()
        msg.header = Header()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in path:
            pose = PoseStamped()
            pose.header = Header()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x * self.resolution + self.origin[0]
            pose.pose.position.y = y * self.resolution + self.origin[1]
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.replanned_path_pub.publish(msg)
        self.get_logger().info('Новый путь опубликован на /replanned_path')

    def plot_paths(self):
        """Отображает оригинальный и обновлённый путь"""
        if not self.original_path:
            return

        plt.figure(figsize=(12, 8))
        grid_vis = np.copy(self.dynamic_grid)
        plt.imshow(grid_vis, cmap='gray', origin='lower', alpha=0.5)

        orig = np.array(self.original_path)
        plt.plot(orig[:, 0], orig[:, 1], 'r--', label='Оригинал')

        plt.plot(orig[:, 0], orig[:, 1], 'go', markersize=3, label='Оригинал')
        plt.plot(orig[:, 0], orig[:, 1], 'g-', linewidth=1, label='Оригинал')

        plt.plot(orig[0, 0], orig[0, 1], 'go', markersize=10, label='Начало')
        plt.plot(orig[-1, 0], orig[-1, 1], 'ro', markersize=10, label='Цель')
        plt.legend()
        plt.title('Путь после MDP перепланирования')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def run(self):
        """Запуск ноды"""
        rclpy.spin(self)
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MDPReplanner()
    node.load_map()
    node.run()


if __name__ == '__main__':
    main()