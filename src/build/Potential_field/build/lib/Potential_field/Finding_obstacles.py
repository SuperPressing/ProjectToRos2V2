#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np

class ObstacleChecker(Node):
    def __init__(self):
        super().__init__('obstacle_checker')
        
        # Загрузка карты
        self.map_image = cv2.imread('/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/update_map.pgm', cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            self.get_logger().error("Не удалось загрузить карту!")
            exit(1)
        
        # Порог для определения препятствий (черный цвет)
        self.obstacle_threshold = 5
        
        # Подписка на путь
        self.path_sub = self.create_subscription(
            Path,
            '/potential_path',
            self.path_callback,
            10)
        
        # Публикация точек с препятствиями
        self.obstacle_pub = self.create_publisher(Path, '/obstacle_points', 10)
        
        self.get_logger().info("Obstacle checker node initialized")

    def path_callback(self, path_msg):
        obstacle_points = Path()
        obstacle_points.header = path_msg.header
        obstacle_list = []  # Для хранения точек препятствий для вывода в консоль
        
        for pose in path_msg.poses:
            x = int(pose.pose.position.x)
            y = int(pose.pose.position.y)
            
            if 0 <= y < self.map_image.shape[0] and 0 <= x < self.map_image.shape[1]:
                if self.map_image[y, x] <= self.obstacle_threshold:
                    obstacle_points.poses.append(pose)
                    obstacle_list.append((x, y))
        
        if len(obstacle_points.poses) > 0:
            self.get_logger().info(f"Найдено {len(obstacle_points.poses)} точек с препятствиями")
            
            # Вывод координат препятствий в консоль
            print("\nОбнаружены препятствия в точках (x, y):")
            for i, (x, y) in enumerate(obstacle_list, 1):
                print(f"{i}. ({x}, {y})")
            print()  # Пустая строка для разделения
            self.obstacle_pub.publish(obstacle_points)

def main(args=None):
    rclpy.init(args=args)
    checker = ObstacleChecker()
    rclpy.spin(checker)
    checker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()