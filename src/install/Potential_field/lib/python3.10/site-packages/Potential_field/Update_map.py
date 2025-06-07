import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from PIL import Image
import math
import os
import cv2
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_srvs.srv import Empty
class DynamicMapUpdater(Node):
    def __init__(self):
        super().__init__('dynamic_map_updater')
        
        # Параметры карты
        self.map_path =  '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/new_map.pgm'
        self.resolution = 20.9  # пикселей на метр
        self.origin_x = 250.0   # смещение по X (метры)
        self.origin_y = 150.0   # смещение по Y (метры)
        
        # Параметры для объединения точек
        line_param = ParameterDescriptor(
            description='Минимальное количество точек для объединения в линию',
            type=ParameterType.PARAMETER_INTEGER,
            additional_constraints='Значение > 0')
        kernel_param = ParameterDescriptor(
            description='Размер ядра для морфологических операций',
            type=ParameterType.PARAMETER_INTEGER,
            additional_constraints='Нечетное число > 0')
        
        self.declare_parameter('min_points_for_line', 10, line_param)
        self.declare_parameter('morph_kernel_size', 5, kernel_param)
        
        self.min_points_for_line = self.get_parameter('min_points_for_line').value
        self.morph_kernel_size = self.get_parameter('morph_kernel_size').value
        
        # Убедимся, что размер ядра нечетный
        if self.morph_kernel_size % 2 == 0:
            self.morph_kernel_size += 1
            self.get_logger().warn(f"Размер ядра должен быть нечетным. Установлено: {self.morph_kernel_size}")
        
        # Загрузка исходной карты
        self.load_original_map()
        
        # Текущая позиция робота
        self.current_x = 0.0
        self.current_y = 0.0
        self.yaw = 0.0
        
        # Подписки
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Публикатор обновленной карты
        self.map_pub = self.create_publisher(OccupancyGrid, '/updated_map', 10)
        
        # Таймер для публикации карты
        self.create_timer(1.0, self.publish_map)
        
        # Таймер для обработки точек
        self.create_timer(2.0, self.process_points)  # Обработка каждые 2 секунды
        self.reset_srv = self.create_service(Empty, 'reset_map', self.reset_callback)
        self.get_logger().info(f"Node initialized. Min points for line: {self.min_points_for_line}, Kernel size: {self.morph_kernel_size}")

    def load_original_map(self):
        """Загрузка исходной карты из PGM файла"""
        try:
            self.original_map_image = Image.open(self.map_path)
            self.map_width, self.map_height = self.original_map_image.size
            
            # Конвертируем в grayscale numpy array
            self.original_map = np.array(self.original_map_image.convert('L'))
            
            # Создаем рабочую копию карты
            self.updated_map = self.original_map.copy()
            
            # Маска для новых препятствий
            self.new_obstacles = np.zeros_like(self.original_map, dtype=bool)
            
            self.get_logger().info(f"Map loaded. Size: {self.map_width}x{self.map_height}")
        except Exception as e:
            self.get_logger().error(f"Failed to load map: {str(e)}")
            rclpy.shutdown()

    def odom_callback(self, msg):
        """Обработка позиции робота"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # self.current_x = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        # self.current_y = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        self.current_x = x*20.9+245
        self.current_y = y*20.9+150
        yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.yaw= yaw + math.pi

    def scan_callback(self, msg):
        """Обработка данных лидара и обновление карты"""
        if not hasattr(self, 'original_map'):
            return

        for i, distance in enumerate(msg.ranges):
            # Пропуск невалидных измерений
            if distance < msg.range_min or distance > 5:
                continue
                
            # Вычисление угла луча
            angle = msg.angle_min + i * msg.angle_increment + self.yaw
            
            # Координаты препятствия относительно робота
            obstacle_rel_x = distance*20 * math.cos(angle+math.pi)
            obstacle_rel_y = distance*20 * math.sin(-angle+math.pi)
            
            # Глобальные координаты препятствия
            obstacle_global_x = self.current_x + obstacle_rel_x
            obstacle_global_y = self.current_y + obstacle_rel_y
            
            # Конвертация в координаты карты (пиксели)
            # map_x = int((obstacle_global_x - self.origin_x) * self.resolution)
            # map_y = int((self.origin_y - obstacle_global_y) * self.resolution + self.map_height)
            map_x = int(obstacle_global_x)
            map_y = int(obstacle_global_y)
            # Проверка границ карты
            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                # Проверка, было ли место свободным
                if self.original_map[map_y, map_x] > 200:  # Белый цвет = свободно
                    # Помечаем препятствие (0 = занято)
                    self.updated_map[map_y, map_x] = 0
                    self.get_logger().debug(f"New obstacle at [{map_x}, {map_y}]")

    def process_points(self):
        """Обработка новых точек: объединение в линии и удаление изолированных"""
        if not hasattr(self, 'new_obstacles') or np.sum(self.new_obstacles) == 0:
            return
            
        try:
            # Создаем бинарное изображение новых препятствий
            binary_new = np.where(self.new_obstacles, 255, 0).astype(np.uint8)
            
            # Морфологические операции для объединения точек
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            closed = cv2.morphologyEx(binary_new, cv2.MORPH_CLOSE, kernel)
            
            # Находим контуры
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Создаем маску для допустимых объектов
            valid_obstacles = np.zeros_like(binary_new)
            
            for contour in contours:
                # Фильтрация по размеру контура (используем min_points_for_line)
                area = cv2.contourArea(contour)
                if area >= self.min_points_for_line:  # Изменили условие
                    cv2.drawContours(valid_obstacles, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Обновляем карту
            self.updated_map[self.new_obstacles] = self.original_map[self.new_obstacles]
            self.updated_map[valid_obstacles > 0] = 0
            
            self.new_obstacles = (valid_obstacles > 0)
            
            self.get_logger().info(
                f"Processed points: kept {np.sum(valid_obstacles)//255} points "
                f"(min {self.min_points_for_line} points required)"
            )
            
        except Exception as e:
            self.get_logger().error(f"Point processing failed: {str(e)}")

    def publish_map(self):
        """Публикация обновленной карты в формате OccupancyGrid"""
        if not hasattr(self, 'updated_map'):
            return
            
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        # Задание параметров карты
        msg.info.resolution = 1.0 / self.resolution  # Разрешение в метрах/пиксель
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        
        # Позиция левого нижнего угла карты
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y - self.map_height / self.resolution
        msg.info.origin.position.z = 0.0
        
        # Конвертация изображения в occupancy data
        map_data = self.updated_map.astype(np.int8).flatten()
        
        # Конвертация цветов в значения occupancy:
        #   0 (черный) -> 100 (занято)
        #   255 (белый) -> 0 (свободно)
        #   Остальное -> -1 (неизвестно)
        occupancy_data = np.where(map_data < 50, 100, 
                                 np.where(map_data > 200, 0, -1))
        
        msg.data = occupancy_data.tolist()
        self.map_pub.publish(msg)
        self.get_logger().debug("Updated map published")

    def save_map(self, filename='updated_map.pgm'):
        """Сохранение карты в файл"""
        try:
            Image.fromarray(self.updated_map).save(filename)
            self.get_logger().info(f"Map saved to {os.path.abspath(filename)}")
        except Exception as e:
            self.get_logger().error(f"Failed to save map: {str(e)}")
    
    def reset_callback(self, request, response):
        """Обработчик сервиса сброса карты"""
        self.updated_map = self.original_map.copy()
        self.new_obstacles = np.zeros_like(self.original_map, dtype=bool)
        self.get_logger().info("Map reset to original state")
        return response
    
    @staticmethod
    def euler_from_quaternion(quat):
        import math
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        # Расчёт yaw (вращение вокруг оси Z)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw  # возвращаем только угол поворота (yaw)
    
def main(args=None):
    rclpy.init(args=args)
    node = DynamicMapUpdater()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_map('/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/update_map.pgm') # Сохранение при завершении
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()