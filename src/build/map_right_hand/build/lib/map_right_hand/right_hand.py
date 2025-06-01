import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
import math
import time
class WallFollower(Node):

    def __init__(self):
        super().__init__('wall_follower')
        self.Kp = 0.001  # Пропорциональный коэффициент
        self.Kd = 0.01  # Дифференциальный коэффициент
        self.last_error = 0.0  # Предыдущая ошWибка
        self.max_angular_speed = 0.1  # Максимальная угловая скорость
        self.last_time = self.get_clock().now().nanoseconds * 1e-9  # в секундах
        # Подписки
        self.create_subscription(OccupancyGrid, '/map_modified', self.map_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.search = True
        # Публикатор
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        start = True
        if(start):
            twist = Twist()
            twist.angular.z = 1.0
            start = False
            self.cmd_pub.publish(twist)
        # Переменные для работы
        self.map_data = None
        self.map_info = None
        self.robot_x = 0
        self.robot_y = 0
        self.robot_yaw = 0.0

    def map_callback(self, msg):
        """Получаем карту"""
        self.map_data = list(msg.data)
        self.map_info = msg.info
        

    def pose_callback(self, msg):
        """Получаем позицию робота"""
        # if not self.map_info:
        #     return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.euler_from_quaternion(msg.pose.pose.orientation)

        self.robot_x = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        self.robot_y = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        self.robot_yaw = yaw

        self.follow_wall(self.search)

    def follow_wall(self):

        # if not self.map_info or not self.map_data:
        #     return

        twist = Twist()

        # Расстояние до стены, которое считаем критическим (в ячейках)
        wall_distance_for_turn = 15 # например, если стена ближе чем 2 ячейки → начинаем поворачивать

        right, dist = self.is_wall_on_right(self.robot_x, self.robot_y, self.robot_yaw, wall_distance_for_turn+30)
        front = self.is_obstacle_in_front(self.robot_x, self.robot_y, self.robot_yaw, wall_distance_for_turn)
        f_r = self.f_r(self.robot_x, self.robot_y, self.robot_yaw, wall_distance_for_turn)
        # self.get_logger().info(f'Препядствие спреди  {front} и справа {right}!')
        ref_dist = 20
        dist  = abs(dist)
        if self.search:
            twist.angular.z = 5.0
            # self.get_logger().info("Еду вперёд  до стены!")
            if front:
                self.search = False
                # self.get_logger().info("Конец стартового движения!")
                twist.angular.z = 0.0


        # Правило правой руки
        if not self.search:
            angular_speed = 0
            if (front and right) or front:
                # Если стена спереди или спереди и справа — отступаем
                twist.linear.x = -0.5
                twist.angular.z = 0.0
            elif right:
                # Стена справа — включаем PD-регулятор
                error = dist - ref_dist
                current_time = self.get_clock().now().nanoseconds * 1e-9
                dt = current_time - self.last_time
                d_error = (error - self.last_error) / dt if dt > 0 else 0
                self.last_error = error
                self.last_time = current_time
                # if (abs(error) >0.5):
                    # Вычисляем угловую скорость через PD
                angular_speed = self.Kp * error + self.Kd * d_error
                angular_speed = max(-self.max_angular_speed, min(angular_speed, self.max_angular_speed))
                twist.linear.x = angular_speed
                twist.angular.z= 2.0  # Двигаемся вперёд с постоянной скоростью
                self.get_logger().info(f'error: {error} dist {dist} angular_speed {angular_speed}')
            elif(not right and not front and f_r):
                twist.angular.z = 1.5
            else:
                twist.linear.x = 0.5
                # self.get_logger().info("Поварачиваю на право!")

        
        right_x = self.robot_x + int(math.cos(self.robot_yaw + math.pi / 2))
        right_y = self.robot_y + int(math.sin(self.robot_yaw + math.pi / 2))
        front_x = self.robot_x + int(math.cos(self.robot_yaw))
        front_y = self.robot_y + int(math.sin(self.robot_yaw))
        # self.get_logger().info(f'robot_x: {self.robot_x}, robot_y: {self.robot_y}')
        right = self.is_occupied(right_x, right_y)
        front = self.is_occupied(front_x, front_y)
        # self.get_logger().info(f'Справа: {right_x}, {right_y} → {"стена" if right else "свободно"}')
        # self.get_logger().info(f'Спереди: {front_x}, {front_y} → {"стена" if front else "свободно"}')
        self.cmd_pub.publish(twist)

    def pose_callback(self, msg):
        if not self.map_info:
            self.get_logger().warn("Карта ещё не загружена")
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.euler_from_quaternion(msg.pose.pose.orientation)

        self.robot_x = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        self.robot_y = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        self.robot_yaw = yaw

        # Логируем позицию и угол
        # self.get_logger().info(f'Получена позиция: ({self.robot_x}, {self.robot_y}), yaw: {math.degrees(yaw):.2f}°')
        
        self.follow_wall()


    def is_wall_on_right(self, x, y, yaw, max_cells=10):
        """Проверяет, есть ли стена справа на заданном расстоянии"""
        dist = 0
        for i in range(1, max_cells + 1):
            nx_0 = x + int(math.cos(yaw+ math.pi/2) * (i-1))
            ny_0 = y + int(math.sin(yaw+ math.pi/2) * (i-1))
            if (self.is_occupied(nx_0, ny_0)):
                dist = self.check_wall(x, y, 100)
                return True, abs(dist)  # стена найденаc
                
        return False, abs(dist)
    
    def is_obstacle_in_front(self, x, y, yaw, max_cells=50):
        """
        Проверяет наличие стены на расстоянии до max_cells ячеек вперёд
        """
        for i in range(1, max_cells + 1):
            nx = x + (math.cos(yaw + math.pi) * i)
            ny = y + (math.sin(yaw +math.pi) * i)

            # self.get_logger().info(f'зНАЧЕНИЕ СИНУСА: ({math.sin(yaw+math.pi / 2 + math.pi)})')
            nx = round(nx)
            ny = round(ny) 
            # self.get_logger().info(f'Координаты поиска: {ny}')
            # self.get_logger().info(f'Angel: ({yaw}')
            if self.is_occupied(nx, ny):
                #self.get_logger().info(f'Координаты стены: ({nx}, {ny})')
                return True  # стена найдена
            
        return False  # препятствий нет

    def is_occupied(self, x, y):
        """Проверяем, занята ли ячейка"""
        # if not self.map_info or not self.map_data:
            # return True

        width = self.map_info.width
        height = self.map_info.height
        idx = x + y * width
        if 0 <= idx < len(self.map_data):
            cell_value = self.map_data[idx]
            return cell_value == 100  # 100 = занято
        if not (0 <= x < width and 0 <= y < height):
            return True  # вне карты = занято
        return False  # если за пределами карты — считаем как стену
    
    def check_wall(self, x, y, d=10):
        dist = 100
        dist_x = 0
        for i in range(d):
            x_0 = int(x + i-d/2)
            for iy in range(d):
                y_0 = int(y+i-d/2)
                dx = x - x_0
                dy = y - y_0
                dist_x = math.sqrt(dx**2 + dy**2)
                if (self.is_occupied(x_0, y_0) and dist_x<dist):
                    dist = dist_x
        return dist
    
    def f_r(self, x, y, yaw, max_cells=50):
        """Проверяет, есть ли стена справа на заданном расстоянии"""
        dist = 0
        for i in range(1, max_cells + 1):
            nx_0 = x + int(math.cos(yaw+ math.pi/2) * (i-1))
            ny_0 = y + int(math.sin(yaw+ math.pi/2) * (i-1+5))
            if (self.is_occupied(nx_0, ny_0)):
                dist = self.check_wall(x, y, 100)
                return True, abs  # стена найденаc
                
        return False
    
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
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()