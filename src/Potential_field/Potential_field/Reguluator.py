import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math
import ast
import subprocess
import json
class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

 
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription = self.create_subscription(
            JointTrajectory,
            '/Trac',
            self.path_callback,
            10)
        self.markov = False
        self.filter_v = FirstOrderFilter(alpha=0.1)
        self.filter_w = FirstOrderFilter(alpha=0.1)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(OccupancyGrid, '/map_modified', self.map_callback, 10)
        file_path = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/trajectory_output.txt'
        self.v_control = 0
        self.w_control = 0
        
        self.time_data = []
        self.x_positions = []
        self.y_positions = []
        self.orientations = []
        self.v = []
        self.w = []
        self.current_waypoint_index = 0


        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Гц
        self.map_info = None
        self.map_data = None

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_v = 0.0
        self.current_w = 0.0
        self.v_control = 0.0
        self.w_control = 0.0

        self.K1 = 1.5 
        self.K2 = 1.2

    def path_callback(self, msg):
        """Обработчик сообщений с траекторией"""
        self.get_logger().info(f"Получена траектория с {len(msg.points)} точками")
        self.current_waypoint_index = 0

        self.time_data.clear()
        self.x_positions.clear()
        self.y_positions.clear()
        self.orientations.clear()
        

        if len(msg.points) < 3:
            self.get_logger().error("Недостаточно joint_names в сообщении!")
            return
        

        for point in msg.points:

            time_sec = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            self.time_data.append(time_sec)
            self.v.append(point.velocities[0])
            self.w.append(point.velocities[1])

            self.x_positions.append(point.positions[0])
            self.y_positions.append(point.positions[1])
            self.orientations.append(point.positions[2])  
            

    
    def process_trajectory_data(self):
        """Обработка полученных данных"""
        self.get_logger().info("Обработка данных траектории...")
        

    def map_callback(self, msg):
        """Получаем карту"""
        self.map_data = list(msg.data)
        self.map_info = msg.info

    def odom_callback(self, msg):
        if not self.map_info:
           return

        pose = msg.pose.pose
        twist = msg.twist.twist
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        

        self.current_x = x*20.9+250
        self.current_y = y*20.9+150
        yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.current_theta = yaw + math.pi



    def angle_diff(self, a, b):
        diff = (a - b + math.pi) % (2 * math.pi) - math.pi
        return diff if diff > -math.pi else diff + 2 * math.pi
    
    def control_loop(self):
        if not self.map_info:
           return
        cmd = Twist()
        if self.current_waypoint_index >= len(self.x_positions):
            self.current_waypoint_index = 0
            self.time_data.clear()
            self.x_positions.clear()
            self.y_positions.clear()
            self.orientations.clear()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().info("Траектория завершена.")
            
            return


        e1 = self.x_positions [self.current_waypoint_index]- self.current_x
        e2 = self.y_positions[self.current_waypoint_index] - self.current_y
        e3 = self.angle_diff(self.orientations[self.current_waypoint_index], self.current_theta)

        k1 = 5
        k2 = 1
        s = e1*math.cos(self.orientations[self.current_waypoint_index]+e3)+e2*math.sin(self.orientations[self.current_waypoint_index]+e3)
        u1 = 0
        p = self.current_waypoint_index
        if s > 0.01:
          u1 = -self.v[p]+(self.v[p]*(e1*math.cos(self.orientations[p])+e2*math.sin(self.orientations[p])))/(e1*math.cos(self.orientations[p]+e3)+e2*math.sin(self.orientations[p]+e3)+0.0001)-k1*(e1*math.cos(self.orientations[p]+e3)+e2*math.sin(self.orientations[p]+e3))
        u2 = -k2*e3
        if(abs(u1) > 10 or abs(u2)>5):
            print(f'u1 {u1} u2 {u2}')
            print(f'Текущий x {self.current_x} Рассчитанный x {self.orientations}')
            print(f'Текущий y {self.current_y} Рассчитанный y {self.orientations}')
            print(f'Текущий theta {self.current_theta} Рассчитанный theta {self.orientations} ошибка е3 {e3}')
        v_control = self.v[p] + u1
        w_control = self.w[p] + u2

        v_control = max(-2.0, min(v_control, 2.0))
        w_control = max(-0.1, min(w_control, 0.1))


        v_filtered = self.filter_v.update(v_control)
        w_filtered = self.filter_w.update(w_control)
        
        cmd.linear.x = w_filtered
        cmd.angular.z = -v_filtered
        self.current_waypoint_index += 1
        self.cmd_pub.publish(cmd)

    def update(self, new_value):
        self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
    @staticmethod
    def euler_from_quaternion(quat):
        import math
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw  


def main(args=None):
    rclpy.init(args=args)
    follower = TrajectoryFollower()
    rclpy.spin(follower)

    follower.destroy_node()
    rclpy.shutdown()

class FirstOrderFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_value = 0.0

    def update(self, new_value):
        self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
    
if __name__ == '__main__':
    main()
    
