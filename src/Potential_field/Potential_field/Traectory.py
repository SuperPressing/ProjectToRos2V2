import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from scipy.interpolate import interp1d
import builtin_interfaces.msg
import matplotlib.pyplot as plt
import os
from datetime import datetime
from builtin_interfaces.msg import Duration
class TrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('trajectory_planner')

        self.r = 0.05      
        self.L = 0.35      
        self.m = 5.0     
        self.mu = 0.02      
        self.I = 64.0     
        self.g = 9.81       
        self.v_max = 5.0   
        self.a = 0.5        
        self.dt = 0.1    


        self.subscription = self.create_subscription(
            Path,
            '/potential_path',
            self.path_callback,
            10)
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/Trac',
            10)
        

        self.plot_dir = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field'
        os.makedirs(self.plot_dir, exist_ok=True)
        self.get_logger().info(f"Saving plots to: {self.plot_dir}")

    def path_callback(self, msg):

        path = []
        for pose in msg.poses:
            path.append([pose.pose.position.x, pose.pose.position.y])
        path = np.array(path)

        if len(path) < 2:
            self.get_logger().warn("Received path with less than 2 points")
            return


        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        total_length = np.sum(distances)
        

        t_accel = self.v_max / self.a
        s_accel = 0.5 * self.a * t_accel**2

        if 2 * s_accel > total_length:
            t_accel = np.sqrt(total_length / self.a)
            t_constant = 0.0
            t_total = 2 * t_accel
        else:
            s_constant = total_length - 2 * s_accel
            t_constant = s_constant / self.v_max
            t_total = 2 * t_accel + t_constant


        t_eval = np.arange(0, t_total, self.dt)
        

        v_profile = []
        for t in t_eval:
            if t <= t_accel:
                v = self.a * t
            elif t <= t_accel + t_constant:
                v = self.v_max
            else:
                v = self.a * (t_total - t)
            v_profile.append(v)
        

        a_profile = np.gradient(v_profile, self.dt)
        

        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        x_path = interp1d(cumulative_distances, path[:, 0], kind='linear', fill_value="extrapolate")
        y_path = interp1d(cumulative_distances, path[:, 1], kind='linear', fill_value="extrapolate")
        

        s_profile = np.cumsum(v_profile) * self.dt
        trajectory_x = x_path(s_profile)
        trajectory_y = y_path(s_profile)
        

        dx = np.gradient(trajectory_x, self.dt)
        dy = np.gradient(trajectory_y, self.dt)
        theta = np.arctan2(dy, dx)
        omega_profile = np.gradient(theta, self.dt)
        

        def inverse_kinematics(v, omega):
            omega_L = (v - (self.L / 2) * omega) / self.r
            omega_R = (v + (self.L / 2) * omega) / self.r
            return omega_L, omega_R

        omega_L_profile, omega_R_profile = [], []
        for v, omega in zip(v_profile, omega_profile):
            wl, wr = inverse_kinematics(v, omega)
            omega_L_profile.append(wl)
            omega_R_profile.append(wr)
        

        F_traction = self.m * a_profile + self.mu * self.m * self.g
        M_rotation = self.I * np.gradient(omega_profile, self.dt) + self.mu * self.m * self.g * self.L / 2


        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        current_time = self.get_clock().now()
        current_time = current_time.nanoseconds / 1e9
        traj_msg.header.frame_id = "odom"
        traj_msg.joint_names = ['left_wheel_joint', 'right_wheel_joint']
        
        for i in range(len(t_eval)):
            point = JointTrajectoryPoint()
            point.velocities = [
                float(v_profile[i]),
                float(omega_profile[i]),
            ]
            point.positions = [
                float(trajectory_x[i]),  
                float(trajectory_y[i]), 
                float(theta[i])      
            ]
    
            sec = int(t_eval[i]+current_time)  
            nanosec = int((t_eval[i] - sec+current_time) * 1e9)  
            
            point.time_from_start = builtin_interfaces.msg.Duration(sec=sec,nanosec = nanosec )
            traj_msg.points.append(point)
        
        self.trajectory_pub.publish(traj_msg)
        self.get_logger().info(f"Published trajectory with {len(traj_msg.points)} points")
        
      
        self.generate_plots(
            path, 
            trajectory_x, 
            trajectory_y, 
            theta, 
            t_eval, 
            v_profile, 
            a_profile,
            omega_L_profile, 
            omega_R_profile, 
            F_traction
        )

    def generate_plots(self, path, trajectory_x, trajectory_y, theta, t_eval, 
                       v_profile, a_profile, omega_L_profile, omega_R_profile, F_traction):
    
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Trajectory Analysis', fontsize=16)
        
      
        axs[0, 0].plot(path[:, 0], path[:, 1], 'ro-', label='Input Path')
        axs[0, 0].plot(trajectory_x, trajectory_y, 'b-', label='Actual Trajectory')
        axs[0, 0].set_title("Robot Trajectory")
        axs[0, 0].set_xlabel("X [m]")
        axs[0, 0].set_ylabel("Y [m]")
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        axs[0, 0].axis('equal')
        
     
        step = max(1, len(trajectory_x) // 10)
        arrow_length = 0.3
        for i in range(0, len(trajectory_x), step):
            x = trajectory_x[i]
            y = trajectory_y[i]
            angle = theta[i]
            dx_arrow = arrow_length * np.cos(angle)
            dy_arrow = arrow_length * np.sin(angle)
            axs[0, 0].arrow(x, y, dx_arrow, dy_arrow,
                            head_width=0.05, length_includes_head=True, color='blue')

       
        axs[0, 1].plot(t_eval, v_profile, 'g-', label='Linear velocity')
        axs[0, 1].set_title("Linear Velocity")
        axs[0, 1].set_xlabel("Time [s]")
        axs[0, 1].set_ylabel("Velocity [m/s]")
        axs[0, 1].grid(True)
        axs[0, 1].legend()

      
        axs[1, 0].plot(t_eval, a_profile, 'm-', label='Linear acceleration')
        axs[1, 0].set_title("Linear Acceleration")
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 0].set_ylabel("Acceleration [m/s²]")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

      
        axs[1, 1].plot(t_eval, np.degrees(theta), 'c-', label='Robot heading')
        axs[1, 1].set_title("Robot Orientation")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].set_ylabel("Heading [deg]")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        
        axs[2, 0].plot(t_eval, omega_L_profile, 'b', label='Left wheel')
        axs[2, 0].plot(t_eval, omega_R_profile, 'r', label='Right wheel')
        axs[2, 0].set_title("Wheel Angular Velocities")
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_ylabel("Angular velocity [rad/s]")
        axs[2, 0].grid(True)
        axs[2, 0].legend()

       
        axs[2, 1].plot(t_eval, F_traction, 'purple', label='Traction force')
        axs[2, 1].set_title("Traction Force")
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 1].set_ylabel("Force [N]")
        axs[2, 1].grid(True)
        axs[2, 1].legend()

      
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.plot_dir, f'trajectory_analysis_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig) 
        
        self.get_logger().info(f"Saved trajectory plots to: {plot_path}")

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
