import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import os
import csv
import time

class MDPReplanner(Node):
    def __init__(self):
        super().__init__('mdp_replanner_node')
        self.get_logger().info("MDP Replanner initialized")

        # Parameters
        self.resolution = 0.05  # meters per pixel
        self.replanning_buffer = 5  # grid cells around path to consider for states
        self.lookahead_distance = 1.0  # meters to check for obstacles ahead
        
        # File paths
        self.output_dir = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/path_data'
        os.makedirs(self.output_dir, exist_ok=True)
        self.original_path_file = os.path.join(self.output_dir, 'original_path.csv')
        self.replanned_path_file = os.path.join(self.output_dir, 'replanned_path.csv')
        self.map_image_file = os.path.join(self.output_dir, 'map.png')
        
        # State variables
        self.original_path = []
        self.current_position = (0, 0)
        self.current_orientation = 0.0
        self.grid = None
        self.dynamic_grid = None
        self.width = 0
        self.height = 0
        self.origin = (0.0, 0.0)
        self.map_loaded = False
        self.current_x = 0
        self.current_y = 0

        # Publishers and Subscribers
        self.path_pub = self.create_publisher(Path, '/replanned_path', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Path, '/potential_path', self.path_callback, 10)
        
        # Timer for periodic path checking
        self.create_timer(1.0, self.check_path)  # Check path every 1 second
        
        # Load map
        self.load_map()

    def load_map(self):
        """Load the static map from file"""
        map_path = '/home/neo/Documents/ros2_ws/src/Potential_field/Potential_field/new_map'
        try:
            with open(f'{map_path}.yaml', 'r') as f:
                map_metadata = yaml.safe_load(f)
            
            img = Image.open(f'{map_path}.pgm').convert('L')
            self.width, self.height = img.size
            self.resolution = map_metadata['resolution']
            
            if isinstance(map_metadata['origin'], list) and len(map_metadata['origin']) >= 2:
                self.origin = (map_metadata['origin'][0], map_metadata['origin'][1])
            else:
                self.get_logger().error("Unexpected origin format in map YAML")
                raise ValueError("Invalid origin format")
            
            pixels = list(img.getdata())
            self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
            
            for y in range(self.height):
                for x in range(self.width):
                    idx = x + y * self.width
                    self.grid[y][x] = 0 if pixels[idx] > 250 else 1
            
            self.dynamic_grid = np.copy(self.grid)
            self.map_loaded = True
            self.get_logger().info(f"Map loaded successfully. Size: {self.width}x{self.height}")
            
            plt.imsave(self.map_image_file, self.grid, cmap='gray')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load map: {str(e)}')
            raise

    def odom_callback(self, msg):
        """Update current robot position from odometry"""
        if not self.map_loaded:
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        
        self.current_position = (grid_x, grid_y)
        self.current_orientation = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.current_x = x*20.9+250
        self.current_y = y*20.9+150

    def path_callback(self, msg):
        """Receive and store the original planned path"""
        self.original_path = []
        for pose in msg.poses:
            x = int((pose.pose.position.x - self.origin[0]) / self.resolution)
            y = int((pose.pose.position.y - self.origin[1]) / self.resolution)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.original_path.append((x, y))
        
        self.get_logger().info(f'Received path with {len(self.original_path)} points')
        self.save_path_to_file(self.original_path, self.original_path_file)
        self.visualize_path()

    def check_path(self):
        """Periodically check if path is blocked and replan if needed"""
        if not self.original_path or not self.map_loaded:
            return

        if self.is_path_blocked():
            self.get_logger().warn("Path blocked - replanning...")
            return_point = self.find_safe_return_point()
            
            if return_point:
                detour_path = self.replan_detour(self.current_position, return_point)
                
                if detour_path:
                    full_path = detour_path + self.get_remaining_path(return_point)
                    self.publish_path(full_path)
                    self.save_path_to_file(full_path, self.replanned_path_file)
                    self.visualize_path(full_path)
                else:
                    self.get_logger().error("Failed to replan detour path")
            else:
                self.get_logger().warn("No safe return point found")

    def is_path_blocked(self):
        """Check if path intersects with obstacles"""
        for x, y in self.original_path:
            if self.dynamic_grid[y][x] == 1:
                return True
        return False

    def find_safe_return_point(self):
        """Find nearest point on original path beyond obstacle"""
        closest_idx = 0
        min_dist = float('inf')
        for i, (x, y) in enumerate(self.original_path):
            dist = math.sqrt((x - self.current_position[0])**2 + 
                          (y - self.current_position[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Find first safe point beyond obstacle
        for i in range(closest_idx + 1, len(self.original_path)):
            x, y = self.original_path[i]
            if self.dynamic_grid[y][x] == 0:  # Free space
                # Check surrounding area
                safe = True
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.width and 0 <= ny < self.height and
                            self.dynamic_grid[ny][nx] == 1):
                            safe = False
                            break
                    if not safe:
                        break
                if safe:
                    return (x, y)
        return None

    def replan_detour(self, start, goal):
        """Replan path using MDP approach"""
        states = self.build_states_around_points([start, goal])
        transitions = self.build_transitions(states)
        rewards = self.build_rewards(states, goal)
        
        policy = self.value_iteration(states, transitions, rewards)
        return self.extract_path_from_policy(start, goal, policy)

    def build_states_around_points(self, key_points):
        """Build state space around key points"""
        states = set()
        buffer_size = 10  # Grid cells around points
        
        for point in key_points:
            px, py = point
            for dx in range(-buffer_size, buffer_size + 1):
                for dy in range(-buffer_size, buffer_size + 1):
                    nx, ny = px + dx, py + dy
                    if (0 <= nx < self.width and 0 <= ny < self.height and
                        self.dynamic_grid[ny][nx] == 0):
                        states.add((nx, ny))
        
        return list(states)

    def build_transitions(self, states):
        """Build transition model for MDP"""
        transitions = defaultdict(dict)
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connected movement
        
        for state in states:
            x, y = state
            for action in actions:
                outcomes = []
                # Intended movement
                nx, ny = x + action[0], y + action[1]
                if (nx, ny) in states:
                    outcomes.append((0.8, (nx, ny)))
                else:
                    outcomes.append((0.8, state))
                
                # Lateral movements
                for da in [(action[1], -action[0]), (-action[1], action[0])]:
                    nx, ny = x + da[0], y + da[1]
                    if (nx, ny) in states:
                        outcomes.append((0.1, (nx, ny)))
                    else:
                        outcomes.append((0.1, state))
                
                transitions[state][action] = outcomes
                
        return transitions

    def build_rewards(self, states, goal):
        """Build reward function for MDP"""
        rewards = {}
        for state in states:
            x, y = state
            if state == goal:
                rewards[state] = 100.0
            elif self.dynamic_grid[y][x] == 1:
                rewards[state] = -100.0
            else:
                rewards[state] = -1.0
                if state in self.original_path:
                    rewards[state] += 5.0
        return rewards

    def value_iteration(self, states, transitions, rewards, gamma=0.95, epsilon=1e-4):
        """Solve MDP using value iteration"""
        V = {s: 0.0 for s in states}
        policy = {s: (0, 0) for s in states}
        
        while True:
            delta = 0
            for s in states:
                if s == states[-1]:
                    continue
                    
                max_value = float('-inf')
                best_action = None
                
                for a in transitions[s]:
                    expected_value = 0.0
                    for prob, next_s in transitions[s][a]:
                        expected_value += prob * (rewards[s] + gamma * V[next_s])
                    
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = a
                
                delta = max(delta, abs(V[s] - max_value))
                V[s] = max_value
                policy[s] = best_action
            
            if delta < epsilon:
                break
                
        return policy

    def extract_path_from_policy(self, start, goal, policy, max_steps=500):
        """Follow policy to extract a path from start to goal"""
        path = [start]
        current = start
        steps = 0
        
        while current != goal and steps < max_steps:
            if current not in policy:
                self.get_logger().warn("Policy undefined for current state")
                break
                
            action = policy[current]
            next_state = (current[0] + action[0], current[1] + action[1])
            
            if (next_state[0] < 0 or next_state[0] >= self.width or 
                next_state[1] < 0 or next_state[1] >= self.height or
                self.dynamic_grid[next_state[1]][next_state[0]] == 1):
                self.get_logger().warn("Invalid state in policy - aborting")
                break
                
            path.append(next_state)
            current = next_state
            steps += 1
            
        if current != goal:
            self.get_logger().warn("Failed to reach goal within step limit")
            
        return path

    def get_remaining_path(self, return_point):
        """Get remaining original path from return point"""
        idx = self.original_path.index(return_point)
        return self.original_path[idx:]

    def publish_path(self, path):
        """Publish the replanned path as a Path message"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for x, y in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x * self.resolution + self.origin[0]
            pose.pose.position.y = y * self.resolution + self.origin[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)
        self.get_logger().info("Published replanned path")

    def save_path_to_file(self, path, filename):
        """Save path to CSV file"""
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['x', 'y'])
                for point in path:
                    writer.writerow([point[0], point[1]])
            self.get_logger().info(f"Path saved to {filename}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save path: {str(e)}")

    def visualize_path(self, replanned_path=None):
        """Visualize the paths"""
        plt.figure(figsize=(12, 10))
        plt.imshow(self.dynamic_grid, cmap='gray', origin='lower', alpha=0.7)
        
        if self.original_path:
            orig_x, orig_y = zip(*self.original_path)
            plt.plot(orig_x, orig_y, 'b-', linewidth=2, label='Original Path')
            plt.plot(orig_x[0], orig_y[0], 'go', markersize=10, label='Start')
            plt.plot(orig_x[-1], orig_y[-1], 'ro', markersize=10, label='Goal')
        
        if replanned_path:
            replan_x, replan_y = zip(*replanned_path)
            plt.plot(replan_x, replan_y, 'g--', linewidth=2, label='Replanned Path')
        
        if self.current_position:
            plt.plot(self.current_position[0], self.current_position[1], 'mo', markersize=10, label='Current Position')
        
        plt.legend()
        plt.title('Path Planning Visualization')
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        plt.grid(True)
        plt.axis('equal')
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_file = os.path.join(self.output_dir, f'path_visualization_{timestamp}.png')
        plt.savefig(plot_file, dpi=300)
        self.get_logger().info(f"Path visualization saved to {plot_file}")
        
        if 'DISPLAY' in os.environ:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            self.get_logger().warn("No display available - skipping interactive plot")

    @staticmethod
    def euler_from_quaternion(quat):
        """Convert quaternion to Euler angle (yaw only)"""
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
    node = MDPReplanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()