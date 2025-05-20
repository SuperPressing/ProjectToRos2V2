# Copyright 2016 Open Source Robotics Foundation, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from collections import deque
import math


def find_connected_groups(data, width, height):
    """Находит все связанные группы ячеек со значением 100 (занято)"""
    visited = set()
    groups = []

    for y in range(height):
        for x in range(width):
            idx = x + y * width
            if data[idx] == 100 and (x, y) not in visited:
                group = []
                connect_cells(x, y, data, width, height, group, visited)
                if group:
                    groups.append(group)

    return groups


def connect_cells(x, y, data, width, height, group, visited):
    """Рекурсивный поиск соседних ячеек методом BFS"""
    queue = deque()

    index = x + y * width
    if data[index] != 100 or (x, y) in visited:
        return

    visited.add((x, y))
    group.append((x, y))
    queue.append((x, y))

    while queue:
        cx, cy = queue.popleft()

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = cx + dx, cy + dy
                n_index = nx + ny * width

                if 0 <= nx < width and 0 <= ny < height:
                    if data[n_index] == 100 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        group.append((nx, ny))
                        queue.append((nx, ny))


class MapEditor(Node):

    def __init__(self):
        super().__init__('map_editor_node')

        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            '/map_modified',
            10
        )

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        data = list(msg.data)

        # Поиск групп занятых ячеек
        groups = find_connected_groups(data, width, height)

        if not groups:
            return

        # Вычисляем центры групп
        centers = []
        for group in groups:
            x_coords = [g[0] for g in group]
            y_coords = [g[1] for g in group]
            center_x = sum(x_coords) // len(group)
            center_y = sum(y_coords) // len(group)
            centers.append((center_x, center_y))

        # Минимальное расстояние между группами (в метрах)
        min_distance_meters = 1.5
        min_distance_cells = int(min_distance_meters / resolution)

        modified_data = list(data)

        # Объединяем близкие группы
        for i, (cx1, cy1) in enumerate(centers):
            for j, (cx2, cy2) in enumerate(centers[i+1:], start=i+1):
                dx = cx2 - cx1
                dy = cy2 - cy1
                dist = math.hypot(dx, dy)

                if dist < min_distance_cells:
                    points = self.bresenham(cx1, cy1, cx2, cy2)
                    for x, y in points:
                        idx = x + y * width
                        if 0 <= idx < len(modified_data):
                            modified_data[idx] = 100

        # Публикация изменённой карты
        new_msg = OccupancyGrid()
        new_msg.header = msg.header
        new_msg.info = msg.info
        new_msg.data = tuple(modified_data)
        self.map_publisher.publish(new_msg)

    def bresenham(self, x0, y0, x1, y1):
        """Алгоритм Брезенхэма для рисования линии между точками"""
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        x, y = x0, y0

        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

        return points


def main(args=None):
    rclpy.init(args=args)
    node = MapEditor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()