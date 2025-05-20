
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')

        # Подписка на /topic
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        
        # Подписка на /map
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)

        # Публикатор для изменённой карты
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map_modified', 10)

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

    def map_callback(self, msg):
        self.get_logger().info('Получена карта /map')

        # Пример изменения первой ячейки
        modified_data = list(msg.data)
        if len(modified_data) > 0:
            modified_data[0] = 100  # помечаем как занятую область

        # Создаём новое сообщение
        new_msg = OccupancyGrid()
        new_msg.header = msg.header
        new_msg.info = msg.info
        new_msg.data = tuple(modified_data)

        # Публикуем новую карту
        self.map_publisher.publish(new_msg)
        self.get_logger().info('Карта изменена и опубликована в /map_modified')


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()