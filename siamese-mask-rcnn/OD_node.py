import rclpy
from rclpy.node import Node

from std_msgs.msg import String
#from package_buddy.msg import Movement
import sys
import os

cwd=os.getcwd()
ROOT_DIR = cwd+'/src/package_buddy/'
sys.path.append(ROOT_DIR)
ROOT_DIR = ROOT_DIR+'package_buddy/'
sys.path.append(ROOT_DIR)
import ODdemo_OD

class OD(Node):


    def __init__(self):
        super().__init__('OD_node')
        self.publisher_ = self.create_publisher(String, 'OD_topic', 10)
        self.subscription = self.create_subscription(
            String,
            'status',
            self.listener_callback,
            10)
        self.subscription


    def listener_callback(self, msg):
        msg = str(msg.data).split(' ')
        message=OD_camera()
        if not (float(msg[0]) == 0 and float(msg[1]) == 0):
            msg2 = String()
            msg2.data = message
            self.publisher_.publish(msg2)
            self.get_logger().info(msg.data)
            



def main(args=None):
    rclpy.init(args=args)
    OD_object = OD()

    rclpy.spin(OD_object)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    OD_object.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
