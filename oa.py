#!/usr/bin/env python3

import rclpy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy

class RobotNavigationController:
    def __init__(self):
        self.ros_node = None
        self.cmd_publisher = None
        self.zone_readings = {
            'left_side': 0,
            'right_side': 0,
            'front_center': 0,
        }
        self.motion_cmd = None
        # Fuzzy set boundaries
        self.NEAR_THRESHOLD = 1.0
        self.MEDIUM_THRESHOLD = 3.0
        self.FAR_THRESHOLD = 10.0

    # Fuzzy membership functions
    def left_shoulder(self, x, a, b):
        if x <= a:
            return 1.0
        elif x >= b:
            return 0.0
        else:
            return (b - x) / (b - a)

    def right_shoulder(self, x, a, b):
        if x <= a:
            return 0.0
        elif x >= b:
            return 1.0
        else:
            return (x - a) / (b - a)

    def triangle(self, x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        else:
            return 0.0

    def membership_function(self, dist_value):
        # Near: high near 0, decreasing towards NEAR_THRESHOLD
        mu_near = self.left_shoulder(
            dist_value,
            a=0.0,
            b=self.NEAR_THRESHOLD
        )
        # Medium: triangular, centered between NEAR and MEDIUM thresholds
        center_med = (self.NEAR_THRESHOLD + self.MEDIUM_THRESHOLD) / 2.0
        mu_medium = self.triangle(
            dist_value,
            a=self.NEAR_THRESHOLD * 0.5,
            b=center_med,
            c=self.MEDIUM_THRESHOLD * 1.2 if self.MEDIUM_THRESHOLD * 1.2 < self.FAR_THRESHOLD else self.FAR_THRESHOLD
        )
        # Far: low until MEDIUM_THRESHOLD, then increasing towards FAR_THRESHOLD
        mu_far = self.right_shoulder(
            dist_value,
            a=self.MEDIUM_THRESHOLD,
            b=self.FAR_THRESHOLD
        )
        memberships = {
            'Near': mu_near,
            'Medium': mu_medium,
            'Far': mu_far
        }
        best_label = max(memberships, key=memberships.get)
        return best_label


    def fuzzy_inference_engine(self, left_input, front_input, right_input):
        left_fuzzy = self.membership_function(left_input)
        front_fuzzy = self.membership_function(front_input)
        right_fuzzy = self.membership_function(right_input)

        # 27 fuzzy rules in if-then format
        if left_fuzzy == 'Near' and front_fuzzy == 'Near' and right_fuzzy == 'Near':
            action = 'Rotate'
        elif left_fuzzy == 'Near' and front_fuzzy == 'Near' and right_fuzzy == 'Medium':
            action = 'Turn Right'
        elif left_fuzzy == 'Near' and front_fuzzy == 'Near' and right_fuzzy == 'Far':
            action = 'Turn Right'
        elif left_fuzzy == 'Near' and front_fuzzy == 'Medium' and right_fuzzy == 'Near':
            action = 'Turn Left'
        elif left_fuzzy == 'Near' and front_fuzzy == 'Medium' and right_fuzzy == 'Medium':
            action = 'Go Forward'
        elif left_fuzzy == 'Near' and front_fuzzy == 'Medium' and right_fuzzy == 'Far':
            action = 'Go Forward'
        elif left_fuzzy == 'Near' and front_fuzzy == 'Far' and right_fuzzy == 'Near':
            action = 'Turn Left'
        elif left_fuzzy == 'Near' and front_fuzzy == 'Far' and right_fuzzy == 'Medium':
            action = 'Turn Right'
        elif left_fuzzy == 'Near' and front_fuzzy == 'Far' and right_fuzzy == 'Far':
            action = 'Go Forward'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Near' and right_fuzzy == 'Near':
            action = 'Turn Right'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Near' and right_fuzzy == 'Medium':
            action = 'Turn Right'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Near' and right_fuzzy == 'Far':
            action = 'Turn Right'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Medium' and right_fuzzy == 'Near':
            action = 'Turn Left'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Medium' and right_fuzzy == 'Medium':
            action = 'Go Forward'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Medium' and right_fuzzy == 'Far':
            action = 'Go Forward'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Far' and right_fuzzy == 'Near':
            action = 'Turn Left'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Far' and right_fuzzy == 'Medium':
            action = 'Turn Right'
        elif left_fuzzy == 'Medium' and front_fuzzy == 'Far' and right_fuzzy == 'Far':
            action = 'Go Forward'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Near' and right_fuzzy == 'Near':
            action = 'Turn Left'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Near' and right_fuzzy == 'Medium':
            action = 'Turn Right'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Near' and right_fuzzy == 'Far':
            action = 'Turn Right'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Medium' and right_fuzzy == 'Near':
            action = 'Turn Left'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Medium' and right_fuzzy == 'Medium':
            action = 'Go Forward'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Medium' and right_fuzzy == 'Far':
            action = 'Go Forward'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Far' and right_fuzzy == 'Near':
            action = 'Turn Left'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Far' and right_fuzzy == 'Medium':
            action = 'Turn Right'
        elif left_fuzzy == 'Far' and front_fuzzy == 'Far' and right_fuzzy == 'Far':
            action = 'Go Forward'
        else:
            action = 'Stop'
        print(f"Fuzzy values - Left: {left_fuzzy}, Front: {front_fuzzy}, Right: {right_fuzzy}")
        print(f"Inference result: {action}")
        return action

    def defuzzification_crisp_output(self):
        #center of set defuzzification method
        zones = self.zone_readings
        left_distance = zones['left_side']
        front_distance = zones['front_center']
        right_distance = zones['right_side']
        fuzzy_action = self.fuzzy_inference_engine(left_distance, front_distance, right_distance)
        # defuzzification: Map linguistic action to crisp velocity values
        velocity_msg = Twist()
        # sharp avoidance when front is Near
        if fuzzy_action == 'Stop':
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
        elif fuzzy_action == 'Go Forward':
            velocity_msg.linear.x = 0.25
            velocity_msg.angular.z = 0.0
        elif fuzzy_action == 'Turn Left':
            # less forward speed when turning near obstacles
            if front_distance <= self.NEAR_THRESHOLD:
                velocity_msg.linear.x = 0.0
                velocity_msg.angular.z = 1.5
            else:
                velocity_msg.linear.x = 0.15
                velocity_msg.angular.z = 1.2
        elif fuzzy_action == 'Turn Right':
            # less forward speed when turning near obstacles
            if front_distance <= self.NEAR_THRESHOLD:
                velocity_msg.linear.x = 0.0
                velocity_msg.angular.z = -1.5
            else:
                velocity_msg.linear.x = 0.15
                velocity_msg.angular.z = -1.2
        elif fuzzy_action == 'Rotate':
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 1.5
        return velocity_msg

    def laser_callback_handler(self, laser_msg):
        print("Scan data points: ", len(laser_msg.ranges))
        self.zone_readings = {
            'front_center': self.get_closest_reading(laser_msg.ranges[120:122]),
            'right_side': self.get_closest_reading(laser_msg.ranges[21:23]),
            'left_side': self.get_closest_reading(laser_msg.ranges[219:221])
        }
        self.motion_cmd = self.defuzzification_crisp_output()

    def get_closest_reading(self, range_array):
        filtered = filter(lambda x: x > 0.0, range_array)
        return min(min(filtered, default=10), 10)

    def emergency_stop(self):
        stop_msg = Twist()
        stop_msg.angular.z = 0.0
        stop_msg.linear.x = 0.0
        return stop_msg

    def periodic_publish(self):
        if self.motion_cmd is not None:
            self.cmd_publisher.publish(self.motion_cmd)

    def run(self):
        rclpy.init()
        self.ros_node = rclpy.create_node('fuzzy_navigation_node')
        qos_settings = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.cmd_publisher = self.ros_node.create_publisher(Twist, '/cmd_vel', 10)
        laser_sub = self.ros_node.create_subscription(
            LaserScan, '/scan', self.laser_callback_handler, qos_settings
        )
        update_rate = 0.2
        timer_obj = self.ros_node.create_timer(update_rate, self.periodic_publish)
        try:
            rclpy.spin(self.ros_node)
        except Exception as error:
            print(error)
            self.emergency_stop()
        finally:
            self.ros_node.destroy_timer(timer_obj)
            self.ros_node.destroy_node()

if __name__ == '__main__':
    navigator = RobotNavigationController()
    navigator.run()
