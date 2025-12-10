#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy

class WallFollower:
    def __init__(self, target_distance):
        self.target_distance=target_distance
        self.prev_error=0.0
        self.integral=0.0

    def calculate(self,current_distance,kp=1.0,ki=0.0,kd=0.3):
        """
        current_distance: measured distance to the wall on the right
        returns: control signal (angular component)
        """
        error=self.target_distance-current_distance
        self.integral=self.integral+error
        derivative=error-self.prev_error
        self.prev_error=error
        u=kp*error+ki*self.integral+kd*derivative
        return u

node_ = None
pub_ = None
twist_cmd_ = None
regions_ = {
    'right': 10.0,
    'fright': 10.0,
    'front1': 10.0,
    'front2': 10.0,
    'fleft': 10.0,
    'left': 10.0,
}
wall_follower_=WallFollower(target_distance=0.4)

def find_nearest(readings):
    filtered=[x for x in readings if x>0.01]
    if not filtered:
        return 10.0
    return min(filtered)


def compute_twist():
    msg=Twist()
    right=regions_['right']
    fright=regions_['fright']
    front=min(regions_['front1'], regions_['front2'])
    fleft=regions_['fleft']
    left=regions_['left']
    # too close in front -> turn left
    front_threshold=0.35
    if front<front_threshold or fright<front_threshold or fleft<front_threshold:
        # Turn left in place
        msg.linear.x=0.0
        msg.angular.z=0.6
        node_.get_logger().info("Obstacle ahead -> turning left")
        return msg
    # no wall on the right -> search for right wall
    no_wall_threshold=1.0  #right > 1m-->assume no wall yet
    if right>no_wall_threshold:
        # Slowly move forward and slightly turn right to find a wall
        msg.linear.x=0.12
        msg.angular.z=-0.25   # negative-->turn right
        node_.get_logger().info("No wall on the right -> searching (forward + right turn)")
        return msg
    # wall on the right -> follow PID
    control=wall_follower_.calculate(current_distance=right)
    # avoiding spins
    max_turn=1.0
    if control>max_turn:
        control=max_turn
    elif control<-max_turn:
        control=-max_turn
    msg.angular.z=control
    # Adjust forward speed depending on how hard we are turning
    if abs(control)<0.2:
        msg.linear.x=0.18   # straight-->faster
    elif abs(control)<0.6:
        msg.linear.x=0.12   # medium-->turn
    else:
        msg.linear.x=0.06   # sharp-->slow
    node_.get_logger().info(
        f"Following wall: right={right:.2f}, front={front:.2f}, "
        f"v={msg.linear.x:.2f}, w={msg.angular.z:.2f}"
    )
    return msg

def laser_callback(msg: LaserScan):
    global regions_,twist_cmd_
    ranges=msg.ranges
    regions_={
        'front1': find_nearest(ranges[0:5]),
        'front2': find_nearest(ranges[355:360]),
        'right':  find_nearest(ranges[265:275]),
        'fright': find_nearest(ranges[310:320]),
        'fleft':  find_nearest(ranges[40:50]),
        'left':   find_nearest(ranges[85:95]),
    }
    twist_cmd_=compute_twist()

def timer_callback():
    global twist_cmd_
    if twist_cmd_ is None:
        msg=Twist()
    else:
        msg=twist_cmd_
    pub_.publish(msg)

def main():
    global node_, pub_
    rclpy.init()
    node_=rclpy.create_node('wall_follower_pid')
    qos=QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    pub_=node_.create_publisher(Twist, '/cmd_vel', 10)
    node_.create_subscription(LaserScan, '/scan', laser_callback, qos)
    timer=node_.create_timer(0.1, timer_callback)
    node_.get_logger().info("Wall follower PID node started.")
    try:
        rclpy.spin(node_)
    except KeyboardInterrupt:
        node_.get_logger().info("Shutting down wall follower PID node.")
    finally:
        stop_msg=Twist()
        pub_.publish(stop_msg)
        node_.destroy_timer(timer)
        node_.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
