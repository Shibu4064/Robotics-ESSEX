#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy


class RightEdgeFollower(Node):
    def __init__(self):
        super().__init__('right_edge_fuzzy')
        qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub=self.create_subscription(
            LaserScan, '/scan', self.laser_callback, qos
        )
        self.pub=self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd=Twist()
        self.create_timer(0.1,self.timer_callback)
        # Front distance (m)
        self.front_sets={
            'Near':   [0.0, 0.0, 0.6],   # very close
            'Medium': [0.3, 0.8, 1.3],   # mid-range
            'Far':    [1.0, 1.7, 2.5],   # far
        }
        # Right distance (m)
        self.right_sets={
            'TooClose':  [0.0, 0.3, 0.5],
            'JustRight': [0.4, 0.6, 0.8],
            'TooFar':    [0.7, 1.2, 2.5],
        }
        # Output centroids (centre-of-sets)
        self.speed_centres={
            'Low':    0.04,
            'Medium': 0.10,
            'High':   0.18,
        }
        # angular.z, + = left, - = right
        self.turn_centres={
            'StrongLeft':  0.9,
            'Left':        0.5,
            'Straight':    0.0,
            'Right':      -0.5,
            'StrongRight': -0.9,
        }
        # Fuzzy rule base: (Front, Right) -> (Speed, Turn)
        self.rules=[
            # FRONT NEAR -> strongly avoid collision, regardless of right
            ('Near', 'TooClose',  'Low', 'StrongLeft'),
            ('Near', 'JustRight', 'Low', 'StrongLeft'),
            ('Near', 'TooFar',    'Low', 'StrongLeft'),
            # FRONT MEDIUM
            ('Medium', 'TooClose',  'Low',    'Left'),
            ('Medium', 'JustRight', 'Medium', 'Straight'),
            ('Medium', 'TooFar',    'Medium', 'Right'),
            # FRONT FAR
            ('Far', 'TooClose',   'Medium', 'Left'),
            ('Far', 'JustRight',  'High',   'Straight'),
            ('Far', 'TooFar',     'High',   'Right'),
        ]
    # ---------- LIDAR sector helpers ----------
    def sector_min(self, msg: LaserScan, center_deg: float, half_width_deg: float):
        n=len(msg.ranges)
        a_min=msg.angle_min
        a_inc=msg.angle_increment
        r_max=msg.range_max if msg.range_max>0 else 3.0
        center=math.radians(center_deg)
        half=math.radians(half_width_deg)
        start_angle=center-half
        end_angle=center+half
        start_idx=int((start_angle-a_min)/a_inc)
        end_idx=int((end_angle-a_min)/a_inc)
        start_idx=max(0,min(n-1,start_idx))
        end_idx=max(0,min(n-1,end_idx))
        if end_idx<start_idx:
            start_idx,end_idx=end_idx,start_idx
        vals=[]
        for i in range(start_idx,end_idx + 1):
            r = msg.ranges[i]
            if r>0.01 and not math.isinf(r) and not math.isnan(r):
                vals.append(r)
        if not vals:
            return r_max
        d=min(vals)
        return d if d<r_max else r_max

    # triangular membership function
    @staticmethod
    def tri(x,a,b,c):
        if x<=a or x>=c:
            return 0.0
        if x==b:
            return 1.0
        if a<x<b:
            return (x-a)/(b-a)
        return (c-x)/(c-b)

    def fuzzify(self, x, sets_dict):
        """Return dict {label: mu} for value x in given triangular sets."""
        return {name: self.tri(x, *params) for name, params in sets_dict.items()}

    # fuzzy controller
    def laser_callback(self, msg: LaserScan):
        #get crisp inputs
        front=self.sector_min(msg,center_deg=0.0,half_width_deg=15.0)
        right=self.sector_min(msg,center_deg=-90.0,half_width_deg=10.0)
        #fuzzify
        F=self.fuzzify(front,self.front_sets)
        R=self.fuzzify(right,self.right_sets)
        #rule evaluation (max–min)
        agg_speed={k: 0.0 for k in self.speed_centres.keys()}
        agg_turn={k: 0.0 for k in self.turn_centres.keys()}
        for f_label,r_label,s_label,t_label in self.rules:
            strength=min(F[f_label],R[r_label])
            if strength<=0.0:
                continue
            agg_speed[s_label]=max(agg_speed[s_label], strength)
            agg_turn[t_label]=max(agg_turn[t_label], strength)
        #centre-of-sets defuzzification
        def defuzz(agg,centres,default):
            den=sum(agg.values())
            if den==0.0:
                return default
            num=sum(agg[label]*centres[label] for label in centres.keys())
            return num/den
        v=defuzz(agg_speed,self.speed_centres,default=0.05)
        w=defuzz(agg_turn,self.turn_centres,default=0.0)
        # extra hard safety: if something is extremely close in front
        if front<0.20:
            v=0.0
            w=0.9
        w=max(-1.0,min(1.0,w))
        cmd=Twist()
        cmd.linear.x=v
        cmd.angular.z=w
        self.cmd=cmd
        self.get_logger().info(
            f"front={front:.2f}, right={right:.2f} -> v={v:.2f}, w={w:.2f}"
        )

    def timer_callback(self):
        self.pub.publish(self.cmd)

def main():
    rclpy.init()
    node=RightEdgeFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
