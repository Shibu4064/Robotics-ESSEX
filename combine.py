#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy


class OAFuzzyLogicController:
    def __init__(self):
        self.speed_sets = {
            'Slow': [0.0, 0.08, 0.15],
            'Medium': [0.10, 0.18, 0.25],
            'Fast': [0.20, 0.28, 0.35]
        }
        self.direction_sets = {
            'TurnLeft':   [-1.0, -0.7, -0.4],
            'SlightLeft': [-0.5, -0.25, 0.0],
            'Straight':   [-0.1, 0.0, 0.1],
            'SlightRight':[0.0, 0.25, 0.5],
            'TurnRight':  [0.4, 0.7, 1.0]
        }
        self.distance_params = {
            'near_range':   [0.0, 0.2, 0.6],
            'medium_range': [0.4, 0.9, 1.4],
            'far_range':    [1.2, 2.0, 3.5]
        }
        # 27 Fuzzy Rules: (FrontLeft, Front, FrontRight, Speed, Direction)
        self.rules = [
            # Row 1: FL = Near
            ('Near', 'Near', 'Near',   'Slow',   'TurnRight'),
            ('Near', 'Near', 'Medium', 'Slow',   'TurnRight'),
            ('Near', 'Near', 'Far',    'Slow',   'TurnRight'),
            # Row 2: FL = Near, F = Medium
            ('Near', 'Medium', 'Near',   'Slow',   'TurnRight'),
            ('Near', 'Medium', 'Medium', 'Medium', 'SlightRight'),
            ('Near', 'Medium', 'Far',    'Medium', 'SlightRight'),
            # Row 3: FL = Near, F = Far
            ('Near', 'Far', 'Near',   'Medium', 'SlightRight'),
            ('Near', 'Far', 'Medium', 'Medium', 'SlightRight'),
            ('Near', 'Far', 'Far',    'Fast',   'SlightRight'),
            # Row 4: FL = Medium
            ('Medium', 'Near', 'Near',   'Slow',   'TurnLeft'),
            ('Medium', 'Near', 'Medium', 'Slow',   'TurnLeft'),
            ('Medium', 'Near', 'Far',    'Slow',   'TurnRight'),
            # Row 5: FL = Medium, F = Medium
            ('Medium', 'Medium', 'Near',   'Medium', 'SlightLeft'),
            ('Medium', 'Medium', 'Medium', 'Medium', 'Straight'),
            ('Medium', 'Medium', 'Far',    'Medium', 'SlightRight'),
            # Row 6: FL = Medium, F = Far
            ('Medium', 'Far', 'Near',   'Medium', 'SlightLeft'),
            ('Medium', 'Far', 'Medium', 'Fast',   'Straight'),
            ('Medium', 'Far', 'Far',    'Fast',   'Straight'),
            # Row 7: FL = Far
            ('Far', 'Near', 'Near',   'Slow',   'TurnLeft'),
            ('Far', 'Near', 'Medium', 'Slow',   'TurnLeft'),
            ('Far', 'Near', 'Far',    'Slow',   'TurnLeft'),
            # Row 8: FL = Far, F = Medium
            ('Far', 'Medium', 'Near',   'Medium', 'SlightLeft'),
            ('Far', 'Medium', 'Medium', 'Medium', 'SlightLeft'),
            ('Far', 'Medium', 'Far',    'Medium', 'Straight'),
            # Row 9: FL = Far, F = Far
            ('Far', 'Far', 'Near',   'Fast', 'SlightLeft'),
            ('Far', 'Far', 'Medium', 'Fast', 'Straight'),
            ('Far', 'Far', 'Far',    'Fast', 'Straight'),
        ]

    # fuzzification
    def fuzzify_distance(self, distance):
        near_params = self.distance_params['near_range']
        medium_params = self.distance_params['medium_range']
        far_params = self.distance_params['far_range']
        memberships = {'Near': 0.0, 'Medium': 0.0, 'Far': 0.0}

        # Near membership
        if distance <= near_params[1]:
            memberships['Near'] = 1.0
        elif distance < near_params[2]:
            memberships['Near'] = (near_params[2] - distance) / (near_params[2] - near_params[1])
        else:
            memberships['Near'] = 0.0
        # Medium membership
        if distance <= medium_params[0]:
            memberships['Medium'] = 0.0
        elif distance < medium_params[1]:
            memberships['Medium'] = (distance - medium_params[0]) / (medium_params[1] - medium_params[0])
        elif distance == medium_params[1]:
            memberships['Medium'] = 1.0
        elif distance < medium_params[2]:
            memberships['Medium'] = (medium_params[2] - distance) / (medium_params[2] - medium_params[1])
        else:
            memberships['Medium'] = 0.0
        # Far membership
        if distance <= far_params[0]:
            memberships['Far'] = 0.0
        elif distance < far_params[1]:
            memberships['Far'] = (distance - far_params[0]) / (far_params[1] - far_params[0])
        else:
            memberships['Far'] = 1.0
        return memberships

    # rule evaluation
    def evaluate_rules(self, fl_dist, f_dist, fr_dist):
        fl_membership = self.fuzzify_distance(fl_dist)
        f_membership = self.fuzzify_distance(f_dist)
        fr_membership = self.fuzzify_distance(fr_dist)
        activated = []
        for idx, (fl_l, f_l, fr_l, speed_l, dir_l) in enumerate(self.rules):
            strength = min(fl_membership[fl_l], f_membership[f_l], fr_membership[fr_l])
            if strength <= 0.0:
                continue
            activated.append({
                'strength': strength,
                'speed': speed_l,
                'direction': dir_l,
                'rule_num': idx,
            })
        if not activated:
            activated.append({
                'strength': 0.5,
                'speed': 'Medium',
                'direction': 'Straight',
                'rule_num': -1,
            })
        return activated

    def defuzzify(self, activated_rules):
        #Weighted-average defuzzification
        speed_num = 0.0
        speed_den = 0.0
        dir_num = 0.0
        dir_den = 0.0
        for rule in activated_rules:
            w = rule['strength']
            s_c = self.speed_sets[rule['speed']][1]
            d_c = self.direction_sets[rule['direction']][1]
            speed_num += w * s_c
            speed_den += w
            dir_num += w * d_c
            dir_den += w
        if speed_den == 0.0:
            crisp_speed = 0.0
        else:
            crisp_speed = speed_num / speed_den
        if dir_den == 0.0:
            crisp_dir = 0.0
        else:
            crisp_dir = dir_num / dir_den
        return crisp_speed, crisp_dir

    def get_command(self, fl_dist, f_dist, fr_dist):
        activated = self.evaluate_rules(fl_dist, f_dist, fr_dist)
        return self.defuzzify(activated)


class RightEdgeOACombined(Node):
    def __init__(self):
        super().__init__('right_edge_oa_combined')
        self.oa_fuzzy = OAFuzzyLogicController()
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd = Twist()
        self.create_timer(0.1, self.timer_callback)

        #fuzzy sets for right-edge follower
        # Front distance
        self.front_sets = {
            'Near':   [0.0, 0.0, 0.6],
            'Medium': [0.3, 0.8, 1.3],
            'Far':    [1.0, 1.7, 2.5],
        }
        # Right distance
        self.right_sets = {
            'TooClose':  [0.0, 0.3, 0.5],
            'JustRight': [0.4, 0.6, 0.8],
            'TooFar':    [0.7, 1.2, 2.5],
        }
        # Output centroids (COSDF)
        self.edge_speed_centres = {
            'Low':    0.04,
            'Medium': 0.10,
            'High':   0.18,
        }
        # angular.z, (+)left, (-)right
        self.edge_turn_centres = {
            'StrongLeft':   0.9,
            'Left':         0.5,
            'Straight':     0.0,
            'Right':       -0.5,
            'StrongRight': -0.9,
        }
        # Fuzzy rule base: (Front, Right) -> (Speed, Turn)
        self.edge_rules = [
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
        self.get_logger().info("Combined Right-edge + Obstacle-avoidance fuzzy controller initialised.")


    def find_nearest(self, readings):
        filtered = [r for r in readings if 0.05 < r < 10.0]
        return min(filtered) if filtered else 3.5

    def sector_min(self, msg: LaserScan, center_deg: float, half_width_deg: float):
        n = len(msg.ranges)
        a_min = msg.angle_min
        a_inc = msg.angle_increment
        r_max = msg.range_max if msg.range_max > 0 else 3.0
        center = math.radians(center_deg)
        half = math.radians(half_width_deg)
        start_ang = center - half
        end_ang = center + half
        idx_start = max(0, int((start_ang - a_min) / a_inc))
        idx_end = min(n - 1, int((end_ang - a_min) / a_inc))
        values = []
        for i in range(idx_start, idx_end + 1):
            r = msg.ranges[i]
            if msg.range_min < r < msg.range_max:
                values.append(r)
        if not values:
            return r_max
        return min(values)

    # edge following fuzzy helpers
    def tri(self, x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if a < x < b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)

    def fuzzify(self, x, sets_dict):
        return {name: self.tri(x, *params) for name, params in sets_dict.items()}

    def edge_follow_command(self, front, right):
        F = self.fuzzify(front, self.front_sets)
        R = self.fuzzify(right, self.right_sets)
        #max–min composition
        agg_speed = {k: 0.0 for k in self.edge_speed_centres.keys()}
        agg_turn = {k: 0.0 for k in self.edge_turn_centres.keys()}
        for f_label, r_label, s_label, t_label in self.edge_rules:
            strength = min(F[f_label], R[r_label])
            if strength <= 0.0:
                continue
            agg_speed[s_label] = max(agg_speed[s_label], strength)
            agg_turn[t_label] = max(agg_turn[t_label], strength)

        def defuzz(agg, centres, default):
            den = sum(agg.values())
            if den == 0.0:
                return default
            num = sum(agg[label] * centres[label] for label in centres.keys())
            return num / den
        v = defuzz(agg_speed, self.edge_speed_centres, default=0.06)
        w = defuzz(agg_turn, self.edge_turn_centres, default=0.0)
        return v, w


    def laser_callback(self, msg: LaserScan):
        front_readings = list(msg.ranges[0:20]) + list(msg.ranges[340:360])
        f_dist = self.find_nearest(front_readings)
        fl_dist = self.find_nearest(msg.ranges[20:70])
        fr_dist = self.find_nearest(msg.ranges[290:340])
        front_sector = self.sector_min(msg, center_deg=0.0,   half_width_deg=15.0)
        right_sector = self.sector_min(msg, center_deg=-90.0, half_width_deg=10.0)
        #Primary=OA
        oa_speed, oa_turn = self.oa_fuzzy.get_command(fl_dist, f_dist, fr_dist)
        #Secondary=right-edge
        edge_speed, edge_turn = self.edge_follow_command(front_sector, right_sector)

        # If something is close in front -> OA dominates
        # Else -> follow right edge, but still follows OA.
        if f_dist < 0.40 or min(fl_dist, fr_dist) < 0.35:
            # Danger zone: just avoid collision.
            speed = oa_speed
            turn = oa_turn
            mode = "OA-EMERGENCY"
        else:
            # use right-edge follower as main behaviour.
            speed = min(edge_speed, oa_speed)
            turn = 0.7 * edge_turn + 0.3 * oa_turn
            mode = "RIGHT-FOLLOW"
        # safety & tuning
        speed = max(-0.1, min(0.30, float(speed)))
        turn = max(-1.0, min(1.0, float(turn)))
        # minimum forward speed-->when clear in front
        if f_dist > 0.8 and speed > 0.0 and speed < 0.06:
            speed = 0.10
        # Hard emergency override (never hit walls head-on)
        if f_dist < 0.20:
            # really close-->turn away from closer side
            self.get_logger().warn(
                f" CRITICAL front={f_dist:.2f} (fl={fl_dist:.2f}, fr={fr_dist:.2f}) -> backing up!"
            )
            speed = -0.10
            if fl_dist < fr_dist:
                turn = 0.9   # rotate strongly right
            else:
                turn = -0.9
            mode = "BACKUP"
        elif f_dist < 0.35:
            # Close but not touching: slow and turn hard away
            self.get_logger().warn(
                f" DANGER front={f_dist:.2f} (fl={fl_dist:.2f}, fr={fr_dist:.2f}) -> hard turn!"
            )
            speed = max(0.04, min(speed, 0.10))
            if turn > 0:
                turn = max(0.7, turn)
            elif turn < 0:
                turn = min(-0.7, turn)
            else:
                if fl_dist < fr_dist:
                    turn = 0.8
                else:
                    turn = -0.8
            mode = "OA-HARD-TURN"

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = turn
        self.cmd = cmd
        self.get_logger().info(
            f"[{mode}] FL={fl_dist:.2f} F={f_dist:.2f} FR={fr_dist:.2f} "
            f"| R={right_sector:.2f} -> v={speed:.2f}, w={turn:.2f}"
        )
    def timer_callback(self):
        self.pub.publish(self.cmd)

def main():
    rclpy.init()
    node = RightEdgeOACombined()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
