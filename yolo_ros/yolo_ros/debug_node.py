#!/usr/bin/env python3

import cv2
import json
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class DebugNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("debug_node")
        
        self._class_to_color = {}
        self.cv_bridge = CvBridge()
        
        # Counting attributes
        self._class_counts = {}
        self._counted_ids = set()
        
        # Line detection parameters
        self.detection_line = [20, 350, 940, 350]  # [x1, y1, x2, y2]
        self.line_crossed = False
        
        # params
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.image_qos_profile = QoSProfile(
            reliability=self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Publishers
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._bb_markers_pub = self.create_publisher(MarkerArray, "dgb_bb_markers", 10)
        self._kp_markers_pub = self.create_publisher(MarkerArray, "dgb_kp_markers", 10)
        self._counts_pub = self.create_publisher(
            String, 
            "/vista/dashcam/debug_counter", 
            10
        )

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # Subscribers
        self.image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=self.image_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.detections_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._dbg_pub)
        self.destroy_publisher(self._bb_markers_pub)
        self.destroy_publisher(self._kp_markers_pub)
        self.destroy_publisher(self._counts_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def draw_box(
        self, cv_image: np.ndarray, detection: Detection, color: Tuple[int]
    ) -> np.ndarray:
        # get detection info
        class_name = detection.class_name
        score = detection.score
        box_msg: BoundingBox2D = detection.bbox
        track_id = detection.id

        min_pt = (
            round(box_msg.center.position.x - box_msg.size.x / 2.0),
            round(box_msg.center.position.y - box_msg.size.y / 2.0),
        )
        max_pt = (
            round(box_msg.center.position.x + box_msg.size.x / 2.0),
            round(box_msg.center.position.y + box_msg.size.y / 2.0),
        )

        # define the four corners of the rectangle
        rect_pts = np.array([
            [min_pt[0], min_pt[1]],
            [max_pt[0], min_pt[1]],
            [max_pt[0], max_pt[1]],
            [min_pt[0], max_pt[1]],
        ])

        # calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            (box_msg.center.position.x, box_msg.center.position.y),
            -np.rad2deg(box_msg.center.theta),
            1.0,
        )

        # rotate the corners of the rectangle
        rect_pts = np.int0(cv2.transform(np.array([rect_pts]), rotation_matrix)[0])

        # Draw the rotated rectangle
        for i in range(4):
            pt1 = tuple(rect_pts[i])
            pt2 = tuple(rect_pts[(i + 1) % 4])
            cv2.line(cv_image, pt1, pt2, color, 2)

        # write text
        label = f"{class_name}"
        label += f" ({track_id})" if track_id else ""
        label += " ({:.3f})".format(score)
        pos = (min_pt[0] + 5, min_pt[1] + 25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font, 1, color, 1, cv2.LINE_AA)

        return cv_image

    def draw_detection_line(self, cv_image: np.ndarray) -> np.ndarray:
        """Draw the detection line"""
        color = (0, 0, 255) if self.line_crossed else (0, 255, 255)
        thickness = 15 if self.line_crossed else 7
        
        cv2.line(cv_image, 
                 (self.detection_line[0], self.detection_line[1]),
                 (self.detection_line[2], self.detection_line[3]),
                 color, thickness)
        return cv_image

    def check_line_crossing(self, detection: Detection) -> bool:
        """Check if detection crosses the line"""
        box_msg: BoundingBox2D = detection.bbox
        
        # Calculate center point
        cx = box_msg.center.position.x
        cy = box_msg.center.position.y
        
        # Check if center crosses line
        if (self.detection_line[0] < cx < self.detection_line[2] and 
            self.detection_line[1] - 20 < cy < self.detection_line[1] + 20):
            
            track_id = detection.id
            if track_id and track_id not in self._counted_ids:
                self._counted_ids.add(track_id)
                
                # Update class count
                class_name = detection.class_name
                if class_name not in self._class_counts:
                    self._class_counts[class_name] = 0
                self._class_counts[class_name] += 1
                
                # Publish updated counts
                self.publish_counts()
                return True
        return False

    def publish_counts(self) -> None:
        """Publish counts as JSON string"""
        counts_msg = String()
        counts_msg.data = json.dumps(self._class_counts)
        self._counts_pub.publish(counts_msg)

    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray) -> None:
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        bb_marker_array = MarkerArray()
        kp_marker_array = MarkerArray()
        
        self.line_crossed = False

        for detection in detection_msg.detections:
            # Get/set color for class
            class_name = detection.class_name
            if class_name not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self._class_to_color[class_name] = (r, g, b)
            color = self._class_to_color[class_name]

            # Check line crossing
            if self.check_line_crossing(detection):
                self.line_crossed = True

            # Draw visualizations
            cv_image = self.draw_box(cv_image, detection, color)
            cv_image = self.draw_detection_line(cv_image)

            # Create 3D markers if available
            if detection.bbox3d.frame_id:
                marker = self.create_bb_marker(detection, color)
                marker.header.stamp = img_msg.header.stamp
                marker.id = len(bb_marker_array.markers)
                bb_marker_array.markers.append(marker)

        # Publish outputs
        self._dbg_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(cv_image, encoding=img_msg.encoding)
        )
        self._bb_markers_pub.publish(bb_marker_array)
        self._kp_markers_pub.publish(kp_marker_array)


def main():
    rclpy.init()
    node = DebugNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()