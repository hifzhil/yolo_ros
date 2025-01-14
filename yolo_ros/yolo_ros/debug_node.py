# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
from yolo_msgs.msg import ObjectCount, ObjectCounts


class ObjectCounter:
    def __init__(self):
        self.detection_lines = [
            [5, 150, 955, 150],
            [5, 350, 955, 350],
            [230, 5, 230, 535],
            [750, 5, 750, 535]
        ]
        self.line_colors = [(0, 255, 255) for _ in self.detection_lines]
        self.color_change_times = [None for _ in self.detection_lines]
        self.color_duration = 1.0
        self._previous_positions = {}
        self._counted_objects = {}
        self._class_counts = {}
        self.MAX_TRACKED_IDS = 1000

    def cleanup_if_needed(self):
        if len(self._counted_objects) > self.MAX_TRACKED_IDS:
            sorted_tracks = sorted(
                self._counted_objects.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.MAX_TRACKED_IDS]
            self._counted_objects = dict(sorted_tracks)

    def check_line_crossings(self, detection: Detection, current_time: float) -> bool:
        track_id = detection.id
        if not track_id:
            return False
            
        if track_id in self._counted_objects:
            self._counted_objects[track_id] = current_time
            return False
            
        current_x = detection.bbox.center.position.x
        current_y = detection.bbox.center.position.y
        
        if track_id not in self._previous_positions:
            self._previous_positions[track_id] = (current_x, current_y)
            return False
            
        previous_x, previous_y = self._previous_positions[track_id]
        self._previous_positions[track_id] = (current_x, current_y)
        
        crossed_any = False
        for i, line in enumerate(self.detection_lines):
            if i < 2:
                line_y = line[1]
                if (previous_y <= line_y and current_y > line_y) or \
                   (previous_y >= line_y and current_y < line_y):
                    self.line_colors[i] = (0, 255, 0)
                    self.color_change_times[i] = current_time
                    crossed_any = True
            else:
                line_x = line[0]
                if (previous_x <= line_x and current_x > line_x) or \
                   (previous_x >= line_x and current_x < line_x):
                    self.line_colors[i] = (0, 255, 0)
                    self.color_change_times[i] = current_time
                    crossed_any = True
        
        if crossed_any:
            self._counted_objects[track_id] = current_time
            if detection.class_name not in self._class_counts:
                self._class_counts[detection.class_name] = 0
            self._class_counts[detection.class_name] += 1
            del self._previous_positions[track_id]
            self.cleanup_if_needed()
            
        return crossed_any

    def update_line_colors(self, current_time: float):
        for i, change_time in enumerate(self.color_change_times):
            if change_time is not None:
                if (current_time - change_time) > self.color_duration:
                    self.line_colors[i] = (0, 255, 255)
                    self.color_change_times[i] = None

    def get_counts(self):
        return self._class_counts.copy()

    def draw_lines(self, image: np.ndarray) -> np.ndarray:
        for i, line in enumerate(self.detection_lines):
            cv2.line(image, 
                    (line[0], line[1]),
                    (line[2], line[3]),
                    self.line_colors[i], 
                    2)
        return image


class DebugNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("debug_node")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()
        self.object_counter = ObjectCounter()
        self.target_width = 960
        self.target_height = 540

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

        # pubs
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._raw_pub = self.create_publisher(Image, "raw_image", 10)
        self._count_pub = self.create_publisher(Image, "dbg_counting", 10)
        self._bb_markers_pub = self.create_publisher(MarkerArray, "dgb_bb_markers", 10)
        self._kp_markers_pub = self.create_publisher(MarkerArray, "dgb_kp_markers", 10)
        self._counts_pub = self.create_publisher(ObjectCounts, "/yolo/objects/counts", 10)
        self._counts_mqtt_pub = self.create_publisher(String, "/vista/dashcam/debug_counter", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # subs
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
        self.destroy_publisher(self._raw_pub)
        self.destroy_publisher(self._count_pub)
        self.destroy_publisher(self._bb_markers_pub)
        self.destroy_publisher(self._kp_markers_pub)
        self.destroy_publisher(self._counts_pub)
        self.destroy_publisher(self._counts_mqtt_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.target_width, self.target_height))

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
        rect_pts = np.array(
            [
                [min_pt[0], min_pt[1]],
                [max_pt[0], min_pt[1]],
                [max_pt[0], max_pt[1]],
                [min_pt[0], max_pt[1]],
            ]
        )

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

    def draw_mask(
        self, cv_image: np.ndarray, detection: Detection, color: Tuple[int]
    ) -> np.ndarray:
        mask_msg = detection.mask
        mask_array = np.array([[int(ele.x), int(ele.y)] for ele in mask_msg.data])

        if mask_msg.data:
            layer = cv_image.copy()
            layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
            cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
            cv_image = cv2.polylines(
                cv_image,
                [mask_array],
                isClosed=True,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        return cv_image

    def draw_keypoints(self, cv_image: np.ndarray, detection: Detection) -> np.ndarray:
        keypoints_msg = detection.keypoints

        ann = Annotator(cv_image)

        kp: KeyPoint2D
        for kp in keypoints_msg.data:
            color_k = (
                [int(x) for x in ann.kpt_color[kp.id - 1]]
                if len(keypoints_msg.data) == 17
                else colors(kp.id - 1)
            )

            cv2.circle(
                cv_image,
                (int(kp.point.x), int(kp.point.y)),
                5,
                color_k,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                cv_image,
                str(kp.id),
                (int(kp.point.x), int(kp.point.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_k,
                1,
                cv2.LINE_AA,
            )

        def get_pk_pose(kp_id: int) -> Tuple[int]:
            for kp in keypoints_msg.data:
                if kp.id == kp_id:
                    return (int(kp.point.x), int(kp.point.y))
            return None

        for i, sk in enumerate(ann.skeleton):
            kp1_pos = get_pk_pose(sk[0])
            kp2_pos = get_pk_pose(sk[1])

            if kp1_pos is not None and kp2_pos is not None:
                cv2.line(
                    cv_image,
                    kp1_pos,
                    kp2_pos,
                    [int(x) for x in ann.limb_color[i]],
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        return cv_image

    def create_bb_marker(self, detection: Detection, color: Tuple[int]) -> Marker:
        bbox3d = detection.bbox3d

        marker = Marker()
        marker.header.frame_id = bbox3d.frame_id

        marker.ns = "yolo_3d"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = bbox3d.center.position.x
        marker.pose.position.y = bbox3d.center.position.y
        marker.pose.position.z = bbox3d.center.position.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = bbox3d.size.x
        marker.scale.y = bbox3d.size.y
        marker.scale.z = bbox3d.size.z

        marker.color.b = color[0] / 255.0
        marker.color.g = color[1] / 255.0
        marker.color.r = color[2] / 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = detection.class_name

        return marker

    def create_kp_marker(self, keypoint: KeyPoint3D) -> Marker:
        marker = Marker()

        marker.ns = "yolo_3d"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = keypoint.point.x
        marker.pose.position.y = keypoint.point.y
        marker.pose.position.z = keypoint.point.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.b = keypoint.score * 255.0
        marker.color.g = 0.0
        marker.color.r = (1.0 - keypoint.score) * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = str(keypoint.id)

        return marker

    def publish_counts(self):
        counts = self.object_counter.get_counts()
        
        # ROS message
        msg = ObjectCounts()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.counts = [
            ObjectCount(class_name=class_name, count=count)
            for class_name, count in counts.items()
        ]
        self._counts_pub.publish(msg)
        
        # MQTT message
        mqtt_data = {
            "timestamp": msg.header.stamp.sec,
            "counts": counts
        }
        mqtt_msg = String()
        mqtt_msg.data = json.dumps(mqtt_data)
        self._counts_mqtt_pub.publish(mqtt_msg)

    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray) -> None:
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        resized_image = self.resize_image(cv_image)
        
        self.scale_y = resized_image.shape[0] / self.target_height
        self.scale_x = resized_image.shape[1] / self.target_width
        
        # Raw image
        self._raw_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(resized_image, encoding=img_msg.encoding)
        )

        bb_marker_array = MarkerArray()
        kp_marker_array = MarkerArray()

        # Debug image (with bounding boxes)
        debug_image = resized_image.copy()
        counting_image = resized_image.copy()
        
        # Update counting lines
        self.object_counter.update_line_colors(current_time)
        counting_image = self.object_counter.draw_lines(counting_image)

        for detection in detection_msg.detections:
            # random color
            class_name = detection.class_name

            if class_name not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self._class_to_color[class_name] = (r, g, b)

            color = self._class_to_color[class_name]

            # Original debug visualization
            debug_image = self.draw_box(debug_image, detection, color)
            debug_image = self.draw_mask(debug_image, detection, color)
            debug_image = self.draw_keypoints(debug_image, detection)

            if detection.bbox3d.frame_id:
                marker = self.create_bb_marker(detection, color)
                marker.header.stamp = img_msg.header.stamp
                marker.id = len(bb_marker_array.markers)
                bb_marker_array.markers.append(marker)

            if detection.keypoints3d.frame_id:
                for kp in detection.keypoints3d.data:
                    marker = self.create_kp_marker(kp)
                    marker.header.frame_id = detection.keypoints3d.frame_id
                    marker.header.stamp = img_msg.header.stamp
                    marker.id = len(kp_marker_array.markers)
                    kp_marker_array.markers.append(marker)

            # Counting functionality
            if self.object_counter.check_line_crossings(detection, current_time):
                self.publish_counts()
            counting_image = self.draw_box(counting_image, detection, color)

        # Publish all outputs
        self._dbg_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(debug_image, encoding=img_msg.encoding)
        )
        self._count_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(counting_image, encoding=img_msg.encoding)
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