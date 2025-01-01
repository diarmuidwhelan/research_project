import cv2
import torch
from ultralytics import YOLO
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional,Any
import time
from datetime import datetime
import json
import pandas as pd

class BallTracker:
    """Specialised ball tracking and interpolation"""
    def __init__(self):
        self.ball_history = []
        self.max_history = 30
        self.interpolation_window = 10
        self.smoothing_window = 5
        self.velocity_history = []

    def update(self, detection: Dict, frame_shape: Tuple[int, int]) -> Optional[Dict]:
        """Update ball tracker with new detection"""
        if detection['confidence'] > 0.25:  # Minimum confidence for ball
            self.ball_history.append({
                'center': self._get_center(detection['bbox']),
                'bbox': detection['bbox'],
                'frame': detection['frame'],
                'confidence': detection['confidence'],
                'velocity': self._calculate_velocity() if self.ball_history else np.array([0, 0])
            })

            # Maintain history length
            if len(self.ball_history) > self.max_history:
                self.ball_history.pop(0)

            return detection
        return None

    def interpolate(self, current_frame: int, frame_shape: Tuple[int, int]) -> Optional[Dict]:
        """Interpolate ball position when detection is missing"""
        if len(self.ball_history) < 2:
            return None

        # Get last known positions
        last_detection = self.ball_history[-1]
        prev_detection = self.ball_history[-2]

        frames_missing = current_frame - last_detection['frame']

        # Only interpolate for reasonable gaps
        if frames_missing > self.interpolation_window:
            return None

        # Calculate predicted position using polynomial fit
        predicted_center = self._predict_position(current_frame)
        if predicted_center is None:
            return None

        # Create interpolated detection
        bbox = self._center_to_bbox(predicted_center, frame_shape)

        return {
            'bbox': bbox,
            'class': 'ball',
            'confidence': last_detection['confidence'] * np.exp(-0.2 * frames_missing),  # Decay confidence
            'frame': current_frame,
            'interpolated': True
        }

    def _predict_position(self, target_frame: int) -> Optional[np.ndarray]:
        """Predict ball position using polynomial fitting"""
        if len(self.ball_history) < 3:
            return None

        # Get recent history
        recent_history = self.ball_history[-min(len(self.ball_history), 10):]

        # Extract timestamps and positions
        frames = np.array([h['frame'] for h in recent_history])
        positions = np.array([h['center'] for h in recent_history])

        try:
            # Fit polynomial for x and y coordinates separately
            x_poly = np.polyfit(frames, positions[:, 0], 2)
            y_poly = np.polyfit(frames, positions[:, 1], 2)

            # Predict new position
            pred_x = np.polyval(x_poly, target_frame)
            pred_y = np.polyval(y_poly, target_frame)

            return np.array([pred_x, pred_y])

        except np.linalg.LinAlgError:
            return None

    def smooth_trajectory(self) -> List[Dict]:
        """Apply smoothing to the ball trajectory"""
        if len(self.ball_history) < self.smoothing_window:
            return self.ball_history

        positions = np.array([h['center'] for h in self.ball_history])

        # Apply Savitzky-Golay filter
        from scipy.signal import savgol_filter
        smoothed_x = savgol_filter(positions[:, 0], self.smoothing_window, 2)
        smoothed_y = savgol_filter(positions[:, 1], self.smoothing_window, 2)

        smoothed_history = []
        for i, hist in enumerate(self.ball_history):
            smoothed_center = np.array([smoothed_x[i], smoothed_y[i]])
            smoothed_history.append({
                **hist,
                'center': smoothed_center,
                'bbox': self._center_to_bbox(smoothed_center, None)  # Frame shape not needed for relative coordinates
            })

        return smoothed_history

    def _calculate_velocity(self) -> np.ndarray:
        """Calculate current ball velocity"""
        if len(self.ball_history) < 2:
            return np.array([0, 0])

        current_pos = self.ball_history[-1]['center']
        prev_pos = self.ball_history[-2]['center']
        time_diff = self.ball_history[-1]['frame'] - self.ball_history[-2]['frame']

        if time_diff == 0:
            return np.array([0, 0])

        velocity = (current_pos - prev_pos) / time_diff

        # Update velocity history
        self.velocity_history.append(velocity)
        if len(self.velocity_history) > 5:
            self.velocity_history.pop(0)

        # Return smoothed velocity
        return np.mean(self.velocity_history, axis=0)

    @staticmethod
    def _get_center(bbox: np.ndarray) -> np.ndarray:
        """Convert bbox to center point"""
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])

    @staticmethod
    def _center_to_bbox(center: np.ndarray, frame_shape: Optional[Tuple[int, int]],
                       size: int = 30) -> np.ndarray:
        """Convert center point to bbox"""
        bbox = np.array([
            center[0] - size/2,
            center[1] - size/2,
            center[0] + size/2,
            center[1] + size/2
        ])

        # Clip to frame boundaries if frame shape is provided
        if frame_shape is not None:
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, frame_shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, frame_shape[0])

        return bbox

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

class TrackingDataExporter:
    def __init__(self, output_dir: str = 'tracking_data'):
        self.output_dir = output_dir
        self.frame_data = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)

    def _convert_to_native_types(self, data: Any) -> Any:
        """Convert numpy types to native Python types"""
        if isinstance(data, dict):
            return {key: self._convert_to_native_types(value) 
                   for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_native_types(item) for item in data]
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
            return int(data)
        elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def add_frame_data(self, frame_number: int, results, frame_size: tuple):
        """Add tracking data for a single frame"""
        frame_info = {
            'frame_number': int(frame_number),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'tracks': []
        }

        # Process YOLO results
        if results.boxes is not None:
            for box in results.boxes:
                # Get box coordinates
                bbox = box.xyxy[0].cpu().numpy()  # Get box in xyxy format
                cls = int(box.cls[0].item())      # Get class index
                conf = float(box.conf[0].item())  # Get confidence

                # Convert class index to name using results.names
                class_name = results.names[cls]

                # Create track data
                track_data = {
                    'track_id': int(frame_number),  # Placeholder ID
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': {
                        'x1': float(bbox[0]),
                        'y1': float(bbox[1]),
                        'x2': float(bbox[2]),
                        'y2': float(bbox[3])
                    },
                    'center': {
                        'x': float((bbox[0] + bbox[2]) / 2),
                        'y': float((bbox[1] + bbox[3]) / 2)
                    },
                    'width': float(bbox[2] - bbox[0]),
                    'height': float(bbox[3] - bbox[1]),
                    'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                }

                frame_info['tracks'].append(track_data)

        self.frame_data.append(frame_info)

    def save_to_json(self):
        """Save tracking data to JSON file"""
        output_file = os.path.join(self.output_dir, f'tracking_data_{self.timestamp}.json')
        
        tracking_data = {
            'metadata': {
                'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_frames': len(self.frame_data)
            },
            'frames': self.frame_data
        }
        
        # Convert to native Python types before saving
        tracking_data = self._convert_to_native_types(tracking_data)
        
        with open(output_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        return output_file

    def save_to_csv(self):
        """Save tracking data to CSV file"""
        flattened_data = []
        for frame_info in self.frame_data:
            frame_number = frame_info['frame_number']
            timestamp = frame_info['timestamp']
            
            for track in frame_info['tracks']:
                row = {
                    'frame_number': int(frame_number),
                    'timestamp': timestamp,
                    'track_id': int(track['track_id']),
                    'class_id': int(track['class_id']),
                    'class_name': str(track['class_name']),
                    'confidence': float(track['confidence']),
                    'x1': float(track['bbox']['x1']),
                    'y1': float(track['bbox']['y1']),
                    'x2': float(track['bbox']['x2']),
                    'y2': float(track['bbox']['y2']),
                    'center_x': float(track['center']['x']),
                    'center_y': float(track['center']['y']),
                    'width': float(track['width']),
                    'height': float(track['height']),
                    'area': float(track['area'])
                }
                
                flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        output_file = os.path.join(self.output_dir, f'tracking_data_{self.timestamp}.csv')
        df.to_csv(output_file, index=False)
        
        return output_file

    def generate_summary(self):
        """Generate summary statistics"""
        summary = {
            'total_frames': int(len(self.frame_data)),
            'detection_counts': {},
            'average_confidences': {},
            'spatial_distribution': {}
        }
        
        for frame in self.frame_data:
            for track in frame['tracks']:
                class_name = track['class_name']
                
                # Update detection counts
                if class_name not in summary['detection_counts']:
                    summary['detection_counts'][class_name] = 0
                summary['detection_counts'][class_name] += 1
                
                # Update confidence averages
                if class_name not in summary['average_confidences']:
                    summary['average_confidences'][class_name] = []
                summary['average_confidences'][class_name].append(track['confidence'])
                
                # Update spatial distribution
                if class_name not in summary['spatial_distribution']:
                    summary['spatial_distribution'][class_name] = {
                        'x': [], 'y': [], 'areas': []
                    }
                summary['spatial_distribution'][class_name]['x'].append(track['center']['x'])
                summary['spatial_distribution'][class_name]['y'].append(track['center']['y'])
                summary['spatial_distribution'][class_name]['areas'].append(track['area'])
        
        # Calculate averages
        for class_name in summary['average_confidences']:
            summary['average_confidences'][class_name] = float(np.mean(
                summary['average_confidences'][class_name]
            ))
        
        # Calculate spatial statistics
        for class_name in summary['spatial_distribution']:
            stats = summary['spatial_distribution'][class_name]
            summary['spatial_distribution'][class_name] = {
                'x_mean': float(np.mean(stats['x'])),
                'y_mean': float(np.mean(stats['y'])),
                'x_std': float(np.std(stats['x'])),
                'y_std': float(np.std(stats['y'])),
                'area_mean': float(np.mean(stats['areas'])),
                'area_std': float(np.std(stats['areas']))
            }
        
        summary_file = os.path.join(self.output_dir, f'summary_{self.timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_file
    
    def evaluate_tracking(self) -> Dict:
        """
        Evaluate tracking performance using self-assessment metrics
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert frame data to DataFrame
        tracking_data = []
        for frame_info in self.frame_data:
            frame_number = frame_info['frame_number']
            for track in frame_info['tracks']:
                tracking_data.append({
                    'frame_number': frame_number,
                    'track_id': track['track_id'],
                    'class_name': track['class_name'],
                    'confidence': track['confidence'],
                    'x1': track['bbox']['x1'],
                    'y1': track['bbox']['y1'],
                    'x2': track['bbox']['x2'],
                    'y2': track['bbox']['y2'],
                    'center_x': track['center']['x'],
                    'center_y': track['center']['y']
                })
        
        df = pd.DataFrame(tracking_data)
        
        # Evaluate tracking
        evaluator = TrackerSelfEvaluation()
        metrics = evaluator.evaluate_tracking(df)
        
        # Save metrics
        metrics_file = os.path.join(self.output_dir, f'metrics_{self.timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


class EnhancedGaelicTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracks = defaultdict(list)
        self.next_id = 0
        self.frame_count = 0  # Add frame counter

        # Tracking parameters
        self.max_age = 15
        self.min_hits = 3
        self.iou_threshold = 0.3

        # Ball tracking parameters
        self.ball_history = []
        self.max_ball_history = 30

        self.conf_thresholds = {
            'ball': 0.25,
            'player': 0.3,
            'goalkeeper': 0.3,
            'referee': 0.4,
            'umpire': 0.4
        }
    def _process_ball_detections(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """Process ball detections with interpolation"""
        # Initialise ball tracker if not exists
        if not hasattr(self, 'ball_tracker'):
            self.ball_tracker = BallTracker()

        # Find ball detections
        ball_detections = [d for d in detections if d['class'] == 'ball']
        other_detections = [d for d in detections if d['class'] != 'ball']

        # Update ball tracker
        current_ball = None
        if ball_detections:
            # Use highest confidence ball detection
            best_ball = max(ball_detections, key=lambda x: x['confidence'])
            current_ball = self.ball_tracker.update(best_ball, frame_shape)
        else:
            # Try to interpolate ball position
            current_ball = self.ball_tracker.interpolate(self.frame_count, frame_shape)

        # Combine detections
        if current_ball is not None:
            return [current_ball] + other_detections
        return other_detections

    # Modify the predict_and_track method to include ball processing
    def predict_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame with enhanced ball tracking"""
        self.frame_count += 1

        # Get detections
        results = self.model(frame)[0]
        detections = self._process_detections(results)

        # Process ball detections
        processed_detections = self._process_ball_detections(detections, frame.shape[:2])

        # Update tracks
        tracks = self._update_tracks(processed_detections)

        # Get smoothed ball trajectory
        if hasattr(self, 'ball_tracker'):
            ball_trajectory = self.ball_tracker.smooth_trajectory()
            if ball_trajectory:
                # Update ball track with smoothed trajectory
                ball_track_id = next((k for k, v in tracks.items()
                                    if v and v[-1]['class'] == 'ball'), None)
                if ball_track_id is not None:
                    tracks[ball_track_id] = ball_trajectory

        # Draw tracks
        annotated_frame = self._draw_tracks(frame.copy(), tracks)

        return annotated_frame, tracks



    def _process_detections(self, results) -> List[Dict]:
        """Process YOLOv8 detections"""
        detections = []

        for i in range(len(results.boxes)):
            box = results.boxes[i]
            cls = int(box.cls)
            conf = float(box.conf)

            class_name = results.names[cls]
            if conf < self.conf_thresholds.get(class_name, 0.3):
                continue

            bbox = box.xyxy[0].cpu().numpy()

            detections.append({
                'bbox': bbox,
                'class': class_name,
                'confidence': conf,
                'frame': self.frame_count
            })

        return detections
    def _predict_tracks(self) -> Dict:
      """Predict new locations of existing tracks"""
      predicted_tracks = {}

      for track_id, track_info in self.tracks.items():
          if not track_info:
              continue

          # Get last known position
          last_detection = track_info[-1]

          if len(track_info) < 2:
              # If only one detection, use it as prediction
              predicted_tracks[track_id] = last_detection
              continue

          # Get previous detection
          prev_detection = track_info[-2]

          # Calculate velocity
          current_bbox = last_detection['bbox']
          prev_bbox = prev_detection['bbox']

          # Calculate centre points
          current_center = self._get_center(current_bbox)
          prev_center = self._get_center(prev_bbox)

          # Calculate velocity
          velocity = current_center - prev_center

          # Predict new centre
          predicted_center = current_center + velocity

          # Convert back to bbox
          bbox_width = current_bbox[2] - current_bbox[0]
          bbox_height = current_bbox[3] - current_bbox[1]

          predicted_bbox = np.array([
              predicted_center[0] - bbox_width/2,
              predicted_center[1] - bbox_height/2,
              predicted_center[0] + bbox_width/2,
              predicted_center[1] + bbox_height/2
          ])

          predicted_tracks[track_id] = {
              'bbox': predicted_bbox,
              'class': last_detection['class'],
              'confidence': last_detection['confidence'] * 0.9,  # Reduce confidence for predictions
              'frame': self.frame_count
          }

      return predicted_tracks

    def _associate_detections_to_tracks(self, detections: List[Dict], predicted_tracks: Dict) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
            """Associate detections with existing tracks using IoU"""
            if not predicted_tracks:
                return [], list(range(len(detections))), []

            if not detections:
                return [], [], list(range(len(predicted_tracks)))

            # Calculate IoU matrix
            iou_matrix = np.zeros((len(detections), len(predicted_tracks)))
            for d, detection in enumerate(detections):
                for t, (track_id, track_info) in enumerate(predicted_tracks.items()):
                    if detection['class'] == track_info['class']:  # Only match same class
                        iou_matrix[d, t] = self._calculate_iou(
                            detection['bbox'],
                            track_info['bbox']
                        )

            # Use Hungarian algorithm for matching
            from scipy.optimize import linear_sum_assignment
            detection_indices, track_indices = linear_sum_assignment(-iou_matrix)

            # Filter matches with low IoU
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(predicted_tracks)))

            for d, t in zip(detection_indices, track_indices):
                if iou_matrix[d, t] >= self.iou_threshold:
                    matches.append((t, d))
                    if d in unmatched_detections:
                        unmatched_detections.remove(d)
                    if t in unmatched_tracks:
                        unmatched_tracks.remove(t)

            return matches, unmatched_detections, unmatched_tracks

    @staticmethod
    def _calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _initialise_track(self, detection: Dict):
          """Initialise new track"""
          self.tracks[self.next_id] = [detection]
          self.next_id += 1

    def _update_track(self, track_id: int, detection: Dict):
          """Update existing track"""
          self.tracks[track_id].append(detection)

          # Keep only recent history
          if len(self.tracks[track_id]) > self.max_age:
              self.tracks[track_id] = self.tracks[track_id][-self.max_age:]

    def _remove_old_tracks(self):
          """Remove old inactive tracks"""
          current_frame = self.frame_count
          tracks_to_remove = []

          for track_id, track_info in self.tracks.items():
              if not track_info:
                  tracks_to_remove.append(track_id)
                  continue

              last_frame = track_info[-1]['frame']
              age = current_frame - last_frame

              # Remove if track is too old
              if age > self.max_age:
                  tracks_to_remove.append(track_id)
                  continue

              # Remove if track quality is poor
              if len(track_info) < self.min_hits and age > self.max_age // 2:
                  tracks_to_remove.append(track_id)

          for track_id in tracks_to_remove:
              del self.tracks[track_id]

    def _update_ball_trajectory(self, detection: Dict):
        """Update ball trajectory with new detection"""
        self.ball_history.append({
            'position': self._get_center(detection['bbox']),
            'frame': self.frame_count,
            'confidence': detection['confidence']
        })

        # Keep history length in check
        if len(self.ball_history) > self.max_ball_history:
            self.ball_history.pop(0)

    def _interpolate_ball_position(self) -> Optional[Dict]:
        """Interpolate ball position when detection is missing"""
        if len(self.ball_history) < 2:
            return None

        # Get last two known positions
        last_pos = self.ball_history[-1]['position']
        prev_pos = self.ball_history[-2]['position']
        last_frame = self.ball_history[-1]['frame']

        # Only interpolate if not too many frames have passed
        frames_since_last = self.frame_count - last_frame
        if frames_since_last > 5:  # Max frames to interpolate
            return None

        # Calculate velocity
        velocity = (last_pos - prev_pos) / (last_frame - self.ball_history[-2]['frame'])

        # Predict new position
        predicted_pos = last_pos + velocity * frames_since_last

        # Convert back to bbox format
        bbox = self._center_to_bbox(predicted_pos)

        return {
            'bbox': bbox,
            'class': 'ball',
            'confidence': 0.3,  # Lower confidence for interpolated positions
            'frame': self.frame_count,
            'interpolated': True
        }

    def _update_tracks(self, detections: List[Dict]) -> Dict:
        """Update tracking states"""
        # Predict new locations of existing tracks
        predicted_tracks = self._predict_tracks()

        # Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = \
            self._associate_detections_to_tracks(detections, predicted_tracks)

        # Update matched tracks
        for track_idx, detection_idx in matches:
            self._update_track(track_idx, detections[detection_idx])

        # Initialise new tracks
        for detection_idx in unmatched_detections:
            self._initialise_track(detections[detection_idx])

        # Remove old tracks
        self._remove_old_tracks()

        return self.tracks

    def _draw_tracks(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """Draw tracking visualisation"""
        colors = {
            'ball': (0, 0, 255),      # Red
            'player': (0, 255, 0),    # Green
            'goalkeeper': (255, 0, 0), # Blue
            'referee': (255, 255, 0),  # Yellow
            'umpire': (255, 0, 255)    # Magenta
        }

        for track_id, track_info in tracks.items():
            if not track_info:
                continue

            current = track_info[-1]
            bbox = current['bbox'].astype(int)
            class_name = current['class']
            color = colors.get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label with track ID
            label = f"{class_name} {track_id}"
            if current.get('interpolated', False):
                label += " (interp)"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trajectory
            if len(track_info) > 1:
                points = np.array([self._get_center(t['bbox']) for t in track_info])
                points = points.astype(np.int32)

                # Draw smoother trajectory for ball
                if class_name == 'ball':
                    # Draw smooth curve through points
                    for i in range(len(points) - 1):
                        cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), color, 2)

                    # Add direction arrow
                    if len(points) > 2:
                        direction = points[-1] - points[-2]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction) * 30
                            end_point = points[-1] + direction
                            cv2.arrowedLine(frame, tuple(points[-1]), tuple(end_point.astype(int)),
                                         color, 2)
                else:
                    # Simple trajectory for other objects
                    cv2.polylines(frame, [points], False, color, 2)

        return frame
    @staticmethod
    def _get_center(bbox: np.ndarray) -> np.ndarray:
          """Get center point of bounding box"""
          return np.array([
              (bbox[0] + bbox[2]) / 2,
              (bbox[1] + bbox[3]) / 2
          ])

    @staticmethod
    def _center_to_bbox(center: np.ndarray, size: int = 30) -> np.ndarray:
      """Convert center point to bounding box"""
      return np.array([
                center[0] - size/2,
                center[1] - size/2,
                center[0] + size/2,
                center[1] + size/2
            ])

import numpy as np
from typing import Dict, List
from collections import defaultdict
import pandas as pd
from scipy.signal import savgol_filter

class TrackerSelfEvaluation:
    """Self-evaluation metrics for tracking without ground truth"""
    
    def __init__(self):
        self.metrics = defaultdict(dict)
        
    def evaluate_tracking(self, tracking_data: pd.DataFrame) -> Dict:
        """
        Evaluate tracking performance using self-assessment metrics
        
        Args:
            tracking_data: DataFrame with columns [frame_number, track_id, class_name, 
                                                 confidence, x1, y1, x2, y2, center_x, center_y]
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate metrics per class
        for class_name in tracking_data['class_name'].unique():
            class_data = tracking_data[tracking_data['class_name'] == class_name]
            self.metrics[class_name] = {
                'track_statistics': self._calculate_track_statistics(class_data),
                'stability_metrics': self._calculate_stability_metrics(class_data),
                'trajectory_metrics': self._calculate_trajectory_metrics(class_data),
                'detection_metrics': self._calculate_detection_metrics(class_data)
            }
            
            # Add special metrics for ball tracking
            if class_name == 'ball':
                self.metrics[class_name]['ball_specific'] = self._calculate_ball_metrics(class_data)
        
        return self._format_metrics()
    
    def _calculate_track_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate basic track statistics"""
        track_lengths = data.groupby('track_id').size()
        active_tracks_per_frame = data.groupby('frame_number')['track_id'].nunique()
        
        return {
            'total_tracks': len(track_lengths),
            'avg_track_length': float(track_lengths.mean()),
            'median_track_length': float(track_lengths.median()),
            'max_track_length': int(track_lengths.max()),
            'min_track_length': int(track_lengths.min()),
            'avg_tracks_per_frame': float(active_tracks_per_frame.mean()),
            'track_length_std': float(track_lengths.std()),
            'track_fragmentations': self._count_track_fragmentations(data)
        }
    
    def _calculate_stability_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate metrics related to tracking stability"""
        stability_metrics = {}
        
        # Calculate bbox size consistency
        data['bbox_area'] = (data['x2'] - data['x1']) * (data['y2'] - data['y1'])
        bbox_variations = data.groupby('track_id')['bbox_area'].agg(['std', 'mean'])
        stability_metrics['bbox_size_consistency'] = float(
            1 - (bbox_variations['std'] / bbox_variations['mean']).mean()
        )
        
        # Calculate position smoothness
        position_smoothness = []
        for _, track_data in data.groupby('track_id'):
            if len(track_data) > 2:
                # Calculate frame-to-frame position changes
                dx = np.diff(track_data['center_x'])
                dy = np.diff(track_data['center_y'])
                distances = np.sqrt(dx**2 + dy**2)
                smoothness = 1 - np.std(distances) / (np.mean(distances) + 1e-6)
                position_smoothness.append(smoothness)
        
        stability_metrics['position_smoothness'] = float(np.mean(position_smoothness)) if position_smoothness else 0.0
        
        # Calculate temporal consistency
        frame_gaps = []
        for _, track_data in data.groupby('track_id'):
            frame_numbers = sorted(track_data['frame_number'])
            gaps = np.diff(frame_numbers)
            frame_gaps.extend(gaps)
        
        stability_metrics['temporal_consistency'] = float(
            1 - (np.std(frame_gaps) / (np.mean(frame_gaps) + 1e-6))
        )
        
        return stability_metrics
    
    def _calculate_trajectory_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate metrics related to movement trajectories"""
        trajectory_metrics = {}
        
        # Calculate velocity consistency
        velocities = []
        accelerations = []
        
        for _, track_data in data.groupby('track_id'):
            if len(track_data) > 2:
                # Sort by frame number
                track_data = track_data.sort_values('frame_number')
                
                # Calculate velocities
                dx = np.diff(track_data['center_x'])
                dy = np.diff(track_data['center_y'])
                dt = np.diff(track_data['frame_number'])
                
                v = np.sqrt(dx**2 + dy**2) / dt
                velocities.extend(v)
                
                # Calculate accelerations
                a = np.diff(v) / dt[:-1]
                accelerations.extend(a)
        
        if velocities:
            trajectory_metrics['velocity_consistency'] = float(
                1 - np.std(velocities) / (np.mean(velocities) + 1e-6)
            )
            trajectory_metrics['avg_velocity'] = float(np.mean(velocities))
        
        if accelerations:
            trajectory_metrics['acceleration_consistency'] = float(
                1 - np.std(accelerations) / (np.mean(np.abs(accelerations)) + 1e-6)
            )
        
        return trajectory_metrics
    
    def _calculate_detection_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate detection-related metrics"""
        return {
            'avg_confidence': float(data['confidence'].mean()),
            'min_confidence': float(data['confidence'].min()),
            'confidence_stability': float(1 - data['confidence'].std()),
            'detection_density': float(len(data) / (data['frame_number'].max() - data['frame_number'].min() + 1))
        }
    
    def _calculate_ball_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate ball-specific tracking metrics"""
        ball_metrics = {}
        
        # Calculate ball coverage
        total_frames = data['frame_number'].max() - data['frame_number'].min() + 1
        frames_with_ball = len(data['frame_number'].unique())
        ball_metrics['coverage'] = float(frames_with_ball / total_frames)
        
        # Calculate trajectory smoothness using Savitzky-Golay filter
        if len(data) > 10:
            try:
                # Sort data by frame number
                sorted_data = data.sort_values('frame_number')
                
                # Apply Savitzky-Golay filter to positions
                window = min(11, len(data) - 2 if len(data) % 2 == 0 else len(data) - 1)
                if window > 2:
                    smoothed_x = savgol_filter(sorted_data['center_x'], window, 2)
                    smoothed_y = savgol_filter(sorted_data['center_y'], window, 2)
                    
                    # Calculate deviation from smoothed trajectory
                    deviation_x = np.mean(np.abs(sorted_data['center_x'] - smoothed_x))
                    deviation_y = np.mean(np.abs(sorted_data['center_y'] - smoothed_y))
                    
                    ball_metrics['trajectory_smoothness'] = float(
                        1 - (deviation_x + deviation_y) / (sorted_data['center_x'].std() + sorted_data['center_y'].std())
                    )
            except Exception as e:
                print(f"Error calculating ball trajectory smoothness: {e}")
                ball_metrics['trajectory_smoothness'] = 0.0
        
        # Calculate typical ball movement patterns
        velocities = []
        for _, track_data in data.groupby('track_id'):
            if len(track_data) > 1:
                # Sort by frame number
                track_data = track_data.sort_values('frame_number')
                
                # Calculate velocities
                dx = np.diff(track_data['center_x'])
                dy = np.diff(track_data['center_y'])
                dt = np.diff(track_data['frame_number'])
                
                v = np.sqrt(dx**2 + dy**2) / dt
                velocities.extend(v)
        
        if velocities:
            ball_metrics['avg_velocity'] = float(np.mean(velocities))
            ball_metrics['max_velocity'] = float(np.max(velocities))
            ball_metrics['velocity_stability'] = float(1 - np.std(velocities) / (np.mean(velocities) + 1e-6))
        
        return ball_metrics
    
    def _count_track_fragmentations(self, data: pd.DataFrame) -> int:
        """Count number of track fragmentations"""
        fragmentations = 0
        for _, track_data in data.groupby('track_id'):
            frame_numbers = sorted(track_data['frame_number'])
            gaps = np.diff(frame_numbers)
            fragmentations += np.sum(gaps > 1)  # Count gaps larger than 1 frame
        return int(fragmentations)
    
    def _format_metrics(self) -> Dict:
        """Format metrics for output"""
        formatted_metrics = {}
        
        for class_name, class_metrics in self.metrics.items():
            formatted_metrics[class_name] = {
                'Track Statistics': {
                    'Total Tracks': class_metrics['track_statistics']['total_tracks'],
                    'Average Track Length': f"{class_metrics['track_statistics']['avg_track_length']:.2f} frames",
                    'Track Fragmentations': class_metrics['track_statistics']['track_fragmentations'],
                    'Average Tracks per Frame': f"{class_metrics['track_statistics']['avg_tracks_per_frame']:.2f}"
                },
                'Stability Metrics': {
                    'Position Smoothness': f"{class_metrics['stability_metrics']['position_smoothness']*100:.2f}%",
                    'Temporal Consistency': f"{class_metrics['stability_metrics']['temporal_consistency']*100:.2f}%",
                    'Bounding Box Consistency': f"{class_metrics['stability_metrics']['bbox_size_consistency']*100:.2f}%"
                },
                'Trajectory Metrics': {
                    'Velocity Consistency': f"{class_metrics['trajectory_metrics'].get('velocity_consistency', 0)*100:.2f}%",
                    'Average Velocity': f"{class_metrics['trajectory_metrics'].get('avg_velocity', 0):.2f} pixels/frame"
                },
                'Detection Metrics': {
                    'Average Confidence': f"{class_metrics['detection_metrics']['avg_confidence']*100:.2f}%",
                    'Detection Density': f"{class_metrics['detection_metrics']['detection_density']*100:.2f}%"
                }
            }
            
            # Add ball-specific metrics if applicable
            if class_name == 'ball' and 'ball_specific' in class_metrics:
                formatted_metrics[class_name]['Ball Specific'] = {
                    'Coverage': f"{class_metrics['ball_specific']['coverage']*100:.2f}%",
                    'Trajectory Smoothness': f"{class_metrics['ball_specific'].get('trajectory_smoothness', 0)*100:.2f}%",
                    'Velocity Stability': f"{class_metrics['ball_specific'].get('velocity_stability', 0)*100:.2f}%"
                }
        
        return formatted_metrics



class VideoProcessor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.data_exporter = TrackingDataExporter()

    def process_video(self, input_path: str, output_path: str, show_display: bool = True):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO detection
                results = self.model(frame)[0]
                
                # Export tracking data
                self.data_exporter.add_frame_data(
                    frame_count,
                    results,
                    frame.shape[:2]
                )

                # Draw results on frame
                annotated_frame = results.plot()

                # Write and display frame
                out.write(annotated_frame)
                if show_display:
                    cv2.imshow('Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Save tracking data
            json_file = self.data_exporter.save_to_json()
            csv_file = self.data_exporter.save_to_csv()
            summary_file = self.data_exporter.generate_summary()
            
            print(f"\nTracking data saved to:")
            print(f"JSON: {json_file}")
            print(f"CSV: {csv_file}")
            print(f"Summary: {summary_file}")
            print("\nEvaluating tracking performance...")
            metrics = self.data_exporter.evaluate_tracking()
            
            print("\nTracking Performance Summary:")
            for class_name, class_metrics in metrics.items():
                print(f"\n{class_name.upper()} METRICS:")
                for category, category_metrics in class_metrics.items():
                    print(f"\n{category}:")
                    for metric_name, value in category_metrics.items():
                        print(f"  {metric_name}: {value}")

def main():
    # Set paths
    model_path = '/Users/dwhelan/Downloads/research_project/detect3/yolov8m_ball_enhanced/weights/best.pt'  # Trained Yolov8 model path
    video_path = '/Users/dwhelan/Documents/GAACV/input_videos/Clare Shot Possession_58.mp4'  # input video path
    output_path = '/Users/dwhelan/Downloads/24_34.mp4' # where to output the processed video

    # Initialise processor and process video
    try:
        processor = VideoProcessor(model_path)
        processor.process_video(video_path, output_path)
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()
