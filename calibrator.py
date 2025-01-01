import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import json
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from ultralytics import YOLO
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from functools import lru_cache
from dataclasses import dataclass
import queue
import shutil
from pathlib import Path
from datetime import datetime
import math
import traceback
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__getitem__') and hasattr(obj, 'keys'):
            return dict(obj)
        elif isinstance(obj, ScaledLandmark):
            return {
                'x': float(obj.x),
                'y': float(obj.y),
                'z': float(obj.z),
                'visibility': float(obj.visibility)
            }
        return json.JSONEncoder.default(self, obj)

class PlayerEditor:
    def __init__(self, master, players_data, frame, template_image, team1_colour, team2_colour, parent=None):
        self.window = tk.Toplevel(master)
        self.parent = parent  # Store parent reference
        self.window.title("Player Editor")
        self.window.geometry("1200x800")
        
        self.template_width = 600  # Add template dimensions
        self.template_height = 450
        
       # Inherit team colours from tagger
        self.team1_colour = team1_colour
        self.team2_colour = team2_colour
        self.team_colours = {
            'team_1': (0, 255, 0),
            'team_2': (0, 0, 255)
        }
        
        self.players_data = players_data
        self.current_player_index = 0
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.zoom_factor = 2
        self.template_image = template_image
        
        self.player_poses = {}  # Store poses separately
        
        # Initialise poses for all players
        for i, player in enumerate(self.players_data):
            if not player.get('pose'):
               player['pose'] = {
                'landmarks': [None] * 33,
                'orientation': 'facing_right'
            }  # MediaPipe has 33 landmarks
            if not player.get('orientation'):
                player['orientation'] = 'facing_right'
        # Add orientation variable
        self.orientation_var = tk.StringVar(value="facing_right")  # Default orientation
        
        self.create_ui()
        self.load_current_player()

    def zoom_in(self):
        self.zoom_factor = min(4, self.zoom_factor * 1.5)
        self.update_display()

    def zoom_out(self):
        self.zoom_factor = max(1, self.zoom_factor / 1.5)
        self.update_display()
        
    def create_ui(self):
        # Navigation frame at top
        nav_frame = ttk.Frame(self.window)
        nav_frame.pack(fill=tk.X)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_player).pack(side=tk.LEFT)
        self.player_label = ttk.Label(nav_frame, text="Player 1/{}")
        self.player_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Next", command=self.next_player).pack(side=tk.LEFT)
        
        # Main content
        content_frame = ttk.Frame(self.window)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for image
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(left_panel)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Template canvas
        template_frame = ttk.LabelFrame(left_panel, text="Field View")
        template_frame.pack(fill=tk.BOTH, expand=True)
        self.template_canvas = tk.Canvas(template_frame, 
                                       width=self.template_width, 
                                       height=self.template_height)
        self.template_canvas.pack(padx=5, pady=5)
        
        # Right panel controls
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Team selection
        team_frame = ttk.LabelFrame(right_panel, text="Team")
        team_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.team_var = tk.StringVar()
        ttk.Radiobutton(team_frame, text="Team 1", variable=self.team_var, 
                       value="team_1", command=self.update_display).pack()
        ttk.Radiobutton(team_frame, text="Team 2", variable=self.team_var, 
                       value="team_2", command=self.update_display).pack()
        
        # Number selection
        number_frame = ttk.LabelFrame(right_panel, text="Number")
        number_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.number_var = tk.StringVar()
        number_combo = ttk.Combobox(number_frame, textvariable=self.number_var,
                                  values=list(range(1, 36)))
        number_combo.pack()
        number_combo.bind('<<ComboboxSelected>>', lambda e: self.update_display())
        
        # Pose adjustment
        pose_frame = ttk.LabelFrame(right_panel, text="Pose Keypoints")
        pose_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.keypoint_var = tk.StringVar()
        for part in ['Right Arm', 'Left Arm', 'Right Leg', 'Left Leg']:
            ttk.Radiobutton(pose_frame, text=part, variable=self.keypoint_var,
                          value=part, command=self.start_pose_edit).pack()
                # Add goalkeeper toggle
        self.is_goalkeeper = tk.BooleanVar(value=False)
        ttk.Checkbutton(right_panel, text="Goalkeeper", 
                       variable=self.is_goalkeeper,
                       command=self.update_display).pack()
        # Zoom control
        zoom_frame = ttk.LabelFrame(right_panel, text="Zoom")
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(zoom_frame, text="+", command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="-", command=self.zoom_out).pack(side=tk.LEFT)
        
        # Add orientation selection
        orientation_frame = ttk.LabelFrame(right_panel, text="Orientation")
        orientation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        orientations = [
            ("Facing Right", "facing_right"),
            ("Facing Left", "facing_left"),
            ("Facing Up", "facing_up"),
            ("Facing Down", "facing_down")
        ]
        
        for text, value in orientations:
            ttk.Radiobutton(orientation_frame, text=text, 
                          variable=self.orientation_var,
                          value=value,
                          command=self.update_display).pack(anchor=tk.W)


        # Save and close buttons
        ttk.Button(right_panel, text="Save Changes", 
                  command=self.save_current).pack(side=tk.BOTTOM, pady=5)
        ttk.Button(right_panel, text="Done", 
                  command=self.save_and_close).pack(side=tk.BOTTOM, pady=5)
                  
    def load_current_player(self):
        player = self.players_data[self.current_player_index]
        self.player_label.config(text=f"Player {self.current_player_index + 1}/{len(self.players_data)}")
        
        # Set current values
        self.team_var.set(player.get('jersey_colour', 'team_1'))
        self.number_var.set(str(player.get('number', '')))
        self.is_goalkeeper.set(player.get('is_goalkeeper', False))
        # Set orientation - handle None cases safely
        orientation = (
            player.get('orientation') or 
            (player.get('pose') or {}).get('orientation') or 
            'facing_right'
        )
        self.orientation_var.set(orientation)
        # Save the zoomed ROI for model training
        player = self.players_data[self.current_player_index]
        bbox = player['bbox']
        # Extract ROI with padding
        pad = 20
        roi = self.original_frame[
            max(0, int(bbox[1])-pad):min(self.original_frame.shape[0], int(bbox[3])+pad),
            max(0, int(bbox[0])-pad):min(self.original_frame.shape[1], int(bbox[2])+pad)
        ]
        
        if roi.size > 0:
            # Save for each model type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_dir = Path("training_data")
            
            # Color classification
            colour_path = base_dir / "colour_classification/images" / f"{timestamp}.jpg"
            colour_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(colour_path), roi)
            
            # Number recognition 
            num_path = base_dir / "number_recognition/images" / f"{timestamp}.jpg"
            num_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(num_path), roi)
            
            # Pose estimation
            pose_path = base_dir / "pose_estimation/images" / f"{timestamp}.jpg"
            pose_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pose_path), roi)
            
            # Save corresponding labels
            if player.get('jersey_colour'):
                with open(str(base_dir / "colour_classification/labels" / f"{timestamp}.txt"), 'w') as f:
                    f.write(player['jersey_colour'])
                    
            if player.get('number'):
                with open(str(base_dir / "number_recognition/labels" / f"{timestamp}.txt"), 'w') as f:
                    f.write(str(player['number']))
                    
            if player.get('pose'):
                with open(str(base_dir / "pose_estimation/keypoints" / f"{timestamp}.json"), 'w') as f:
                    json.dump(player['pose'], f, cls=NumpyJSONEncoder)
        self.update_display()
        
    def next_player(self):
        self.save_current()
        self.current_player_index = (self.current_player_index + 1) % len(self.players_data)
        self.load_current_player()
        
    def prev_player(self):
        self.save_current()
        self.current_player_index = (self.current_player_index - 1) % len(self.players_data)
        self.load_current_player()
        
    def update_display(self):
        """Update display with current player data"""

        player = self.players_data[self.current_player_index]
        self.display_frame = self.original_frame.copy()
        
        # Draw current annotations on full frame
        for i, p in enumerate(self.players_data):
            colour = (0, 255, 0) if p['jersey_colour'] == 'team_1' else (0, 0, 255)
            if i == self.current_player_index:
                colour = (255, 255, 0)  # Highlight current player
                
            self.draw_player_annotation(self.display_frame, p, colour)
        
        # Extract and zoom ROI for current player
        x1, y1, x2, y2 = player['bbox']
        pad = 20
        roi = self.display_frame[max(0, y1-pad):min(self.display_frame.shape[0], y2+pad),
                                max(0, x1-pad):min(self.display_frame.shape[1], x2+pad)]
        
        if roi.size > 0:
            roi = cv2.resize(roi, None, fx=self.zoom_factor, fy=self.zoom_factor)
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_roi))
            self.canvas.config(width=roi.shape[1], height=roi.shape[0])
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Update template display through parent
        if self.parent and hasattr(self.parent, 'draw_template_annotations'):
            template_display = self.parent.draw_template_annotations()
            if template_display is not None:
                self.display_image(template_display, self.template_canvas)

    def draw_player_annotation(self, frame, player, colour):
        """Draw player annotation with orientation"""

        bbox = player['bbox']
        
        # Draw more aesthetic ellipse
        y2 = int(bbox[3])
        x_centre = int((bbox[0] + bbox[2]) / 2)
        width = int(bbox[2] - bbox[0])
        # Special marker for goalkeeper
        is_goalkeeper = player.get('is_goalkeeper', False)
        if is_goalkeeper:
            # Draw star or special marker
            star_points = self.calculate_star_points(x_centre, y2, width//2)
            cv2.polylines(frame, [np.int32(star_points)], True, colour, 2, cv2.LINE_AA)
        else:
            # Shadow effect
            cv2.ellipse(frame,
                    center=(x_centre+2, y2+2),
                    axes=(int(width/2), int(width*0.35)),
                    angle=0, startAngle=-45, endAngle=235,
                    color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                    
            # Main ellipse
            cv2.ellipse(frame,
                    center=(x_centre, y2),
                    axes=(int(width/2), int(width*0.35)),
                    angle=0, startAngle=-45, endAngle=235,
                    color=colour, thickness=2, lineType=cv2.LINE_AA)

        # Number with better visibility
        if player.get('number'):
            number = str(player['number'])
            if is_goalkeeper:
                number = 'GK' + number
            rect_w = max(15, len(number) * 10)
            rect_h = 12
            x1_rect = x_centre - rect_w//2
            y1_rect = y2 + 15
            
            # Number background
            cv2.rectangle(frame,
                        (int(x1_rect), int(y1_rect)),
                        (int(x1_rect + rect_w), int(y1_rect + rect_h)),
                        colour, cv2.FILLED)
            
            # Number text with outline
            cv2.putText(frame, number,
                    (int(x1_rect + rect_w/4), int(y1_rect + rect_h - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, number,
                    (int(x1_rect + rect_w/4), int(y1_rect + rect_h - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        # Draw player position on template
        # Only draw on template if it exists
        if self.template_image is not None and player.get('transformed_position'):
            tx, ty = player['transformed_position']
            marker_size = 8 if is_goalkeeper else 5
            cv2.circle(self.template_image,
                    (int(tx), int(ty)),
                    marker_size, colour, -1, cv2.LINE_AA)
            if player.get('number'):
                prefix = 'GK' if is_goalkeeper else ''
                cv2.putText(self.template_image, prefix + str(player['number']),
                        (int(tx) + 5, int(ty) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        colour, 1, cv2.LINE_AA)


            # Draw pose if available
            if player.get('pose') and player['pose'].get('landmarks'):
                self.draw_pose(frame, player['pose']['landmarks'], colour)
            # Draw orientation arrow
            orientation = player.get('orientation') or player.get('pose', {}).get('orientation')
            if orientation:
                arrow_length = 20
                end_x, end_y = x_centre, y2
                
                if orientation == "facing_left":
                    end_x = x_centre - arrow_length
                elif orientation == "facing_right":
                    end_x = x_centre + arrow_length
                elif orientation == "facing_up":
                    end_y = y2 - arrow_length
                elif orientation == "facing_down":
                    end_y = y2 + arrow_length
                    
                # Draw orientation arrow
                cv2.arrowedLine(frame, 
                            (x_centre, y2),
                            (int(end_x), int(end_y)),
                            colour, 2, cv2.LINE_AA)
    def calculate_star_points(self, x_centre, y_centre, radius):
        points = []
        for i in range(10):
            angle = math.pi/2 + i * 2*math.pi/10
            r = radius if i % 2 == 0 else radius/2
            x = x_centre + r * math.cos(angle)
            y = y_centre + r * math.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def draw_pose(self, frame, landmarks, colour):
        if not landmarks:
            return

        connections = [
            (11, 12), (12, 14), (14, 16), (11, 13), (13, 15),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
            (11, 23), (12, 24)
        ]
        
        try:
            for start_idx, end_idx in connections:
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    landmarks[start_idx] and landmarks[end_idx]):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    if hasattr(start, 'visibility') and hasattr(end, 'visibility'):
                        if start.visibility > 0.5 and end.visibility > 0.5:
                            cv2.line(frame, 
                                   (int(start.x), int(start.y)),
                                   (int(end.x), int(end.y)),
                                   colour, 2)
        except Exception as e:
            print(f"Error drawing pose: {e}")
                            
    def start_pose_edit(self):
        if self.keypoint_var.get():
            self.canvas.bind('<Button-1>', self.place_keypoint)
            self.window.config(cursor='crosshair')
            
    def place_keypoint(self, event):
        part = self.keypoint_var.get()
        player = self.players_data[self.current_player_index]
        
        bbox = player['bbox']
        pad = 20
        frame_x = event.x / self.zoom_factor + bbox[0] - pad
        frame_y = event.y / self.zoom_factor + bbox[1] - pad
        
        # Initialise pose structure if it doesn't exist
        if not player.get('pose'):
            player['pose'] = {}
        if not player['pose'].get('landmarks'):
            player['pose']['landmarks'] = [None] * 33

        part_to_landmarks = {
            'Right Arm': [11, 13, 15],
            'Left Arm': [12, 14, 16],
            'Right Leg': [23, 25, 27],
            'Left Leg': [24, 26, 28]
        }
        
        if part in part_to_landmarks:
            for idx in part_to_landmarks[part]:
                player['pose']['landmarks'][idx] = ScaledLandmark(
                    x=frame_x, y=frame_y, z=0.0, visibility=1.0)

        self.update_display()
        
    def get_player_orientation(self, landmarks):
        """Calculate player orientation from pose landmarks"""
        if not landmarks:
            return None
            
        try:
            # Get key landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            if not all(lm.visibility > 0.5 for lm in [left_shoulder, right_shoulder, left_hip, right_hip]):
                return None

            # Calculate shoulder vector
            shoulder_vector = np.array([left_shoulder.x - right_shoulder.x,
                                    left_shoulder.y - right_shoulder.y])
            
            # Calculate hip vector 
            hip_vector = np.array([left_hip.x - right_hip.x,
                                left_hip.y - right_hip.y])
            
            # Combine vectors for more robust orientation
            body_vector = shoulder_vector + hip_vector
            
            # Determine orientation based on body vector
            angle = np.arctan2(body_vector[1], body_vector[0])
            angle_deg = np.degrees(angle)
            
            # Convert angle to orientation
            if -45 <= angle_deg <= 45:
                return "facing_right"
            elif 45 < angle_deg <= 135:
                return "facing_down"
            elif angle_deg > 135 or angle_deg <= -135:
                return "facing_left"
            else:
                return "facing_up"
                
        except Exception as e:
            print(f"Error calculating orientation: {e}")
        return None
    
    def draw_player_orientations(self):
        """Draw player orientations on template after player editing"""
        for player in self.players:
            if player.get('transformed_position') and player.get('pose'):
                tx, ty = player['transformed_position']
                
                # Calculate orientation from pose landmarks
                orientation = self.get_player_orientation(player['pose']['landmarks'])
                
                # Draw arrow indicating direction
                arrow_length = 15
                if orientation == "facing left":
                    end_x = tx - arrow_length
                    end_y = ty
                elif orientation == "facing right":
                    end_x = tx + arrow_length
                    end_y = ty 
                else:  # front/back - draw circle
                    cv2.circle(self.template_image, (int(tx), int(ty)), 
                            3, (0, 0, 255), -1)
                    continue
                    
                cv2.arrowedLine(self.template_image,
                            (int(tx), int(ty)),
                            (int(end_x), int(end_y)),
                            (0, 0, 255), 2)
    def display_image(self, img, canvas):
        """Convert and display an image on a canvas"""
        if img is None:
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(image=img_pil)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo 


    def save_current(self):
        """Save current player data"""
        player = self.players_data[self.current_player_index]
                # Save all player attributes
        player['jersey_colour'] = self.team_var.get()
        player['is_goalkeeper'] = self.is_goalkeeper.get()
        player = self.players_data[self.current_player_index]
       # Save orientation both directly and in pose data
        orientation = self.orientation_var.get()
        player['orientation'] = orientation
        if player.get('pose') is None:
            player['pose'] = {}
        player['pose']['orientation'] = orientation
        
        if self.number_var.get() and self.number_var.get() != 'None':
            try:
                player['number'] = int(self.number_var.get())
            except ValueError:
                pass
        # Save orientation both directly and in pose data
        orientation = self.orientation_var.get()
        player['orientation'] = orientation
        if player.get('pose') is None:
            player['pose'] = {}
        player['pose']['orientation'] = orientation
        
        player['manually_adjusted'] = True

        # Update transformed position
        if hasattr(self.parent, 'transform_coordinates'):
            self.parent.transform_coordinates()
            
        # Redraw template annotations
        if hasattr(self.parent, 'draw_template_annotations'):
            template_display = self.parent.draw_template_annotations()
            if template_display is not None:
                self.display_image(template_display, self.template_canvas)        

    def save_and_close(self):
        self.save_current()
        if hasattr(self.parent, 'draw_player_orientations'):
            self.parent.draw_player_orientations()
        self.window.destroy()


class PoseEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ScaledLandmark):
            return {
                'x': float(obj.x),
                'y': float(obj.y),
                'z': float(obj.z),
                'visibility': float(obj.visibility)
            }
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
    
@dataclass
class ScaledLandmark:
    x: float
    y: float
    z: float
    visibility: float

class NumberRecogniser(nn.Module):
    def __init__(self, num_classes=35):
        super(NumberRecogniser, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class GaelicFootballCalibrator:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Gaelic Football Field Calibrator")
        self.master.geometry("1280x720")

        # Canvas dimensions
        self.template_width = 600
        self.template_height = 450
                
        # Colours and classes
        self.colour_classes = {
            0: 'black',
            1: 'blue', 
            2: 'green',
            3: 'maroon',
            4: 'orange',
            5: 'red',
            6: 'white',
            7: 'yellow'
        }       
        self.team_colours = {
            'team_1': (0, 255, 0),
            'team_2': (0, 0, 255)
        }
        self.team1_colour = None
        self.team2_colour = None
        
        # Initialize models and variables
        self.template_path = "/Users/diarmuidwhelan/Downloads/research_project/PitchTemplate.png"
        self.current_frame = None
        self.homography = None
        self.processed = False
        self.calibrated = False
        
        # Model paths
        self.yolo_path = "/Users/diarmuidwhelan/Downloads/research_project/detect3/yolov8m_ball_enhanced/weights/best.pt"
        self.jersey_classifier_path = "/Users/diarmuidwhelan/Downloads/research_project/best_jersey_classifier.pth"
        self.number_classifier_path = "/Users/diarmuidwhelan/Downloads/research_project/best_number_classifier.pth"
        
        # Load models
        self.load_models()
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Detection storage
        self.players = []
        self.goalkeepers = []
        self.referees = []
        self.ball = None
        self.manual_adjustments = set()
        
        # UI state variables
        self.manual_mode = False
        self.interaction_mode = tk.StringVar(value="none")
        self.current_team = tk.StringVar(value="team_1")
        self.selected_object = None
        self.dragging = False
        self.drag_start = None
        self.max_workers = min(cpu_count() * 2, 8)  # 2 threads per CPU core, max 8
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer 30 frames

        
        # Calibration points
        self.image_points = []
        self.template_points = []

        self.load_fixed_template()
        self.create_ui()

    def load_models(self):
        try:
            # Load YOLO model
            self.player_ball_model = YOLO(self.yolo_path)
            
            # Load jersey classifier
            self.jersey_classifier = models.resnet18(pretrained=False)
            self.jersey_classifier.fc = nn.Linear(self.jersey_classifier.fc.in_features, len(self.colour_classes))
            jersey_checkpoint = torch.load(self.jersey_classifier_path, map_location=torch.device('cpu'))
            self.jersey_classifier.load_state_dict(jersey_checkpoint['model_state_dict'])
            self.jersey_classifier.eval()
            
            # Load number classifier
            self.number_classifier = NumberRecogniser()
            number_checkpoint = torch.load(self.number_classifier_path, map_location=torch.device('cpu'))
            self.number_classifier.load_state_dict(number_checkpoint['model_state_dict'])
            self.number_classifier.eval()
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.player_ball_model = None
            self.jersey_classifier = None
            self.number_classifier = None
    def load_fixed_template(self):
        try:
            if os.path.exists(self.template_path):
                self.template_image = cv2.imread(self.template_path)
                if self.template_image is not None:
                    self.template_image = cv2.resize(self.template_image, 
                                                (self.template_width, self.template_height))
                else:
                    raise ValueError("Failed to load template image")
            else:
                # Create default template if file doesn't exist
                self.template_image = np.ones((self.template_height, self.template_width, 3), 
                                            dtype=np.uint8) * 255
         
                
        except Exception as e:
            print(f"Template loading error: {str(e)}")
            # Create blank template as fallback
            self.template_image = np.ones((self.template_height, self.template_width, 3), 
                                        dtype=np.uint8) * 255
    def process_frame(self):
        if self.current_frame is None or not self.calibrated:
            messagebox.showerror("Error", "Please calibrate first")
            return
                
        try:
            self.players = []
            self.goalkeepers = []
            self.ball = None

            if self.player_ball_model is None:
                raise ValueError("YOLO model not loaded")

            results = self.player_ball_model(self.current_frame)
            all_detections = []
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls.item())

                    all_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': float(box.conf.item()),
                        'cls': int(box.cls.item()),
                         'is_goalkeeper': class_id == 1  #  class 1 is goalkeeper

                    })



            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for detection in all_detections:
                    if detection['cls'] in [1, 2]:
                        futures.append(executor.submit(self.process_single_detection, 
                                                    detection, 
                                                    self.team1_colour,
                                                    self.team2_colour))
                    elif detection['cls'] == 0:
                        self.ball = detection['bbox']

                for future in futures:
                    result = future.result()
                    if result:
                        if result.get('is_goalkeeper'):
                            self.goalkeepers.append(result)
                        else:
                            self.players.append(result)

            self.transform_coordinates()
            self.processed = True
            self.display_processed_frame()
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
            messagebox.showerror("Error", "Frame processing failed")

    def get_player_orientation(self, landmarks):
        if not landmarks:
            return None
            
        try:
            # Get shoulder landmarks
            left_shoulder = next((lm for lm in landmarks if isinstance(lm, dict) and 
                                lm.get('visibility', 0) > 0.5), None)
            right_shoulder = next((lm for lm in landmarks if isinstance(lm, dict) and 
                                lm.get('visibility', 0) > 0.5), None)
            
            if left_shoulder and right_shoulder:
                shoulder_diff = left_shoulder['x'] - right_shoulder['x']
                
                if abs(shoulder_diff) < 0.1:
                    return "front/back"
                elif shoulder_diff > 0:
                    return "facing left"
                else:
                    return "facing right"
            return None
        except Exception as e:
            print(f"Orientation error: {e}")
            return None
        
    def process_single_detection(self, detection, team1_colour, team2_colour):
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            player_roi = self.current_frame[max(0, y1):min(y2, self.current_frame.shape[0]),
                                        max(0, x1):min(x2, self.current_frame.shape[1])]

            with ThreadPoolExecutor(max_workers=2) as executor:
                jersey_future = executor.submit(self.classify_jersey, self.current_frame, bbox)
                pose_future = executor.submit(self.estimate_pose, player_roi)

                jersey_colour, colour_conf = jersey_future.result()
                pose_results = pose_future.result()

            # Process pose if available
            pose_data = None
            orientation = None
            # Process pose results
            if pose_results and pose_results.get('pose_landmarks'):
                landmarks = []
                for landmark in pose_results['pose_landmarks'].landmark:
                    # Scale landmarks to original frame coordinates
                    scaled_landmark = ScaledLandmark(
                        x=x1 + landmark.x * (x2 - x1),
                        y=y1 + landmark.y * (y2 - y1),
                        z=landmark.z,
                        visibility=landmark.visibility
                    )
                    landmarks.append(scaled_landmark)
                pose_data = {
                    'landmarks': landmarks,
                    'confidence': pose_results['confidence']
                }
                # Calculate orientation
                orientation = self.get_player_orientation(landmarks)
                pose_data['orientation'] = orientation
            # Determine team
            if jersey_colour == team1_colour:
                team = 'team_1'
            elif jersey_colour == team2_colour:
                team = 'team_2'
            else:
                return None
            
            return {
                'bbox': bbox,
                'confidence': detection['conf'],
                'cls': detection['cls'],
                'jersey_colour': team,
                'colour_confidence': colour_conf,
                'pose': pose_data,
                'orientation': orientation,
                'manually_adjusted': False,
                'is_goalkeeper': detection.get('is_goalkeeper', False),
                'number': None  # Initialise number as None, can be set in player editor
            }
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return None

    def process_goalkeeper_detection(self, bbox, conf):
        try:
            pose_results = self.estimate_pose(self.current_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            jersey_colour, colour_conf = self.classify_jersey(self.current_frame, bbox)
            
            goalkeeper_data = {
                'bbox': bbox,
                'confidence': conf,
                'pose': pose_results,
                'jersey_colour': jersey_colour,
                'colour_confidence': colour_conf,
                'number': 1,
                'manually_adjusted': False
            }
            self.goalkeepers.append(goalkeeper_data)
            
        except Exception as e:
            print(f"Error processing goalkeeper: {str(e)}")

    def estimate_pose(self, player_img):
        if player_img.size == 0:
            return None
            
        try:
            results = self.pose.process(cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                return None
                
            # Return the results with calculated confidence
            return {
                'pose_landmarks': results.pose_landmarks,
                'confidence': self._calculate_pose_confidence(results.pose_landmarks.landmark)
            }   
        except Exception as e:
            print(f"Pose estimation error: {str(e)}")
            return None
    
    def _calculate_pose_confidence(self, landmarks):
        """Calculate average visibility of landmarks as confidence measure"""
        if not landmarks:
            return 0.0
        visibilities = [lm.visibility for lm in landmarks if hasattr(lm, 'visibility')]
        return sum(visibilities) / len(visibilities) if visibilities else 0.0

            
    def classify_jersey(self, frame, bbox):
        if self.jersey_classifier is None:
            print("Jersey classifier not loaded")
            return None, 0
            
        try:
            x1, y1, x2, y2 = map(int, bbox)
            player_img = frame[y1:y2, x1:x2]
            
            # Convert to RGB
            player_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(player_img)
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(pil_img).unsqueeze(0)
            
            with torch.no_grad():
                output = self.jersey_classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Print all probabilities
                print("\nAll colour probabilities:")
                for idx, prob in enumerate(probabilities):
                    print(f"{self.colour_classes[idx]}: {prob.item():.3f}")
                
                # Get team colour indices
                print(f"\nTeam colours: {self.team1_colour}, {self.team2_colour}")
                team1_idx = list(self.colour_classes.values()).index(self.team1_colour)
                team2_idx = list(self.colour_classes.values()).index(self.team2_colour)
                print(f"Team indices: {team1_idx}, {team2_idx}")
                
                team1_prob = probabilities[team1_idx].item()
                team2_prob = probabilities[team2_idx].item()
                print(f"Team probabilities: {team1_prob:.3f}, {team2_prob:.3f}")
                
                if team1_prob > team2_prob:
                    return self.team1_colour, team1_prob
                else:
                    return self.team2_colour, team2_prob
                    
        except Exception as e:
            print(f"Jersey classification error: {str(e)}\n{traceback.format_exc()}")
            return None, 0


    def process_player_detection(self, bbox, conf):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            player_roi = self.current_frame[y1:y2, x1:x2]
            if player_roi.size == 0:
                return
                
            rgb_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_roi)
            pose_data = None
            orientation = None

            if results.pose_landmarks:
                landmarks = []
                roi_h, roi_w = player_roi.shape[:2]
                
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    scaled_landmark = ScaledLandmark(
                        x=x1 + landmark.x * roi_w,
                        y=y1 + landmark.y * roi_h,
                        z=landmark.z,
                        visibility=landmark.visibility
                    )
                    landmarks.append(scaled_landmark)
                
                pose_data = {
                    'landmarks': landmarks,
                    'confidence': sum(lm.visibility for lm in landmarks) / len(landmarks)
                }
                
                # Get orientation from original landmarks
                ls = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                if ls.visibility > 0.35 and rs.visibility > 0.35:
                    shoulder_diff = ls.x - rs.x
                    if abs(shoulder_diff) < 0.1:
                        orientation = "front/back"
                    elif shoulder_diff > 0:
                        orientation = "facing left"
                    else:
                        orientation = "facing right"

            jersey_colour, colour_conf = self.classify_jersey(self.current_frame, bbox)
            
            # Assign to closest team colour if confidence is high enough
            if colour_conf > 0.001:
                if self.team1_colour is None or self.team2_colour is None:
                    print("Team colours not set")
                    return
                    
                # Get the nearest team colour
                team_colour = None
                if jersey_colour == self.team1_colour or jersey_colour in self.team1_colour:  # Handle substring matches
                    team_colour = self.team_colours['team_1']
                elif jersey_colour == self.team2_colour or jersey_colour in self.team2_colour:
                    team_colour = self.team_colours['team_2']
                else:
                    # Try fuzzy matching
                    sim1 = self.colour_similarity(jersey_colour, self.team1_colour)
                    sim2 = self.colour_similarity(jersey_colour, self.team2_colour)
                    team_colour = self.team_colours['team_1'] if sim1 > sim2 else self.team_colours['team_2']

                player_data = {
                    'bbox': bbox,
                    'confidence': conf,
                    'pose': pose_data,
                                'orientation': orientation,

                    'jersey_colour': jersey_colour,
                    'colour_confidence': colour_conf,
            'team_colour': self.team_colours['team_1'] if jersey_colour == self.team1_colour else self.team_colours['team_2'],
                    'manually_adjusted': False
                }
                self.players.append(player_data)
                
        except Exception as e:
            print(f"Error processing player: {str(e)}")

    def detect_number(self, frame, bbox):
        if self.number_classifier is None:
            return None, 0
            
        try:
            x1, y1, x2, y2 = bbox
            player_img = frame[y1:y2, x1:x2]
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(player_img).unsqueeze(0)
            
            with torch.no_grad():
                output = self.number_classifier(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                max_prob, pred = torch.max(probs, 0)
                
                number = pred.item() + 1  # Add 1 since labels start at 0
                return number, max_prob.item()
                
        except Exception as e:
            print(f"Number detection error: {str(e)}")
            return None, 0

    def colour_similarity(self, colour1, colour2):
        """Compare colour names for similarity"""
        colour1 = colour1.lower()
        colour2 = colour2.lower()
        
        # Direct match
        if colour1 == colour2:
            return 1.0
            
        # Substring match
        if colour1 in colour2 or colour2 in colour1:
            return 0.8
            
        # Basic colour groupings
        colour_groups = {
            'red': {'red', 'maroon', 'crimson'},
            'blue': {'blue', 'navy', 'azure'},
            'green': {'green', 'lime', 'forest'},
        }
        
        # Check if colours belong to same group
        for group in colour_groups.values():
            if colour1 in group and colour2 in group:
                return 0.6
                
        return 0.0

    def transform_coordinates(self):
        """Transform all coordinates using homography matrix"""
        if self.homography is None:
            return
            
        try:
            # Get template dimensions for bounds checking
            h, w = self.template_image.shape[:2]

            # Transform player positions
            for obj_list in [self.players, self.goalkeepers]:
                for obj in obj_list:
                    if 'bbox' in obj:
                        # Get centre point of bbox
                        x1, y1, x2, y2 = obj['bbox']
                        centre = np.array([[(x1 + x2)/2, y2]])  # Use bottom centre point
                        
                        # Transform using homography
                        transformed = cv2.perspectiveTransform(
                            centre.reshape(-1,1,2).astype(np.float32), 
                            self.homography
                        )
                        
                        # Check if point is within template bounds
                        tx, ty = transformed[0][0]
                        if 0 <= tx < w and 0 <= ty < h:
                            obj['transformed_position'] = (float(tx), float(ty))
                        else:
                            obj['transformed_position'] = None
            
                            
            # Transform ball position if exists
            if self.ball:
                x1, y1, x2, y2 = self.ball
                ball_centre = np.array([[(x1 + x2)/2, (y1 + y2)/2]])
                transformed = cv2.perspectiveTransform(
                    ball_centre.reshape(-1,1,2).astype(np.float32), 
                    self.homography
                )
                tx, ty = transformed[0][0]
                if 0 <= tx < w and 0 <= ty < h:
                    self.ball_transformed = (float(tx), float(ty))
                else:
                    self.ball_transformed = None

                
        except Exception as e:
            print(f"Coordinate transformation error: {str(e)}")

    def draw_template_annotations(self):
        """Draw all annotations on template"""
        if self.template_image is None or self.homography is None:
            return
            
        template_display = self.template_image.copy()
        
        # Draw players
        for player in self.players + self.goalkeepers:
            if player.get('transformed_position'):
                tx, ty = player['transformed_position']
                colour = (0, 255, 0) if player['jersey_colour'] == 'team_1' else (0, 0, 255)
                # Different marker for goalkeeper
                # Draw player marker
                cv2.circle(template_display, (int(tx), int(ty)), 
                          8 if player.get('is_goalkeeper') else 5, 
                          colour, -1, cv2.LINE_AA)
                
                # Draw jersey number
                if player.get('number'):
                    cv2.putText(template_display, str(player['number']),
                              (int(tx) + 5, int(ty) + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
                
                # Draw orientation if available
                if player.get('pose') and player.get('pose', {}).get('orientation'):
                    self.draw_orientation_arrow(template_display, tx, ty, 
                                             player['pose']['orientation'], colour)
        
        # Draw ball - Add null check
        if hasattr(self, 'ball_transformed') and self.ball_transformed is not None:
            try:
                tx, ty = self.ball_transformed
                # Draw ball with orange colour and slight transparency
                overlay = template_display.copy()
                cv2.circle(overlay, (int(tx), int(ty)), 4, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, (int(tx), int(ty)), 6, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.7, template_display, 0.3, 0, template_display)
            except Exception as e:
                print(f"Error drawing ball: {e}")
        return template_display

    def draw_orientation_arrow(self, img, x, y, orientation, colour):
        """Draw arrow indicating player orientation with proper error handling"""
        try:
            arrow_length = 15
            
            # Initialise end coordinates with default values
            end_x, end_y = x, y  # Default to same point
            
            # Set end coordinates based on orientation
            if orientation == "facing_left":
                end_x = x - arrow_length
                end_y = y
            elif orientation == "facing_right":
                end_x = x + arrow_length
                end_y = y
            elif orientation == "facing_up":
                end_x = x
                end_y = y - arrow_length
            elif orientation == "facing_down":
                end_x = x
                end_y = y + arrow_length
            elif orientation == "facing_front" or orientation == "facing_back":
                # Draw a circle for front/back facing
                cv2.circle(img, (int(x), int(y)), 
                        arrow_length//2, colour, 1, cv2.LINE_AA)
                return
            else:
                print(f"Unknown orientation: {orientation}")
                return
                
            # Draw the arrow
            cv2.arrowedLine(img, 
                        (int(x), int(y)), 
                        (int(end_x), int(end_y)),
                        colour, 2, cv2.LINE_AA, tipLength=0.3)
                        
        except Exception as e:
            print(f"Error drawing orientation arrow: {e}")

    def get_calibration_data(self):
        def convert_pose_data(pose_data):  
            if not pose_data:
                return None
                
            landmarks = pose_data.get('landmarks', [])
             
            orientation = self.get_player_orientation(landmarks) if landmarks else None
                
            valid_landmarks = []
            for lm in landmarks:
                if lm and all(hasattr(lm, attr) for attr in ('x', 'y', 'z', 'visibility')):
                    valid_landmarks.append({
                        'x': float(lm.x),
                        'y': float(lm.y),
                        'z': float(lm.z),
                        'visibility': float(lm.visibility)
                    })
                else:
                    valid_landmarks.append(None)
                    
            return {
                'landmarks': valid_landmarks,
                'confidence': float(pose_data.get('confidence', 0.0)),
                'orientation': orientation
            }

        calibration_data = {
            'players': [{
                'bbox': player['bbox'],
                'transformed_position': player.get('transformed_position'),
            'jersey_colour': player.get('jersey_colour', ''),
                'colour_confidence': float(player.get('colour_confidence', 0)),
                'number': player.get('number'),
            'pose': convert_pose_data(player.get('pose')) if player else None,
            'orientation': (player.get('pose', {}) or {}).get('orientation'),
            'manually_adjusted': bool(player.get('manually_adjusted', False))
            } for player in self.players],
            
            'goalkeepers': [{
                'bbox': gk['bbox'],
                'transformed_position': gk.get('transformed_position'),
                'jersey_colour': gk['jersey_colour'],
                'colour_confidence': float(gk.get('colour_confidence', 0)),
                'number': gk.get('number', 1),
              'pose': convert_pose_data(gk.get('pose')) if gk else None,
            'orientation': (gk.get('pose', {}) or {}).get('orientation'),
            'manually_adjusted': bool(gk.get('manually_adjusted', False))
            } for gk in self.goalkeepers]
        }

        calibration_data.update({
            'ball': {
                'bbox': self.ball,
                'transformed_position': getattr(self, 'ball_transformed', None),
                'manually_adjusted': 'ball' in self.manual_adjustments
            },
            'homography': self.homography.tolist() if self.homography is not None else None,
            'calibration_points': {
                'image_points': self.image_points,
                'template_points': self.template_points
            },
            'visible_area': self.get_visible_area()
        })
        
        return calibration_data
        
    def get_visible_area(self):
        if self.homography is None:
            return None
            
        try:
            h, w = self.current_frame.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1,1,2)
            transformed = cv2.perspectiveTransform(corners, self.homography)
            return transformed.reshape(-1,2).tolist()
        except Exception as e:
            print(f"Error getting visible area: {str(e)}")
            return None

    # - UI creation and management methods
    # - Manual adjustment handling
    # - Display methods
    # - Event handlers

    def create_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Button frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Process Frame", command=self.process_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Calibrate", command=self.calibrate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Points", command=self.clear_points).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Data", command=self.save_data).pack(side=tk.LEFT, padx=5)

        # Canvas frame
        canvas_frame = ttk.Frame(self.main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Image canvas
        self.image_canvas = tk.Canvas(canvas_frame, width=self.template_width, height=self.template_height)
        self.image_canvas.pack(side=tk.LEFT, padx=5, pady=5)
        self.image_canvas.bind("<Button-1>", self.on_image_click)

        # Template canvas
        self.template_canvas = tk.Canvas(canvas_frame, width=self.template_width, height=self.template_height)
        self.template_canvas.pack(side=tk.LEFT, padx=5, pady=5)
        self.template_canvas.bind("<Button-1>", self.on_template_click)

        # Manual adjustment controls
        self.manual_mode_button = ttk.Button(self.main_frame, text="Manual Adjustment Mode", 
                                           command=self.toggle_manual_mode)
        self.manual_mode_button.pack()

        self.manual_frame = ttk.Frame(self.main_frame)
        ttk.Radiobutton(self.manual_frame, text="None", variable=self.interaction_mode, 
                       value="none").pack(side=tk.LEFT)
        ttk.Radiobutton(self.manual_frame, text="Add Player", variable=self.interaction_mode, 
                       value="add_player").pack(side=tk.LEFT)
        ttk.Radiobutton(self.manual_frame, text="Add Ball", variable=self.interaction_mode, 
                       value="add_ball").pack(side=tk.LEFT)
        ttk.Radiobutton(self.manual_frame, text="Remove", variable=self.interaction_mode, 
                       value="remove").pack(side=tk.LEFT)
        ttk.Radiobutton(self.manual_frame, text="Adjust", variable=self.interaction_mode, 
                       value="adjust").pack(side=tk.LEFT)

        # Team selection
        ttk.Label(self.manual_frame, text="Team:").pack(side=tk.LEFT)
        ttk.Radiobutton(self.manual_frame, text="Team 1", variable=self.current_team, 
                       value="team_1").pack(side=tk.LEFT)
        ttk.Radiobutton(self.manual_frame, text="Team 2", variable=self.current_team, 
                       value="team_2").pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Edit Players", command=self.launch_player_editor).pack(side=tk.LEFT, padx=5)                 

        # Change team button
        self.change_team_button = ttk.Button(self.manual_frame, text="Change Team", 
                                           command=self.change_team)
        self.change_team_button.pack(side=tk.LEFT)

    def toggle_manual_mode(self):
        self.manual_mode = not self.manual_mode
        if self.manual_mode:
            self.manual_frame.pack(after=self.manual_mode_button)
            self.manual_mode_button.config(text="Exit Manual Mode")
            self.setup_manual_bindings()
        else:
            self.manual_frame.pack_forget()
            self.manual_mode_button.config(text="Enter Manual Mode")
            self.remove_manual_bindings()

    def setup_manual_bindings(self):
        self.image_canvas.bind("<ButtonPress-1>", self.on_manual_click)
        self.image_canvas.bind("<B1-Motion>", self.on_manual_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_manual_release)
        self.image_canvas.bind("<Double-Button-1>", self.on_double_click)

    def remove_manual_bindings(self):
        self.image_canvas.unbind("<ButtonPress-1>")
        self.image_canvas.unbind("<B1-Motion>")
        self.image_canvas.unbind("<ButtonRelease-1>")
        self.image_canvas.unbind("<Double-Button-1>")



    def on_manual_click(self, event):
        if not self.manual_mode:
            return

        x, y = event.x, event.y
        mode = self.interaction_mode.get()

        if mode == "add_player":
            self.add_player(x, y)
        elif mode == "add_ball":
            self.add_ball(x, y)
        elif mode == "remove":
            self.remove_object(x, y)
        elif mode == "adjust":
            self.start_adjust(x, y)
        elif mode == "change_team":
            self.start_team_change(x, y)
        elif mode == "change_number":
            self.change_player_number(x, y)

    def on_manual_drag(self, event):
        if not self.manual_mode or self.interaction_mode.get() != "adjust" or not self.dragging:
            return
        self.adjust_object(event.x, event.y)

    def on_manual_release(self, event):
        if not self.manual_mode or self.interaction_mode.get() != "adjust":
            return
        self.finish_adjust()

    def edit_player(self, player_data):
            editor = PlayerEditor(self.master, player_data, self.current_frame)
            self.master.wait_window(editor.window)
            self.display_processed_frame()

    def on_double_click(self, event):
        clicked_obj = self.find_clicked_object(event.x, event.y)
        if clicked_obj:
            self.change_number(clicked_obj)


    def change_number(self, obj):
        current_number = obj.get('number')
        new_number = simpledialog.askinteger("Change Number", 
                                           "Enter new number (0 to remove):",
                                           minvalue=0, maxvalue=99, 
                                           initialvalue=current_number or 0)
        if new_number is not None:
            obj['number'] = new_number if new_number > 0 else None
            obj['manually_adjusted'] = True
            self.display_processed_frame()

    def add_player(self, x, y):
        avg_width, avg_height = self.calculate_average_player_size()
        bbox = (x - avg_width//2, y - avg_height//2, 
            x + avg_width//2, y + avg_height//2)

        # Get team colour based on current team selection
        team = self.current_team.get()
        colour = self.team_colours.get(team, (128, 128, 128))
        
        player = {
            'bbox': bbox,
            'confidence': 1.0,
            'jersey_colour': self.team1_colour if team == 'team_1' else self.team2_colour,
            'colour_confidence': 1.0,
            'number': None,
            'team_colour': colour,
            'manually_adjusted': True
        }
        
        self.players.append(player)
        self.transform_coordinates()
        self.display_processed_frame()

    def add_ball(self, x, y):
        size = 20
        self.ball = (x - size//2, y - size//2, x + size//2, y + size//2)
        self.manual_adjustments.add('ball')
        self.transform_coordinates()
        self.display_processed_frame()

    def remove_object(self, x, y):
        obj = self.find_clicked_object(x, y)
        if obj:
            if obj in self.players:
                self.players.remove(obj)
            elif obj in self.goalkeepers:
                self.goalkeepers.remove(obj)
            self.display_processed_frame()
        elif self.ball and self.point_in_bbox(x, y, self.ball):
            self.ball = None
            self.display_processed_frame()

    def change_team(self):
        if self.selected_object and isinstance(self.selected_object, dict):
            new_team = self.current_team.get()
            if new_team == 'team_1':
                self.selected_object['jersey_colour'] = self.team1_colour
                self.selected_object['team_colour'] = self.team_colours['team_1']
            else:
                self.selected_object['jersey_colour'] = self.team2_colour
                self.selected_object['team_colour'] = self.team_colours['team_2']
            self.selected_object['manually_adjusted'] = True
            self.display_processed_frame()


    def get_centre_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_bbox_width(self, bbox):
        x1, y1, x2, y2 = bbox
        return x2 - x1
    
    def draw_ellipse(self, frame, bbox, colour, number=None):
        x1, y1, x2, y2 = map(int, bbox)
        centre_x = (x1 + x2) // 2
        width = x2 - x1
        height = y2 - y1

        cv2.ellipse(frame, (centre_x, y2),
                    (width//2, height//3), 0, 0, 360, colour, 2)
                    
        if number:
            cv2.putText(frame, str(number), 
                        (x1 + 5, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            
    def display_processed_frame(self):
        if self.current_frame is None:
            return

        display_frame = self.current_frame.copy()

        # Draw players
        for player in self.players + self.goalkeepers:
            # Use team_colour for display
            if 'team_colour' in player:
                colour = player['team_colour']
            else:
                if player['jersey_colour'] == 'team_1':
                    colour = self.team_colours['team_1']
                elif player['jersey_colour'] == 'team_2':
                    colour = self.team_colours['team_2']
                else:
                    print(f"Unknown jersey colour: {player['jersey_colour']}")
                    continue

            bbox = player['bbox']
            self.draw_ellipse(display_frame, bbox, colour, player.get('number'))

            
            # Draw pose if available
            if 'pose' in player and player['pose']:
                self.draw_pose(display_frame, player['pose']['landmarks'], colour)

        # Draw ball
        if self.ball:
            cv2.circle(display_frame, 
                    (int((self.ball[0] + self.ball[2])/2), 
                    int((self.ball[1] + self.ball[3])/2)),
                    10, (0, 165, 255), 2)

        self.display_image(display_frame, self.image_canvas)
        
        # Update template with visible area
        if self.calibrated and self.homography is not None:
            template_with_overlay = self.template_image.copy()
            h, w = self.current_frame.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, self.homography).reshape(-1, 2)
            
            overlay = template_with_overlay.copy()
            cv2.fillPoly(overlay, [np.int32(transformed_corners)], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, template_with_overlay, 0.7, 0, template_with_overlay)
            
            self.display_image(template_with_overlay, self.template_canvas)

    def draw_pose(self, frame, landmarks, colour):
        if not landmarks:
            return

        # Define the BODY_25 keypoint pairs for skeleton
        pose_pairs = [
            # Body
            (1, 0), (1, 2), (1, 5), (2, 3), (3, 4),   # Torso & Head
            (5, 6), (6, 7),                            # Left Arm
            (1, 8), (8, 9), (9, 10),                   # Left Leg
            (1, 11), (11, 12), (12, 13),               # Right Leg
            (1, 14), (14, 15),                         # Right Arm
            (0, 15), (15, 17),                         # Shoulders
            (0, 16), (16, 18),                         # Shoulders
            (14, 21), (19, 21), (21, 23),             # Right Hand refinement
            (11, 22), (20, 22), (22, 24)              # Left Hand refinement
        ]

        # Draw connections
        for pair in pose_pairs:
            if (pair[0] < len(landmarks) and pair[1] < len(landmarks) and 
                landmarks[pair[0]] and landmarks[pair[1]]):
                pt1 = landmarks[pair[0]]
                pt2 = landmarks[pair[1]]
                
                if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                    p1 = (int(pt1.x), int(pt1.y))
                    p2 = (int(pt2.x), int(pt2.y))
                    
                    # Draw anti-aliased line with thickness
                    cv2.line(frame, p1, p2, colour, 2, cv2.LINE_AA)
                    
                    # Draw joints as circles
                    cv2.circle(frame, p1, 3, colour, -1, cv2.LINE_AA)
                    cv2.circle(frame, p2, 3, colour, -1, cv2.LINE_AA)

    def colour_distance(self, colour1, colour2):
        # Simple RGB distance
        if isinstance(colour1, str):
            colour1 = self.colour_to_rgb(colour1)
        if isinstance(colour2, str):
            colour2 = self.colour_to_rgb(colour2)
        return sum((a - b) ** 2 for a, b in zip(colour1, colour2)) ** 0.5

    def colour_to_rgb(self, colour_name):
        # Basic colour map
        colour_map = {
            'black': (0, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'maroon': (128, 0, 0),
            'orange': (255, 165, 0),
            'red': (255, 0, 0),
            'white': (255, 255, 255),
            'yellow': (255, 255, 0)
        }
        return colour_map.get(colour_name, (128, 128, 128))
    def display_image(self, img, canvas):
        if img is None:
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(image=img_pil)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo

    def save_data(self):
        calibration_data = {
            'players': self.players,
            'goalkeepers': self.goalkeepers,
            'ball': self.ball,
            'homography': self.homography.tolist() if self.homography is not None else None,
            'calibration_points': {
                'image_points': self.image_points,
                'template_points': self.template_points
            },
            'visible_area': self.get_visible_area()
        }
        self.data_collector.save_training_instance(
            self.current_frame,
            calibration_data,
            video_source=self.video_path
        )
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(calibration_data, f, cls=PoseEncoder, indent=2)
                messagebox.showinfo("Success", "Data saved successfully")
            except Exception as e:
                print(f"Save error: {str(e)}")
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    @lru_cache(maxsize=128)
    def calculate_average_player_size(self):
        if not self.players:
            return 40, 80  # Default size
            
        widths = []
        heights = []
        for player in self.players:
            bbox = player['bbox']
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
            
        avg_width = int(sum(widths) / len(widths))
        avg_height = int(sum(heights) / len(heights))
        return avg_width, avg_height
    
    def calibrate(self):
        if len(self.image_points) != len(self.template_points) or len(self.image_points) < 4:
            messagebox.showerror("Error", "Please select at least 4 corresponding points on both images.")
            return
            
        try:
            src_points = np.array(self.image_points, dtype=np.float32)
            dst_points = np.array(self.template_points, dtype=np.float32)
            
            self.homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            
            if self.homography is None:
                messagebox.showerror("Error", "Failed to compute homography. Please try different points.")
                return
                
            self.calibrated = True
            self.transform_coordinates()
            self.display_processed_frame()
            messagebox.showinfo("Success", "Calibration completed successfully.")
            
        except Exception as e:
            print(f"Calibration error: {str(e)}")
            messagebox.showerror("Error", "Calibration failed. Please try again.")

    def on_image_click(self, event):
        if self.current_frame is None:
            return
        
        if self.manual_mode:
            self.on_manual_click(event)
            return
            
        x, y = event.x, event.y
        self.image_points.append((x, y))
        self.image_canvas.create_oval(x-3, y-3, x+3, y+3, fill='red')

    def on_template_click(self, event):
        if self.template_image is None:
            return
            
        x, y = event.x, event.y
        self.template_points.append((x, y))
        self.template_canvas.create_oval(x-3, y-3, x+3, y+3, fill='blue')
    
    def clear_points(self):
        self.image_points = []
        self.template_points = []
        self.image_canvas.delete("all")
        self.template_canvas.delete("all")
        self.display_image(self.current_frame, self.image_canvas)
        self.display_template()

    def start_adjust(self, x, y):
        for obj_list in [self.players, self.goalkeepers]:
            for obj in obj_list:
                if self.point_in_bbox(x, y, obj['bbox']):
                    self.selected_object = obj
                    # Set current team radio button based on selected player's team
                    if obj['jersey_colour'] == self.team1_colour:
                        self.current_team.set('team_1')
                    else:
                        self.current_team.set('team_2')
                    self.dragging = True
                    self.drag_start = (x, y)
                    return

        if self.ball and self.point_in_bbox(x, y, self.ball):
            self.selected_object = 'ball'
            self.dragging = True
            self.drag_start = (x, y)

    def change_player_number(self, x, y):
        """Change player number"""
        clicked_player = None
        
        # Find clicked player
        for player in self.players + self.goalkeepers:
            if self.point_in_bbox(x, y, player):
                clicked_player = player
                break
                
        if clicked_player:
            current_number = clicked_player.get('player_number')
            new_number = simpledialog.askinteger("Change Number", 
                                               "Enter new player number (1-99):",
                                               minvalue=1, maxvalue=99,
                                               initialvalue=current_number)
            if new_number is not None:
                clicked_player['player_number'] = new_number
                clicked_player['manually_adjusted'] = True
                self.manual_adjustments.add(f'number_change_{id(clicked_player)}')
                self.display_processed_frame()

    def start_team_change(self, x, y):
        """Start team change process"""
        clicked_player = None
        
        # Find clicked player
        for player in self.players + self.goalkeepers:
            if self.point_in_bbox(x, y, player):
                clicked_player = player
                break
                
        if clicked_player:
            # Toggle team
            current_team = clicked_player.get('jersey_colour')
            new_team = 'team_2' if current_team == 'team_1' else 'team_1'
            clicked_player['jersey_colour'] = new_team
            clicked_player['manually_adjusted'] = True
            self.manual_adjustments.add(f'team_change_{id(clicked_player)}')
            self.display_processed_frame()

    def adjust_object(self, x, y):
        if not self.selected_object:
            return

        dx = x - self.drag_start[0]
        dy = y - self.drag_start[1]

        if self.selected_object == 'ball':
            x1, y1, x2, y2 = self.ball
            self.ball = (x1+dx, y1+dy, x2+dx, y2+dy)
            self.manual_adjustments.add('ball')
        else:
            x1, y1, x2, y2 = self.selected_object['bbox']
            self.selected_object['bbox'] = (x1+dx, y1+dy, x2+dx, y2+dy)
            self.selected_object['manually_adjusted'] = True

        self.drag_start = (x, y)
        self.transform_coordinates()
        self.display_processed_frame()

    def finish_adjust(self):
        self.dragging = False
        self.selected_object = None
        self.drag_start = None

    def find_clicked_object(self, x, y):
        for obj in self.players + self.goalkeepers:
            if self.point_in_bbox(x, y, obj['bbox']):
                return obj
        return None

    def point_in_bbox(self, x, y, bbox):
        x1, y1, x2, y2 = bbox
        return x1 <= x < x2 and y1 <= y < y2

    def load_frame(self, frame):
        if frame is not None:
            self.current_frame = cv2.resize(frame, (self.template_width, self.template_height))
            self.display_image(self.current_frame, self.image_canvas)
            self.processed = False
            self.calibrated = False
            self.clear_points()

    def display_template(self):
        if self.template_image is not None:
            self.display_image(self.template_image, self.template_canvas)
            
    def launch_player_editor(self):
        if not self.players:
            messagebox.showwarning("No Players", "No players detected to edit")
            return
            
        editor = PlayerEditor(
            self.master, 
            self.players,
            self.current_frame,
            self.template_image,
            self.team1_colour,
            self.team2_colour,
            parent=self
        )
        self.master.wait_window(editor.window)
        self.display_processed_frame()

    if __name__ == "__main__":
        root = tk.Tk()
        app = GaelicFootballCalibrator(root)
        root.mainloop()