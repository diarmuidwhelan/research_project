import cv2 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from calibrator import GaelicFootballCalibrator
import json
import numpy as np
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
# from actionrecogniser_integrated import ActionRecognitionModel
import torch.nn as nn
from torch import device
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt

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


@dataclass
class TrainingDataPoint:
    frame_path: str
    calibration_data: dict
    field_registration_points: dict
    timestamp: str
    video_source: str


class ModelTrainingDataCollector:
    def __init__(self, base_dir="training_data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Create organised directory structure for different training purposes"""
        directories = {
            'player_detection': ['images', 'annotations'],
            'ball_detection': ['images', 'annotations'],
            'colour_classification': ['images', 'labels'],
            'number_recognition': ['images', 'labels'],
            'pose_estimation': ['images', 'keypoints'],
            'field_registration': ['images', 'keypoints', 'homography_matrices'],
            'action_recognition': ['images', 'labels']
        }
        
        for model, subdirs in directories.items():
            model_dir = self.base_dir / model
            for subdir in subdirs:
                (model_dir / subdir).mkdir(parents=True, exist_ok=True)
   


    def save_training_instance(self, frame, calibration_data, video_source):
        """Save a complete training instance with frame and all annotations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save frame for each model's training
        frame_filename = f"frame_{timestamp}.jpg"
        
        # Save data for player detection
        self._save_player_detection_data(frame, frame_filename, calibration_data, timestamp)
        
        # Save data for ball detection
        self._save_ball_detection_data(frame, frame_filename, calibration_data, timestamp)
        
        # Save field registration data
        self._save_field_registration_data(frame, calibration_data, timestamp)
        
        # Save complete training instance metadata
        self._save_metadata(frame_filename, calibration_data, video_source, timestamp)

    def _save_player_detection_data(self, frame, frame_filename, calibration_data, timestamp):
       #save image to the appropriate directory
        image_path = self.base_dir / "player_detection" / "images" / frame_filename
        cv2.imwrite(str(image_path), frame)
        
        # Convert player boxes to YOLO format
        yolo_annotations = []
        h, w = frame.shape[:2]
        #loop through players and append bounding box coordinates
        for player in calibration_data['players']:
            bbox = player['bbox']
            x_centre = ((bbox[0] + bbox[2]) / 2) / w
            y_centre = ((bbox[1] + bbox[3]) / 2) / h
            width = (bbox[2] - bbox[0]) / w
            height = (bbox[3] - bbox[1]) / h
            class_id = '1' if player.get('is_goalkeeper', False) else '0'
            yolo_annotations.append(f"{class_id} {x_centre} {y_centre} {width} {height}")
        #loop through goalkeeper and append bounding box coordinates
        for gk in calibration_data.get('goalkeepers', []):
            bbox = gk['bbox']
            x_centre = ((bbox[0] + bbox[2]) / 2) / w
            y_centre = ((bbox[1] + bbox[3]) / 2) / h
            width = (bbox[2] - bbox[0]) / w
            height = (bbox[3] - bbox[1]) / h
            yolo_annotations.append(f"1 {x_centre} {y_centre} {width} {height}")
        #save data to file
        if yolo_annotations:
            annotation_path = self.base_dir / "player_detection/annotations" / f"{timestamp}.txt"
            with open(annotation_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

    def _save_ball_detection_data(self, frame, frame_filename, calibration_data, timestamp):
              #save image to the appropriate directory
        if calibration_data.get('ball', {}).get('bbox'):
            image_path = self.base_dir / "ball_detection/images" / frame_filename
            cv2.imwrite(str(image_path), frame)
            #append bounding box coordinates
            bbox = calibration_data['ball']['bbox']
            h, w = frame.shape[:2]
            x_centre = ((bbox[0] + bbox[2]) / 2) / w
            y_centre = ((bbox[1] + bbox[3]) / 2) / h
            width = (bbox[2] - bbox[0]) / w
            height = (bbox[3] - bbox[1]) / h
            
            #save data to file
            annotation_path = self.base_dir / "ball_detection/annotations" / f"{timestamp}.txt"
            with open(annotation_path, 'w') as f:
                f.write(f"2 {x_centre} {y_centre} {width} {height}")

    def _save_field_registration_data(self, frame, calibration_data, timestamp):
         #save image to the appropriate directory
        if calibration_data.get('homography') is not None:
            image_path = self.base_dir / "field_registration/images" / f"{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)
         #save keypoint coordinates to the appropriate directory
            keypoints_path = self.base_dir / "field_registration/keypoints" / f"{timestamp}.json"
            with open(keypoints_path, 'w') as f:
                json.dump(calibration_data['calibration_points'], f)
             #save homography matrix to the appropriate directory
            matrix_path = self.base_dir / "field_registration/homography_matrices" / f"{timestamp}.npy"
            np.save(matrix_path, np.array(calibration_data['homography']))

    def _save_metadata(self, frame_filename, calibration_data, video_source, timestamp):
        #create metadata
        metadata = TrainingDataPoint(
            frame_path=frame_filename,
            calibration_data=calibration_data,
            field_registration_points=calibration_data['calibration_points'],
            timestamp=timestamp,
            video_source=video_source
        )
         #save metadata to the appropriate directory

        metadata_path = self.base_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, cls=NumpyJSONEncoder, indent=2)

    def get_dataset_statistics(self):
        """Return statistics about collected training data"""
        stats = {}
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                stats[model_dir.name] = {
                    'total_samples': len(list(model_dir.glob('images/*'))),
                    'last_updated': max((f.stat().st_mtime for f in model_dir.glob('images/*')), default=0)
                }
        return stats

@dataclass
class ScaledLandmark:
    x: float
    y: float
    z: float
    visibility: float

    
class ActionRecognitionModel(nn.Module):
    """3D CNN for Gaelic football action recognition"""
    def __init__(self, num_classes=4):
        super(ActionRecognitionModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(6, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),
            
            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2)),
            
            nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )
        # Add adaptive pooling to get fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        
    def forward(self, x):
        # Forward pass with architecture matching saved state
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x    

class FlowComputer:
    def __init__(self):
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def compute_flow(self, frame1, frame2):
        grey1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        grey2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            grey1, grey2, None, **self.flow_params)
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_rgb   
class ActionRecognitionIntegrator:
    """Helper class to manage action recognition integration"""
    def __init__(self, model_path="/Users/diarmuidwhelan/Downloads/research_project/2024_12_24.pth"):
        self.frame_buffer = []
        self.buffer_size = 16
        self.model = None
        self.transform = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])        
        self.flow_computer = FlowComputer()  # Add this line
        self.setup_model(model_path)
        
        # Action mapping from model outputs to event names
        self.action_map = {
        0: 'hand passes',
         1: 'foot passes',
         2:'carries',
         3:'Tackles'
        }
                # Add training data collection paths
        self.training_data_path = Path("training_data/action_recognition")
        self.images_path = self.training_data_path / "images"
        self.labels_path = self.training_data_path / "labels"
        
        # Create directories if they don't exist
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.labels_path.mkdir(parents=True, exist_ok=True)

        self.last_detected_frame = 0
        self.detection_cooldown = 30  # Frames to wait before next detection
        self.preprocessed_buffer = []
        self.class_cooldowns = {1: 30}  # Longer cooldown for foot passes
        self.last_class_detection = {1: 0}  # Track last detection frame for specific classes

    def save_training_frames(self, frame_buffer, event_name, timestamp, confidence=None):
        """
        Save frame sequence for model retraining with metadata
        
        Args:
            frame_buffer: List of frames capturing the action
            event_name: Name of the detected/tagged action
            timestamp: Unique timestamp identifier
            confidence: Confidence score for automatically detected events
        """
        try:
            # Create event-specific directory
            event_dir = self.images_path / event_name
            event_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each frame in the sequence
            for i, frame in enumerate(frame_buffer):
                frame_path = event_dir / f"{timestamp}_{i:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
            
            # Save metadata including detection method and confidence
            metadata = {
                'event_name': event_name,
                'timestamp': timestamp,
                'frame_count': len(frame_buffer),
                'detection_method': 'automatic' if confidence else 'manual',
                'confidence': confidence,
                'date_collected': datetime.now().isoformat()
            }
            
            metadata_path = self.labels_path / f"{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error saving training frames: {e}")
            return False

    def setup_model(self, model_path):
        """Initialise the action recognition model and transforms"""
        try:
            # Initialise model 
            self.model = ActionRecognitionModel()
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)  
            self.model.eval()
            return True
        except Exception as e:
            print(f"Failed to initialize action recognition: {e}")
            return False
        
    def preprocess_frame(self, frame):
        """Preprocess frame once when added to buffer"""
        if frame is None:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(rgb_frame)       

    def update_buffer(self, frame):
        """Update frame buffer with new frame"""
        if frame is not None:
            if len(self.frame_buffer) >= self.buffer_size:
                self.frame_buffer.pop(0)
            self.frame_buffer.append(frame.copy())

    
    def predict_action(self, current_frame_num):
       """Predict action from frame buffer"""
       if (current_frame_num - self.last_detected_frame) < self.detection_cooldown:
           return None

       if len(self.frame_buffer) < self.buffer_size:
           return None
               
       try:
           # Sample frames uniformly
           indices = np.linspace(0, len(self.frame_buffer)-1, 16, dtype=int)
           rgb_frames = []
           flow_frames = []
           
           # Process frames
           for i in range(len(indices)-1):
               frame1 = cv2.cvtColor(self.frame_buffer[indices[i]], cv2.COLOR_BGR2RGB)
               frame2 = cv2.cvtColor(self.frame_buffer[indices[i+1]], cv2.COLOR_BGR2RGB)
               
               # Compute optical flow
               flow = self.flow_computer.compute_flow(frame1, frame2)
               
               # Transform frames
               frame_tensor = self.transform(Image.fromarray(frame1))
               flow_tensor = self.transform(Image.fromarray(flow))
               
               rgb_frames.append(frame_tensor)
               flow_frames.append(flow_tensor)

           # Stack frames and prepare sequence
           rgb_sequence = torch.stack(rgb_frames)  # [T, C, H, W]
           flow_sequence = torch.stack(flow_frames)  # [T, C, H, W]
           sequence = torch.cat([rgb_sequence, flow_sequence], dim=1)  # [T, 6, H, W]
           sequence = sequence.unsqueeze(0)  # Add batch dimension [1, T, 6, H, W]
           sequence = sequence.permute(0, 2, 1, 3, 4)  # [1, 6, T, H, W]

           # Make prediction
           with torch.no_grad():
               output = self.model(sequence)
               prob = F.softmax(output, dim=1)
               pred = output.argmax(1).item()
               confidence = prob[0][pred].item()

               if confidence > 0.5:
                   self.last_detected_frame = current_frame_num
                   return self.action_map[pred], confidence
               
       except Exception as e:
           print(f"Action prediction failed: {e}")
       return None


class VideoEventTimeline:
    def __init__(self, master, width=800, height=100):
        """ create a timeline to visualise the events in a video"""
        self.frame = ttk.Frame(master)
        self.frame.pack(fill=tk.X, pady=5)
        
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg='white')
        self.canvas.pack(fill=tk.X)
        
        self.width = width
        self.height = height
        self.events = []
        self.total_frames = 0
        self.selected_event = None
        
        self.canvas.bind('<Button-1>', self.on_click)
        self.tooltip = None
        
        # Event type colours
        self.event_colours = {
            'kickout': '#FFB6C1',
            'point': '#98FB98',
            'goal': '#87CEEB',
            'pass': '#DDA0DD',
            'carry': '#F0E68C',
            'block': '#CD853F',
            'tackle': '#FA8072',
            'free': '#B0C4DE',
            'save': '#20B2AA'
        }

    def update_timeline(self, events, total_frames):
        self.events = events
        self.total_frames = total_frames
        self.selected_event = None
        self.draw_timeline()

    def draw_timeline(self):
        self.canvas.delete('all')
        
        # Draw base timeline
        self.canvas.create_line(50, self.height/2, self.width-50, self.height/2, 
                              width=2, fill='gray')
        
        if not self.total_frames:
            return
            
        # Draw time markers
        for i in range(0, self.total_frames, self.total_frames//10):
            x = self.frame_to_x(i)
            self.canvas.create_line(x, self.height/2-10, x, self.height/2+10, 
                                  fill='gray')
            self.canvas.create_text(x, self.height-10, 
                                  text=f'{i//30//60:02d}:{(i//30)%60:02d}')

        # Draw events
        for event in self.events:
            x = self.frame_to_x(event['start_frame'])
            end_x = self.frame_to_x(event['end_frame'])
            
            # Determine colour based on event type
            colour = self.get_event_colour(event['event_name'])
            
            # Draw event marker
            event_height = 20
            marker = self.canvas.create_rectangle(x, self.height/2-event_height/2,
                                               end_x, self.height/2+event_height/2,
                                               fill=colour, outline='darkgray',
                                               tags=('event', str(event['event_number'])))
            
            # Add invisible wider area for hover detection
            self.canvas.tag_bind(marker, '<Enter>', 
                               lambda e, ev=event: self.show_tooltip(e, ev))
            self.canvas.tag_bind(marker, '<Leave>', self.hide_tooltip)

    def get_event_colour(self, event_name):
        for key, colour in self.event_colours.items():
            if key in event_name.lower():
                return colour
        return '#D3D3D3'  # Default grey

    def frame_to_x(self, frame):
        if self.total_frames == 0:
            return 50
        usable_width = self.width - 100  # Account for margins
        return 50 + (frame / self.total_frames * usable_width)

    def x_to_frame(self, x):
        usable_width = self.width - 100
        scaled_x = x - 50
        if scaled_x <= 0:
            return 0
        if scaled_x >= usable_width:
            return self.total_frames
        return int((scaled_x / usable_width) * self.total_frames)

    def show_tooltip(self, event, event_data):
        if self.tooltip:
            self.hide_tooltip(None)
            
        x = self.canvas.winfo_rootx() + event.x
        y = self.canvas.winfo_rooty() + event.y
        
        self.tooltip = tk.Toplevel()
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x+10}+{y+10}")
        
        text = (f"Event {event_data['event_number']}\n"
               f"Type: {event_data['event_name']}\n"
               f"Team: {event_data['team']}\n"
               f"Player: {event_data['player_number']}")
        
        label = ttk.Label(self.tooltip, text=text, background="#ffffe0", 
                         relief='solid', padding=2)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def on_click(self, event):
        x = event.x
        frame = self.x_to_frame(x)
        
        # Find clicked event
        for event_widget in self.canvas.find_withtag('event'):
            coords = self.canvas.coords(event_widget)
            if coords[0] <= x <= coords[2]:
                event_id = self.canvas.gettags(event_widget)[1]
                for event in self.events:
                    if str(event['event_number']) == event_id:
                        self.selected_event = event
                        self.highlight_selected_event(event_widget)
                        break
                    
    def highlight_selected_event(self, widget_id):
        # Reset previous selections
        for item in self.canvas.find_withtag('event'):
            self.canvas.itemconfig(item, width=1)
        # Highlight new selection
        self.canvas.itemconfig(widget_id, width=3)

class VideoEventTagger:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Event Tagger")
        self.master.geometry("1200x800")
        self.canvas_width = 800
        self.canvas_height = 450
        # Add template dimensions
        self.template_width = 600
        self.template_height = 450
        self.video_path = None
        self.cap = None
        self.current_frame = None 
        self.current_frame_number = 0
        self.total_frames = 0
        self.events = []
        self.slider_update_flag = False
        self.is_playing = False
        self.play_id = None
        self.playback_speeds = [1.0, 1.5, 2.0, 5.0, 10.0]
        self.current_speed_index = 0
        self.playback_speed = self.playback_speeds[self.current_speed_index]
        self.frame_interval = 33  # Default frame interval (approx. 30 FPS)

        self.possession_team = tk.StringVar(value="Team 1")
        self.current_half = tk.StringVar(value="1")
        self.event_counter = 1
        self.deleted_events = set()
        self.next_event_number = 1
        self.team1_colour = tk.StringVar() 
        self.team2_colour = tk.StringVar()
        self.team_colours_set = False
        
        self.specific_events = [
            "kickout won", "kickout lost", "point from play", "point from free",
            "goal from play", "goal from free", "miss from play", "miss from free", 
            "carry", "hand pass", "foot pass", "block", "challenge", "ball recovery",
            "Tackle", "Free", "Free conceded", "sideline", "save"
        ]
        self.data_collector = ModelTrainingDataCollector()
        self.frame_buffer = []
        self.buffer_size = 16
        self.action_model = None
        self.transform = None
        self.action_recogniser = ActionRecognitionIntegrator()
        if not hasattr(self.action_recogniser, 'model') or self.action_recogniser.model is None:
            messagebox.showwarning("Warning", "Action recognition model failed to load")
        self.create_ui()
        self.bind_shortcuts()

    def handle_manual_event(self, event_name):
        """Handler for manual event tagging buttons"""
        self.is_manual_tagging = True
        self.tag_event(event_name)
        self.is_manual_tagging = False
        
    def create_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.master, padding="5")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Video display canvas
        self.video_canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height)
        self.video_canvas.pack(pady=5)
        
        # Team colour selection
        colour_frame = ttk.Frame(self.main_frame)
        colour_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(colour_frame, text="Team 1 Colour:").pack(side=tk.LEFT)
        team1_colour_combo = ttk.Combobox(colour_frame, textvariable=self.team1_colour,
                                       values=['black', 'blue', 'green', 'maroon', 'orange', 'red', 'white', 'yellow'],
                                       state='readonly', width=10)
        team1_colour_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(colour_frame, text="Team 2 Colour:").pack(side=tk.LEFT)
        team2_colour_combo = ttk.Combobox(colour_frame, textvariable=self.team2_colour,
                                       values=['black', 'blue', 'green', 'maroon', 'orange', 'red', 'white', 'yellow'],
                                       state='readonly', width=10)
        team2_colour_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(colour_frame, text="Set Team Colours", command=self.set_team_colours).pack(side=tk.LEFT, padx=5)

        # Video controls frame
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        button_width = 8
        ttk.Button(controls_frame, text="Load", command=self.load_video, width=button_width).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="◀◀", command=self.rewind, width=button_width).pack(side=tk.LEFT, padx=2)
        self.play_pause_button = ttk.Button(controls_frame, text="▶", command=self.play_pause, width=button_width)
        self.play_pause_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="▶▶", command=self.fast_forward, width=button_width).pack(side=tk.LEFT, padx=2)
        self.speed_button = ttk.Button(controls_frame, text="1.0x", command=self.toggle_speed, width=button_width)
        self.speed_button.pack(side=tk.LEFT, padx=2)
        
        self.video_slider = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.video_slider.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.frame_counter_label = ttk.Label(controls_frame, text="Frame: 0 / 0 | 00:00:00")
        self.frame_counter_label.pack(side=tk.RIGHT, padx=5)

        # Team and half selection frame
        team_frame = ttk.Frame(self.main_frame)
        team_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(team_frame, text="Team:").pack(side=tk.LEFT)
        ttk.Radiobutton(team_frame, text="1", variable=self.possession_team, value="Team 1").pack(side=tk.LEFT)
        ttk.Radiobutton(team_frame, text="2", variable=self.possession_team, value="Team 2").pack(side=tk.LEFT)
        
        ttk.Label(team_frame, text="Half:").pack(side=tk.LEFT, padx=(20,0))
        for i in range(1, 5):
            ttk.Radiobutton(team_frame, text=str(i), variable=self.current_half, value=str(i)).pack(side=tk.LEFT)

        # Event buttons frame  
        event_buttons_frame = ttk.Frame(self.main_frame)
        event_buttons_frame.pack(fill=tk.X, pady=2)
        
        num_columns = 6
        for i, event in enumerate(self.specific_events):
            row = i // num_columns
            col = i % num_columns
            ttk.Button(event_buttons_frame, text=event, command=lambda e=event: self.handle_manual_event(e),
                    width=10).grid(row=row, column=col, padx=1, pady=1, sticky='nsew')


        # Timeline Section
        # Timeline and controls in compact frame
        timeline_frame = ttk.LabelFrame(self.main_frame, text="Timeline & Controls")
        timeline_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Timeline
        self.timeline = VideoEventTimeline(timeline_frame, height=80)  # Reduced height


        # Event Control Buttons
        # Control buttons in single row
        controls = ttk.Frame(timeline_frame)
        controls.pack(fill=tk.X, padx=2, pady=2)
        
        for text, cmd in [("Delete", self.delete_tag), 
                        ("Re-edit", self.reedit_tag), 
                        ("Save", self.save_data)]:
            ttk.Button(controls, text=text, command=cmd, 
                    width=8).pack(side=tk.LEFT, padx=2)
            
    def set_team_colours(self):
        if not self.team1_colour.get() or not self.team2_colour.get():
            messagebox.showerror("Error", "Please select colours for both teams")
            return
            
        if self.team1_colour.get() == self.team2_colour.get():
            messagebox.showerror("Error", "Teams must have different colours")
            return
            
        self.team_colours_set = True
        messagebox.showinfo("Success", f"Team colours set:\nTeam 1: {self.team1_colour.get()}\nTeam 2: {self.team2_colour.get()}")

    def bind_shortcuts(self):
        # Video control shortcuts
        self.master.bind('<space>', lambda event: self.play_pause())
        self.master.bind('<Left>', lambda event: self.rewind())
        self.master.bind('<Right>', lambda event: self.fast_forward())
        
        # Event shortcuts
        self.master.bind('c', lambda event: self.tag_event('carry'))
        self.master.bind('t', lambda event: self.tag_event('Tackle'))
        self.master.bind('kl', lambda event: self.tag_event('kickout lost'))
        self.master.bind('kw', lambda event: self.tag_event('kickout won'))
        self.master.bind('hp', lambda event: self.tag_event('hand pass'))
        self.master.bind('fp', lambda event: self.tag_event('foot pass'))
        self.master.bind('b', lambda event: self.tag_event('block'))
        self.master.bind('s', lambda event: self.tag_event('save'))
        self.master.bind('f', lambda event: self.tag_event('Free'))
        self.master.bind('fc', lambda event: self.tag_event('Free conceded'))
        self.master.bind('spt', lambda event: self.tag_event('point from play'))
        self.master.bind('sg', lambda event: self.tag_event('goal from play'))
        self.master.bind('fpt', lambda event: self.tag_event('point from free'))
        self.master.bind('fg', lambda event: self.tag_event('goal from free'))
        self.master.bind('spm', lambda event: self.tag_event('miss from play'))
        self.master.bind('fpm', lambda event: self.tag_event('miss from free'))
        self.master.bind('sl', lambda event: self.tag_event('sideline'))
        self.master.bind('r', lambda event: self.tag_event('ball recovery'))
        
        # Half shortcuts
        for i in range(1, 5):
            self.master.bind(str(i), lambda event, half=i: self.set_half(half))

    def save_action_frames(self, start_frame, end_frame, event_name, is_automated=False):
        """Save frames for action recognition training"""
        try:
            frames = []
            for frame_num in range(start_frame, end_frame + 1):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.cap.read()
                if ret:
                    frames.append(frame)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Save to action recogniser if available
            if hasattr(self, 'action_recogniser'):
                self.action_recogniser.save_training_frames(
                    frames,
                    event_name,
                    timestamp,
                    confidence=None if not is_automated else 1.0
                )
                
        except Exception as e:
            print(f"Error saving action frames: {e}")

    def set_half(self, half):
        self.current_half.set(str(half))

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_number = 0
            self.video_slider.config(to=self.total_frames - 1)
            self.update_frame()

    def update_frame(self):
        """Update frame with automatic action detection and training data collection"""

        if not self.team_colours_set:
            return
            
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            ret, frame = self.cap.read()
            if ret:
                # Only do action recognition if not manually tagging an event
                if not hasattr(self, 'is_manual_tagging') or not self.is_manual_tagging:
                    if self.current_frame_number % 3 == 0:
                        self.action_recogniser.update_buffer(frame)
                            
                    if len(self.action_recogniser.frame_buffer) == self.action_recogniser.buffer_size:
                        action = self.action_recogniser.predict_action(self.current_frame_number)
                        if action:
                            event_name, confidence = action
                            if messagebox.askyesno("Action Detected", 
                                                    f"Detected {event_name} with {confidence:.2%} confidence.\n"
                                                    f"Would you like to tag this event?"):
                                
                                # Save frames for training
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                self.save_action_frames(
                                    self.current_frame_number - self.buffer_size,
                                    self.current_frame_number,
                                    event_name,
                                    is_automated=True
                                )
                                
                                if self.is_playing:
                                    self.play_pause()
                                self.tag_event(event_name)


                                # Pause video
                                if self.is_playing:
                                    self.play_pause()
                                # Launch event tagging
                                self.tag_event(event_name)
                
                # Display frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.update_frame_counter()
                
                if not self.slider_update_flag:
                    self.video_slider.set(self.current_frame_number)



    def predict_action(self, current_frame_num):
        if current_frame_num - self.last_detected_frame < self.detection_cooldown:
            return None
            
        # confidence threshold dictionary
        confidence_thresholds = {
        'hand passes': 0.3,
        'foot passes': 0.5,
        'carries': 0.30,
        'Tackles': 0.30
    }
        try:
            frames = []
            indices = np.linspace(0, len(self.frame_buffer)-1, 16, dtype=int)
            for idx in indices:
                frame = self.frame_buffer[idx]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames.append(frame)
                
            clip = torch.stack(frames).permute(1, 0, 2, 3)
            
            with torch.no_grad():
                output = self.action_model(clip.unsqueeze(0))
                prob = F.softmax(output, dim=1)
                pred = output.argmax(1).item()
                confidence = prob[0][pred].item()
                predicted_action = self.action_map[pred]

            # Check class-specific threshold
            if confidence > confidence_thresholds.get(predicted_action, 0.7):
                # Update last detection time
                self.last_detected_frame = current_frame_num
                
                # Print debug info
                print(f"Frame {current_frame_num}: Detected {predicted_action} with {confidence:.2f}")
                return predicted_action, confidence
            
        except Exception as e:
            print(f"Action prediction failed: {e}")
        return None




    def update_frame_counter(self):
        
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        hours, remainder = divmod(timestamp, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.frame_counter_label.config(text=f"Frame: {self.current_frame_number + 1} / {self.total_frames} | "
                                           f"Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    def play_pause(self):
        if self.is_playing:
            self.master.after_cancel(self.play_id)
            self.is_playing = False
            self.play_pause_button.config(text="▶")
        else:
            self.is_playing = True
            self.play_pause_button.config(text="⏸")
            self.play_video()

    def play_video(self):
        if not self.is_playing:
            return
            
        if self.current_frame_number < self.total_frames - 1:
            # Calculate number of frames to skip based on playback speed
            frames_to_advance = max(1, int(self.playback_speed))
            self.current_frame_number = min(self.current_frame_number + frames_to_advance, 
                                        self.total_frames - 1)
            
            # Update display
            self.update_frame()
            
            # Calculate next frame delay
            delay = int(self.frame_interval / self.playback_speed)
            
            # Schedule next frame
            self.play_id = self.master.after(max(1, delay), self.play_video)
        else:
            self.is_playing = False
            self.play_pause_button.config(text="▶")

    def toggle_speed(self):
        """Toggle between different playback speeds"""
        self.current_speed_index = (self.current_speed_index + 1) % len(self.playback_speeds)
        self.playback_speed = self.playback_speeds[self.current_speed_index]
        
        # Update speed button text
        self.speed_button.config(text=f"{self.playback_speed}x")

    def rewind(self):
        frames_to_rewind = 5 * 30
        self.current_frame_number = max(0, self.current_frame_number - frames_to_rewind)
        self.update_frame()

    def fast_forward(self):
        frames_to_skip = 5 * 30
        self.current_frame_number = min(self.current_frame_number + frames_to_skip, self.total_frames - 1)
        self.update_frame()

    def on_slider_change(self, value):
        self.slider_update_flag = True
        self.current_frame_number = int(float(value))
        self.update_frame()
        self.slider_update_flag = False
    def select_event(self, event):
        self.current_frame_number = event['start_frame']
        self.update_frame()
        
    def tag_event(self, event_name):
        """Handle complete event tagging with all components"""
        if self.cap is None:
            messagebox.showerror("Error", "No video loaded.")
            return

    
        # Store starting frame
        start_frame = self.current_frame_number
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not read frame.")
            return
            
        # Ensure frame is uint8
        frame = np.uint8(frame)
        
        # Create duration selection window
        duration_window = tk.Toplevel(self.master)
        duration_window.title("Select Action Duration")
        duration_window.geometry("400x200")
        
        # Store initial playback state
        was_playing = self.is_playing
                
        # Pause video if playing
        if self.is_playing:
            self.play_pause()

        # Add control frame
        controls = ttk.Frame(duration_window)
        controls.pack(fill=tk.X, pady=10)
        
        # Video navigation controls
        ttk.Label(controls, text="Navigate to end of action:").pack()
        nav_buttons = ttk.Frame(controls)
        nav_buttons.pack(pady=5)
        
        ttk.Button(nav_buttons, text="◀◀", command=lambda: self.step_frames(-30)).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_buttons, text="◀", command=lambda: self.step_frames(-1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_buttons, text="▶", command=lambda: self.step_frames(1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_buttons, text="▶▶", command=lambda: self.step_frames(30)).pack(side=tk.LEFT, padx=5)
        
        # Frame counter
        counter_var = tk.StringVar()
        def update_counter():
            counter_var.set(f"Frame: {self.current_frame_number}")
            duration_window.after(50, update_counter)
        
        ttk.Label(duration_window, textvariable=counter_var).pack(pady=10)
        update_counter()
        
        end_frame = [None]  # Use list to store value from callback


        def confirm_end():
            end_frame[0] = self.current_frame_number
            # Always ensure video is paused after selection
            if self.is_playing:
                self.play_pause()
            duration_window.destroy()

            
        ttk.Button(duration_window, text="Set End Frame", command=confirm_end).pack(pady=10)
        
        # Wait for end frame selection
        self.master.wait_window(duration_window)
        
        # If window was closed without setting end frame
        if end_frame[0] is None:
            if was_playing:  # Restore original playback state
                self.play_pause()
            return
        
        self.save_action_frames(
        start_frame,
        end_frame[0], 
        event_name,
        is_automated=not hasattr(self, 'is_manual_tagging')
                )
        # Get player number
        player_number = simpledialog.askstring("Player Number", "Enter the player number:")
        if player_number is None:
            return
        
        try:
            player_number = int(player_number)
            if player_number < 1 or player_number > 99:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid player number (1-99).")
            return
        

        # Add body part selection
        body_part = simpledialog.askstring("Body Part", 
            "Enter body part used (RF, LF, RH, LH, O):")
        if not body_part:
            return

        # Create location selection window
        location_window = tk.Toplevel(self.master)
        location_window.title("Select Ball Positions")
        
        # Create template canvas
        template_canvas = tk.Canvas(location_window, width=self.template_width, 
                                height=self.template_height)
        template_canvas.pack()

        # Load and display template
        template_image = cv2.imread("/Users/diarmuidwhelan/Downloads/research_project/PitchTemplate.png")
        if template_image is not None:
            template_image = cv2.resize(template_image, (self.template_width, self.template_height))
            template_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(template_image, 
                                                                        cv2.COLOR_BGR2RGB)))
            template_canvas.create_image(0, 0, image=template_img, anchor=tk.NW)
            location_window.template_img = template_img  # Keep reference to prevent garbage collection

        # Initialise ball positions dictionary
        ball_positions = {'start': {}, 'end': {}}
        click_count = 0

        def get_position(event):
            """Handle click events for ball position selection"""
            nonlocal click_count
            if click_count == 0:
                # First click for start position
                ball_positions['start'] = {'x': event.x, 'y': event.y}
                template_canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, 
                                        fill='green')
                click_count += 1
            elif click_count == 1:
                # Second click for end position
                ball_positions['end'] = {'x': event.x, 'y': event.y}
                template_canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, 
                                        fill='red')
                # Draw arrow from start to end
                template_canvas.create_line(
                    ball_positions['start']['x'], 
                    ball_positions['start']['y'],
                    event.x, event.y, 
                    arrow=tk.LAST
                )
                location_window.destroy()

        # Bind click event and wait for window closure
        template_canvas.bind('<Button-1>', get_position)
        location_window.wait_window()

        # Launch calibrator for the start frame
        calibrator_window = tk.Toplevel(self.master)
        calibrator = GaelicFootballCalibrator(calibrator_window)
        
        # Pass team colours to calibrator
        calibrator.team1_colour = self.team1_colour.get()
        calibrator.team2_colour = self.team2_colour.get()

        # Load frame into calibrator
        calibrator.load_frame(frame)
        
        # Wait for calibrator window to close
        self.master.wait_window(calibrator_window)
        
        # Get calibration results
        calibration_data = calibrator.get_calibration_data()

        if not calibration_data:
            messagebox.showerror("Error", "No calibration data received")
            return
        # Now find matching player in calibration data
        active_player = None
        for player in calibration_data.get('players', []):
            # Convert both to integers for comparison
            if player.get('number') and int(player.get('number')) == int(player_number):
                active_player = player
                break

        # Also check goalkeepers list
        if active_player is None:
            for goalkeeper in calibration_data.get('goalkeepers', []):
                if goalkeeper.get('number') and int(goalkeeper.get('number')) == int(player_number):
                    active_player = goalkeeper
                    break

        if active_player is None:
            messagebox.showerror("Error", f"Player {player_number} not found in calibration data")
            return

        # Create the complete event with all data
        event = {
            'event_number': self.next_event_number,
            'timestamp': self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
            'start_frame': start_frame,
            'end_frame': end_frame[0],
            'event_name': event_name,
            'team': self.possession_team.get(),
            'player_number': player_number,
            'body_part': body_part,
            'ball_start_location': ball_positions['start'],
            'ball_end_location': ball_positions['end'],
            'half': self.current_half.get(),
            'calibration_data': calibration_data
        }



        # Add ball position from active player
        if active_player.get('transformed_position'):
            event['ball_position'] = active_player['transformed_position']

        # Save frames for training if needed
        if hasattr(self, 'action_recogniser'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.action_recogniser.save_training_frames(
                self.frame_buffer,
                event_name,
                timestamp
            )
        # Add event and update timeline
        self.events.append(event)
        self.timeline.update_timeline(self.events, self.total_frames)
        
        # Update event counter
        while self.next_event_number in self.deleted_events:
            self.next_event_number += 1
        self.next_event_number += 1
        self.event_counter += 1
        
    
    def step_frames(self, frames):
        """Step forward/backward by specified number of frames"""
        new_frame = max(0, min(self.current_frame_number + frames, self.total_frames - 1))
        self.current_frame_number = new_frame
        self.update_frame()

    def get_training_statistics(self):
        """Get statistics about collected training data"""
        stats = {
            'total_sequences': 0,
            'by_event': {},
            'by_detection': {
                'manual': 0,
                'automatic': 0
            }
        }
        
        try:
            # Count sequences for each event type
            for event_dir in self.action_recogniser.images_path.iterdir():
                if event_dir.is_dir():
                    sequence_count = len(list(event_dir.glob("*_000.jpg")))  # Count first frames
                    stats['by_event'][event_dir.name] = sequence_count
                    stats['total_sequences'] += sequence_count
            
            # Count detection methods from metadata
            for metadata_file in self.action_recogniser.labels_path.glob("*.json"):
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    detection_method = metadata.get('detection_method', 'manual')
                    stats['by_detection'][detection_method] += 1
                    
            return stats
            
        except Exception as e:
            print(f"Error getting training statistics: {e}")
            return None

    def plot_training_metrics(history, save_path=None):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train')
        ax1.plot(history['val_loss'], label='Validation')
        ax1.set_title('Loss Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train')
        ax2.plot(history['val_acc'], label='Validation')
        ax2.set_title('Accuracy Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '/training_metrics.png')
        plt.show()

    def plot_confusion_matrix(model, val_loader, classes, save_path=None):
        """Generate and plot confusion matrix"""
        y_pred = []
        y_true = []
        
        model.eval()
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                outputs = model(clips)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path + '/confusion_matrix.png', bbox_inches='tight')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=classes))

    def display_training_info(self):
        """Display training data collection statistics"""
        stats = self.get_training_statistics()
        if not stats:
            return
            
        info_text = f"""Training Data Collection Statistics:
        
    Total Sequences: {stats['total_sequences']}

    By Event Type:
    {chr(10).join(f'  {event}: {count} sequences' for event, count in stats['by_event'].items())}

    Detection Method:
    Manual: {stats['by_detection']['manual']} sequences
    Automatic: {stats['by_detection']['automatic']} sequences
    """
        messagebox.showinfo("Training Data Statistics", info_text)
        
    def delete_tag(self):
        #when the delete button is pressed remove the selected event from timeline and data
        selected_event = self.timeline.selected_event
        if selected_event:
            self.deleted_events.add(selected_event['event_number'])
            self.events = [e for e in self.events if e['event_number'] != selected_event['event_number']]
            self.timeline.update_timeline(self.events, self.total_frames)
        else:
            messagebox.showwarning("No Selection", "Please select an event to delete.")

    def reedit_tag(self):
        #when the reedit button is pressed trigger the calibrator module

        selected_event = self.timeline.selected_event
        if selected_event:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, selected_event['start_frame'])
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Could not read the frame for this event.")
                return

            calibrator_window = tk.Toplevel(self.master)
            calibrator = GaelicFootballCalibrator(calibrator_window)
            calibrator.team1_colour = self.team1_colour
            calibrator.team2_colour = self.team2_colour
            calibrator.load_frame(frame)

            self.master.wait_window(calibrator_window)
            selected_event['start_calibration_data'] = calibrator.get_calibration_data()
            self.timeline.update_timeline(self.events, self.total_frames)
        else:
            messagebox.showwarning("No Selection", "Please select an event to re-edit.")

    def save_data(self):
        #wave th data generated

        if not self.events:
            messagebox.showwarning("No Data", "No events have been tagged.")
            return

        # Get video filename without extension
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        default_filename = f"{video_name}_annotations.json"
        
        file_path = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            data = {
                'video_path': self.video_path,
                'total_events': len(self.events),
                'team1_colour': self.team1_colour.get(),
                'team2_colour': self.team2_colour.get(),
                'events': self.events
            }
            try:
                with open(file_path, 'w') as f:
                    json.dump(data, f, cls=NumpyJSONEncoder, indent=2)
                messagebox.showinfo("Success", f"Event data saved to {os.path.basename(file_path)}")

                    # Save training data for each event
                for event in self.events:
                    # Get frames
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, event['start_frame'])
                    ret, start_frame = self.cap.read()
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, event['end_frame'])
                    ret, end_frame = self.cap.read()
                    
                    if ret:
                        self.data_collector.save_training_instance(
                            start_frame,
                            event['calibration_data'],
                            self.video_path
                        )
                        
                

                messagebox.showinfo("Success", f"Event data saved to {os.path.basename(file_path)}")
                
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEventTagger(root)
    root.mainloop()
