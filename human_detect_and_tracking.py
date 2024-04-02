import time
import torch
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

class ObjectTracker:
    def __init__(self, yolo_model_path, deep_sort_weights, video_path, output_path):
        self.yolo_model_path = yolo_model_path
        self.deep_sort_weights = deep_sort_weights
        self.video_path = video_path
        self.output_path = output_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tracker = None

    def load_yolo_model(self):
        try:
            self.model = YOLO(self.yolo_model_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")

    def load_deep_sort_model(self):
        try:
            self.tracker = DeepSort(model_path=self.deep_sort_weights, max_age=70)
        except Exception as e:
            print(f"Error loading DeepSort model: {e}")
    
    def is_bbox_similar(self, bbox1, bbox2, threshold=0.5):
        """
        Check if two bounding boxes are similar based on IoU.
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        area1 = w1 * h1
        area2 = w2 * h2
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        iou = inter_area / (area1 + area2 - inter_area)
        # print(iou)
        return iou >= threshold

    def process_video(self):
        print(f"Processing the Input Video")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
        
        frames = []
        unique_track_ids = set()
        counter, elapsed = 0, 0
        unique_track_bboxes = {}
        start_time = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            try:
                og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = og_frame.copy()

                results = self.model(frame)

                for result in results:
                    boxes = result.boxes
                    conf = boxes.conf.detach().cpu().numpy()
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    xywh = boxes.xywh.cpu().numpy()
                    cls = boxes.cls.tolist()

                    for i, class_index in enumerate(cls):
                        class_name = self.model.names[int(class_index)]
                        bbox_xywh = xywh[i]
                        if class_name.lower()=="person":
                            # Check if a similar bbox is already being tracked
                            found = False
                            for track_id, values in unique_track_bboxes.items():
                                track_bbox = values["bbox_xywh"]
                                to_tlbr=values["to_tlbr"]
                                if self.is_bbox_similar(bbox_xywh, track_bbox):
                                    found = True
                                    x1, y1, x2, y2 = to_tlbr
                                    w = x2 - x1
                                    h = y2 - y1
                                    cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
                                    cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                    break
                            # print(unique_track_bboxes)
                            if not found:
                                tracks = self.tracker.update(np.expand_dims(bbox_xywh, axis=0), np.array([conf[i]]), og_frame)
                                for track in self.tracker.tracker.tracks:
                                    track_id = track.track_id
                                    # unique_track_bboxes[track_id] = bbox_xywh
                                    
                                    x1, y1, x2, y2 = track.to_tlbr()
                                    unique_track_bboxes[track_id] = {"bbox_xywh":bbox_xywh,"to_tlbr":track.to_tlbr()}
                                    w = x2 - x1
                                    h = y2 - y1
                                
                                    color = (0, 255, 0)  # Green color for new track

                                    cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
                                    cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                                    unique_track_ids.add(track_id)
                            
                person_count = len(unique_track_ids)

                current_time = time.perf_counter()
                elapsed = (current_time - start_time)
                counter += 1
                if elapsed > 1:
                    fps = counter / elapsed
                    counter = 0
                    start_time = current_time

                cv2.putText(og_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                frames.append(og_frame)

                out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error processing frame: {e}")

        cap.release()
        out.release()

if __name__ == "__main__":
    yolo_model_path = "yolov8n.pt"
    deep_sort_weights = "deep_sort/deep/checkpoint/ckpt.t7"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="input_video.mp4", help="Path to input video file")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path to output video file")
    args = parser.parse_args()
    video_path = args.video_path
    output_path = args.output_path
    print(f"Recieved Input_Path: {video_path} and Output_Path: {output_path}")
    print(f"YOLO Model Used: {yolo_model_path}")

    tracker = ObjectTracker(yolo_model_path, deep_sort_weights, video_path, output_path)
    tracker.load_yolo_model()
    tracker.load_deep_sort_model()
    tracker.process_video()


# python human_detect_and_tracking.py --video_path testing_data/Input_Data/input_video.mp4 --output_path testing_data/Output_Data/output_human.mp4
