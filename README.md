# Object Tracker using YOLOV8.

This project demonstrates an object tracking system using YOLO (You Only Look Once) for object detection and DeepSort for object tracking. The system is capable of tracking multiple objects in a video stream, focusing specifically on persons.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Ultralytics YOLO
- DeepSort

## Installation

1. Clone the repository:
- git clone https://github.com/Elton997/yolov8-human-detect-tracking.git
  cd yolov8-human-detect-tracking

2. Install the required dependencies:
- pip install -r requirements.txt


## Usage

1. Run the object tracker script with the desired video file:
- python object_tracker.py --video_path input_video.mp4 --output_path output_video.mp4


Replace `input_video.mp4` with the path to your input video file and `output_video.mp4` with the desired output video file name.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
