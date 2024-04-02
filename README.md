# Human Tracker using YOLOV8.

This project demonstrates an Human tracking system using YOLO (You Only Look Once) for human detection and DeepSort for human tracking. The system is capable of tracking multiple objects in a video stream, focusing specifically on persons.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Ultralytics YOLO
- DeepSort

## Installation

1. Clone the repository:
- git clone https://github.com/Elton997/yolov8-human-detect-tracking.git
- cd yolov8-human-detect-tracking

2. Install the required dependencies:
- pip install -r requirements.txt


## Usage for running normal python file:

1. Run the human tracker script with the desired video file:
- python human_detect_and_tracking.py --video_path input_video.mp4 --output_path output_video.mp4

## Implementation & Usage for running docker:

1. Build the Docker image:
- docker build -t human_detect_and_tracking .

2. Run the Docker container with the following command:
- docker run human_detect_and_tracking --video_path app/testing_data/Input_Data/input_video.mp4 --output_path app/testing_data/Output_Data/output_human.mp4

Replace `input_video.mp4` with the path to your input video file and `output_video.mp4` with the desired output video file name.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
