# Real-Time Lane Detection

This is a project focused on real-time lane detection using various computer vision techniques. The primary algorithm detects lane lines in videos and marks them for visualization.
The main focus of the project is to leverage the utilization of computer vision algorithms and OpenCV-Python processing to handle the data in real-time or near-real-time constraint, and as a result, in our test, a 27-second 720p video (50 FPS) was able to be processed in 35 seconds including the draw-line functions.

## Demo Output
- 

https://github.com/user-attachments/assets/c82a795b-b7d1-4cf6-ac7d-1042ba869c8b



## Repository Overview

- **Main File**: `Lane_Detection.ipynb`
- **Primary Language**: Jupyter Notebook

## Features

The Jupyter Notebook `Lane_Detection.ipynb` includes the following features:
1. **Import Libraries**: The notebook starts by importing essential libraries such as `numpy`, `cv2`, and `moviepy`.
2. **Lane Detection Pipeline**: The notebook demonstrates a comprehensive pipeline for detecting lane lines in video frames.
3. **Video Processing**: The notebook processes a video file to detect and visualize lane lines.

## Repository Structure

- `Lane_Detection.ipynb`: The main Jupyter Notebook containing the lane detection implementation.

## How to Use

To use this project, follow these steps:
1. Clone the repository.
2. Install the libraries in `requirements.txt`.
3. Open `Lane_Detection.ipynb` in Jupyter Notebook.
4. Follow the instructions in the notebook to see the lane detection process in action.

## Detailed Algorithm

### 1. Import Libraries

The necessary libraries are imported to handle image processing and video editing.

```python
import numpy as np
import cv2
from moviepy import editor
```

### 2. Hough Transform

The `hough_transform` function applies the Hough Line Transform to detect lines in an edge-detected image.

```python
def hough_transform(image):
    rho = 1
    theta = np.pi/180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines if lines is not None else []
```

### 3. Region Selection

The `region_selection` function masks the image to focus on the region of interest where lanes are likely to be found.

```python
def region_selection(image):
    mask = np.zeros_like(image) 
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
```

### 4. Average Slope Intercept

The `average_slope_intercept` function calculates the average slope and intercept for the left and right lane lines.

```python
def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane
```

### 5. Pixel Points

The `pixel_points` function converts the slope and intercept of a line to pixel points.

```python
def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))
```

### 6. Lane Lines

The `lane_lines` function identifies the coordinates of the left and right lane lines.

```python
def lane_lines(image, lines):
    if lines is None or len(lines) == 0:
        return None, None
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = int(y1 * 0.52)
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
```

### 7. Draw Lane Lines

The `draw_lane_lines` function draws the lane lines on the image.

```python
def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=12):
    line_image = np.zeros_like(image)
    if lines:
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
```

### 8. Frame Processor

The `frame_processor` function processes each video frame to detect and mark the lane lines.

```python
def frame_processor(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 3
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low_t = 20
    high_t = 100
    edges = cv2.Canny(blur, low_t, high_t)
    region = region_selection(edges)
    hough = hough_transform(region)
    lanes = lane_lines(image, hough)
    result = draw_lane_lines(image, lanes)
    
    grayscale_3ch = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    region_3ch = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    
    concatenated = np.concatenate((grayscale_3ch, edges_3ch, region_3ch, result), axis=1)
    
    return concatenated
```

### 9. Process Video

The `process_video` function processes a video file, applying the lane detection algorithm to each frame.

```python
def process_video(test_video, output_video, fps=25.0):
    try:
        input_video = editor.VideoFileClip(test_video, audio=False)
    except Exception as e:
        raise ValueError(f"Error loading video {test_video}: {e}")
    fps = float(input_video.fps) if input_video.fps is not None else fps
    print(f"Processing video {test_video} at {fps} FPS")

    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_video, fps=fps, audio=False)

process_video('test2.mp4', 'output.mp4')
```

### Summary

The algorithm follows these steps:
- Import necessary libraries.
- Define functions for Hough Transform, region selection, averaging slopes, converting to pixel points, identifying lane lines, drawing lane lines, and processing video frames.
- Process each frame of the input video to detect and mark lane lines.
- Save the processed video with detected lane lines.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.
