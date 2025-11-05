Blue Hex Mat Tracker
A real-time computer vision application for tracking the position and orientation of a blue hexagonal mat using a USB webcam. Built with Python, OpenCV, and Streamlit.
Overview
This application detects a blue, 6-sided floating mat in real-time video and tracks its deviation from an initial reference position. Perfect for motion analysis, physical therapy monitoring, balance training assessment, or any application requiring precise 2D object tracking.
Features

Real-time Detection: Tracks blue hexagonal objects at adjustable frame rates (1-30 FPS)
Position Tracking: Measures X/Y displacement and total distance from reference point
Orientation Tracking: Calculates angular drift using PCA-based orientation analysis
Calibration Support: Convert pixel measurements to centimeters using known mat dimensions
Visual Feedback: Live annotated video stream with reference markers and displacement vectors
Data Logging: Records tracking data every 5 seconds with CSV export capability
Interactive Charts: Real-time plots of distance and angle drift over time
Customizable Detection: Adjustable HSV thresholds, morphological operations, and polygon approximation parameters

Requirements
python >= 3.7
opencv-python
numpy
streamlit
pandas
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/blue-hex-mat-tracker.git
cd blue-hex-mat-tracker

Install dependencies:

bashpip install -r requirements.txt
Usage

Start the application:

bashstreamlit run blue_hex_mat_tracker.py

Connect your USB webcam
Adjust settings in the sidebar:

HSV Thresholds: Fine-tune color detection for your specific mat
Camera Settings: Select camera index and resolution
Detection Parameters: Adjust minimum area, polygon approximation, and morphological operations


Click ▶️ Start to begin video capture
Tune HSV sliders until only your blue mat is highlighted in the mask preview (top-left corner)
Position the mat at your desired reference point and click 📍 Set reference (current)
Move the mat and observe real-time tracking data and visualizations

Configuration
HSV Presets

Default Blue: (H: 90-140, S: 80-255, V: 80-255)
Light Blue: (H: 100-140, S: 50-255, V: 50-255)
Dark Blue: (H: 90-140, S: 100-255, V: 50-200)

Calibration
Enter your mat's side length in centimeters for accurate distance measurements. The application uses the mat's circumscribed circle diameter for pixel-to-cm conversion.
Processing Control
Adjust processing FPS to balance between tracking accuracy and CPU usage.
Output Data
The application records and exports:

time_s: Elapsed time in seconds
dx_cm: X-axis displacement (cm)
dy_cm: Y-axis displacement (cm)
distance_cm: Total displacement from reference (cm)
angle_drift_deg: Angular rotation from reference (degrees)

Troubleshooting
Camera not detected:

Try different camera indices (0, 1, 2...)
Use the "Test Camera" button to verify camera access
Close other applications using the webcam

Mat not detected:

Ensure good, even lighting
Remove other blue objects from view
Adjust HSV thresholds using the sliders
Increase morphological kernel size for noisy environments

Inaccurate measurements:

Verify the mat's side length is correctly entered
Keep camera height and angle consistent
Ensure the mat is fully visible and not occluded

Cross-Platform Support
The application automatically tries multiple OpenCV backends for camera access:

Windows: DirectShow (CAP_DSHOW)
Linux: Video4Linux2 (CAP_V4L2)
Fallback: Generic (CAP_ANY)

Tips

Use a mat with 6 clear, distinct edges for best results
Avoid shadows and reflections on the mat surface
For precise measurements, mount the camera directly above the tracking area
The polygon approximation epsilon parameter affects edge detection sensitivity
Debug mode shows additional detection information for troubleshooting

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
Acknowledgments
Built with:

OpenCV - Computer vision processing
Streamlit - Web application framework
NumPy - Numerical computations
Pandas - Data handling and export
