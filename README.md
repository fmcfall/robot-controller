# Wearable Controller for Robot-Assisted Microsurgery

This reporsitory contains the files used for my final year project, in which I designed a Wearable Controller for Robot-Assisted Microsurgery.

Description of the folders and files:
- filters:
  - kf.py - kalman filter class
  - ekf.py - extended kalman filter class
  - ukf.py - unscented kalman filter class
- tests:
  - implementations of all filters
  - test files used for the results
  - implementation of robot control using ArUco UKF method
- old tests:
  - individual filter testing files
- aruco:
  - aruco generation and detection scripts
- simulation:
  - UR5 simulation using ArUco UKF method

The video of the code running using micromanipulator is found at:
https://youtu.be/066_cpsxXQU

The corresonding video of the wearable leader controller is found at:
https://youtube.com/shorts/YFJpbKTj-bY?feature=share

Thank you and enjoy!
