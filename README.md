# Wearable Controller for Robot-Assisted Microsurgery

This reporsitory contains the files used for my final year project, in which I designed a Wearable Controller for Robot-Assisted Microsurgery.

The important files are:
-kf.py - kalman filter class
-ekf.py - extended kalman filter class
-ukf.py - unscented kalman filter class

The filters with aruco or hands are the full pipeline scripts to be run with all the correct attachments:
-stereo camera
-imu

robot_test.py tests and controls the robot
run_all.py/run_all_mp.py runs every filter for compairson

The video of the code running using micromanipulator is found at:
https://youtu.be/066_cpsxXQU

The corresonding video of the wearable leader controller is found at:
https://youtube.com/shorts/YFJpbKTj-bY?feature=share

Thank you.
