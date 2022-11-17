from sensapex import UMP
import cv2 as cv

ump = UMP.get_ump()
dev_ids = ump.list_devices()

#stage = ump.get_device(1)
#stage.calibrate_zero_position()

new_pos = [9999.8818359375, 9999.986328125, 16999.9501953125, 9999.90625]

'''
[x, y, z, insertion]
calibrated = [9999.8818359375, 9999.986328125, 9999.9501953125, 9999.90625]


'''

manipulator = ump.get_device(1)
manipulator.goto_pos(new_pos, speed=1500)

