from sensapex import UMP

ump = UMP.get_ump()
dev_ids = ump.list_devices()
print(dev_ids)

stage = ump.get_device(1)
stage.calibrate_zero_position()