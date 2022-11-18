from sensapex import UMP

ump = UMP.get_ump()
dev_ids = ump.list_devices()

stage = ump.get_device(1)
stage.calibrate_zero_position()