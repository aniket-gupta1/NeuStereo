from glob import glob
from PIL import Image
import numpy as np
import os
from data_utils import frame_utils


dataset_root = '/media/hulk/1TB_data_drive/optical_flow/datasets/FlyingThings3D/'

image_files = sorted(glob(dataset_root+'frames_cleanpass/*/*/*/*/*.png'))
print(len(image_files))
for image_file in image_files:
	print("Entered")
	print(image_file)

	pil_image = Image.open(image_file)
	image = np.array(pil_image).astype(np.uint8)

	image_name = os.path.splitext(image_file)[0]
	np.save(image_name, image)

 	# os.remove(image_file)

flow_files = sorted(glob(dataset_root+'flyingthings3d__disparity/disparity/*/*/*/*/*.pfm'))
for flow_file in flow_files:
	print("Entered Flow")
	print(flow_file)

	flow = frame_utils.read_gen(flow_file)
	flow = np.array(flow).astype(np.float32)

	flow_name = os.path.splitext(flow_file)[0]
	np.save(flow_name, flow)

	# os.remove(flow_file)
