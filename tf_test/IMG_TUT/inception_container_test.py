import sys
import os
import requests
import numpy as np
import json
import base64

def load_labels_names_info(path):
	file_contents = open(path, "r")
	lines = file_contents.readlines()
	labels_names = np.array([tuple(line.split(" ")) for line in lines])
	return labels_names

if __name__ == "__main__":
	if len(sys.argv) < 4:
		raise
	labels_names_path = sys.argv[1]
	images_path = sys.argv[2]
	num_images = sys.argv[3]

	labels_names_info = load_labels_names_info(labels_names_path)
	for i in range(0,10):
		print(labels_names_info[i])
	np.random.shuffle(labels_names_info)

	image_file_name, label = labels_names_info[0]
	image_file_path = os.path.join(images_path, "%s.JPEG" % image_file_name)

	image_file = open(image_file_path, "r")
	image_contents = bytes(image_file.read())
	encoded = base64.b64encode(image_contents)

	url = "http://localhost:1337/app1/predict"

	req_json = json.dumps({'input': encoded})
	headers = {'Content-type': 'application/json'}
	r = requests.post(url, headers=headers, data=req_json)

	print(r.text)
	print(label)