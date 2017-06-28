import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath('%s/../../' % cur_dir))

from clipper_admin import Clipper

CLIPPER_MODEL_NAME = "imgnet_inception_model"
CLIPPER_DOCKER_LABEL = "ai.clipper.container.label"
CLIPPER_MODEL_CONTAINER_LABEL = "ai.clipper.model_container.model_version"
BENCH_NW = "BENCHNET"

def launch_container(clipper, model_name, model_version, model_input_type, clipper_ip, checkpoint_path, tf_slim_path):
	docker_checkpoint_path = "/model_checkpoint"
	docker_tf_slim_path = "/tfslim"
	image_name = "clipper/tf_inception_container"
	add_container_cmd = (
		"docker run -d --network={nw} --restart={restart_policy} -v {ckptpath}:{dcp} -v {tfspath}:{dtfsp} "
		"-e \"CLIPPER_MODEL_NAME={mn}\" -e \"CLIPPER_MODEL_VERSION={mv}\" "
		"-e \"CLIPPER_MODEL_CHECKPOINT_PATH={dcp}\" "
		"-e \"CLIPPER_IP={ip}\" -e \"CLIPPER_INPUT_TYPE={mip}\" -l \"{clipper_label}\" -l \"{mv_label}\" "
		"{image}".format(
			ckptpath=checkpoint_path,
			tfspath=tf_slim_path,
			nw=BENCH_NW,
			image=image_name,
			mn=model_name,
			dcp=docker_checkpoint_path,
			dtfsp=docker_tf_slim_path,
			mv=model_version,
			mip=model_input_type,
			ip=clipper_ip,
			clipper_label=CLIPPER_DOCKER_LABEL,
			mv_label="%s=%s:%s" % (CLIPPER_MODEL_CONTAINER_LABEL,
									model_name, model_version),
			restart_policy='no'))
	result = clipper._execute_root(add_container_cmd)
	return result.return_code == 0


if __name__ == "__main__":
	if len(sys.argv) < 4:
		raise

	clipper_ip = sys.argv[1]
	checkpoint_path = sys.argv[2]
	tf_slim_path = sys.argv[3]
	num_containers = int(sys.argv[4])

	name = CLIPPER_MODEL_NAME
	version = 1
	input_type = "bytes"

	clipper = Clipper("localhost")

	for i in range(0, num_containers):
		print("Launched container {}?: {}".format(i, launch_container(clipper, name, version, input_type, clipper_ip, checkpoint_path, tf_slim_path)))

