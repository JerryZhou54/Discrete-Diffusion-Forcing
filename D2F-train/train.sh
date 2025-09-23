# export CUDA_LAUNCH_BLOCKING=1
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch   --config_file config/acc_config --num_processes 4 --main_process_port 29503 train.py --config config/llada.yaml

# CUDA_VISIBLE_DEVICES=4 accelerate launch   --config_file config/acc_config --num_processes 1 --main_process_port 29577 train.py --config config/dream_eagle.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch   --config_file config/acc_config --num_processes 8 --main_process_port 29577 train.py --config config/llada_distill.yaml