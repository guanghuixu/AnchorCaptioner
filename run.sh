CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node $2 --master_port $4 tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
	--config configs/captioning/m4c_textcaps/m4c_captioner.yml \
	--save_dir save/$3 --resume_file save/$3/m4c_textcaps_m4c_captioner_2021/best.ckpt \
	training_parameters.distributed True
