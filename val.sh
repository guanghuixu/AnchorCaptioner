CUDA_VISIBLE_DEVICES=$1 python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
	--config configs/captioning/m4c_textcaps/m4c_captioner.yml \
	--save_dir save/$2 \
	--run_type $3 --resume_file $4 \
	--evalai_inference 1
