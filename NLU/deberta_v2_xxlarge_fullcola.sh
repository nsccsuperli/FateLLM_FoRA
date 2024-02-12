export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
python -m torch.distributed.run   --nproc_per_node=$num_gpus --master_addr="127.0.0.1" --master_port=29501 \
examples/text-classification/fullhg_run_cola.py \
--model_name_or_path /dt/ft/deberta \
--task_name cola \
--do_train \
--do_eval \
--max_seq_length 64 \
--per_device_train_batch_size 16 \
--learning_rate 1.3e-4 \
--num_train_epochs 10 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 20 \
--save_strategy steps \
--save_steps 5000 \
--warmup_steps 100 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0 \

