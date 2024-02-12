export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
python -m torch.distributed.run   --nproc_per_node=$num_gpus --master_addr="127.0.0.1" --master_port=29512 \
examples/text-classification/fullhg_run_qnli.py \
--model_name_or_path /dt/ft/deberta \
--task_name qnli \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 12 \
--learning_rate 1e-4 \
--num_train_epochs 8 \
--output_dir qnli/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir qnli/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 50 \
--save_strategy steps \
--save_steps 500000 \
--warmup_steps 500 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \
