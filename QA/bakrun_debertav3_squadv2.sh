
export output_dir="./output/debertav3-base/squadv2"
python -m torch.distributed.run --master_port=8679 --nproc_per_node=1 \
examples/question-answering/lora_run_qa.py \
--model_name_or_path /dt/ft/deberta \
--dataset_name squad_v2 \
--apply_lora --apply_adalora \
--reg_orth_coef 0.1 \
--init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--do_train --do_eval --version_2_with_negative \
--max_seq_length 384 --doc_stride 128 \
--per_device_train_batch_size 16 \
--learning_rate 1e-3 \
--num_train_epochs 12 \
--max_step 1563828 \
--warmup_steps 1000 --per_device_eval_batch_size 128 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 1000000000000 \
--logging_steps 300 \
--tb_writter_loginterval 300 \
--report_to tensorboard \
--seed 9 \
--output_dir ./output/debertav3-base/squadv2 \
--logging_dir $output_dir/log \
--overwrite_output_dir
