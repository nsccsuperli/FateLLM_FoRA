
export output_dir="./output/debertav3-base/squadv2"
python examples/question-answering/lora_run_qa.py \
--model_name_or_path /dt/ft/deberta \
--dataset_name squad \
--do_train \
--do_eval \
--learning_rate 1e-3 \
--num_train_epochs 12 \
--max_seq_length 512 \
--doc_stride 128 \
--output_dir relative_squad \
--per_device_eval_batch_size=60 \
--per_device_train_batch_size 24 \
--evaluation_strategy steps --eval_steps 300 \
--logging_steps 100 \
--overwrite_output_dir
