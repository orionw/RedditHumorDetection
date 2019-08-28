export BERT_BASE_DIR=./uncased_L-24_H-1024_A-16
export CUDA_VISIBLE_DEVICES=0
python3 run_classifier.py \
--task_name=cola \
--do_train \
--do_eval \
--data_dir=./data \
--max_seq_length=128 \
--train_batch_size=256 \
--learning_rate=2e-5 \
--num_train_epochs=1.0 \
--do_lower_case \
--gradient_accumulation_steps 24 \
--bert_model=bert-base-uncased \
--output_dir=./final_results \
