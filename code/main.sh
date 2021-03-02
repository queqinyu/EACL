#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=0

#python data_aug.py


for (( i = 0; i < 5; i++ ));

do
python train.py \
--model_type roberta \
--do_train \
--do_eval_during_train \
--model_name_or_path ../roberta/ \
--data_dir ../data/EACL2021/merge/EDI/eng/data_StratifiedKFold_42/data_origin_$i \
--output_dir ../checkpoints/EDI/eng/roberta/$i \
--max_seq_length 128 \
--learning_rate 2e-5 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4

done

exec /bin/bash



#python predict.py \
#--model_type roberta \
#--vote_model_paths ../checkpoints/EDI/mala/roberta/ \
#--predict_file ../data/EACL2021/merge/EDI/mala/test.csv \
#--predict_result_file ../data/EACL2021/result/EDI/mala/roberta/EDI_mala_roberta.csv \


#usage: predict.py [-h] --model_type MODEL_TYPE
#                  [--fold_model_paths FOLD_MODEL_PATHS]
#                  [--vote_model_paths VOTE_MODEL_PATHS]
#                  [--predict_file PREDICT_FILE]
#                  [--predict_result_file PREDICT_RESULT_FILE]
#                  [--max_seq_length MAX_SEQ_LENGTH]
#                  [--do_lower_case DO_LOWER_CASE] [--batch_size BATCH_SIZE]
#                  [--no_cuda]
#predict.py: error: the following arguments are required: --model_type