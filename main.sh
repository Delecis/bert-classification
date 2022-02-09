

export CUDA_VISIBLE_DEVICES=2

#python data_aug.py

for ((i=0;i<5;i++));
do

python train.py \
--do_train \
--do_eval_during_train \
--max_seq_length 128 \
--model_name_or_path ../model/roberta_en \
--data_dir ../DATA/MAMI/Kfold/$i \
--output_dir ../user_data/checkpoints_MAMI/roberta_en/$i \
--learning_rate 5e-5

done

python predict.py \
--vote_model_paths ../user_data/checkpoints_MAMI/roberta_en \
--predict_file ../DATA/test_new.csv \
--predict_result_file ../DATA/MAMI/pre1_ro1.csv


for ((i=0;i<5;i++));
do

python train.py \
--do_train \
--do_eval_during_train \
--max_seq_length 128 \
--model_name_or_path ../model/roberta_en \
--data_dir ../DATA/MAMI/Kfold_shaming/$i \
--output_dir ../user_data/checkpoints_MAMI_shamin/roberta_en/$i \
--learning_rate 5e-5

done


python predict.py \
--vote_model_paths ../user_data/checkpoints_MAMI_shamin/roberta_en \
--predict_file ../DATA/test_new.csv \
--predict_result_file ../DATA/MAMI/pre_shaming_ro1.csv

for ((i=0;i<5;i++));
do

python train.py \
--do_train \
--do_eval_during_train \
--max_seq_length 128 \
--model_name_or_path ../model/roberta_en \
--data_dir ../DATA/MAMI/Kfold_stereotype/$i \
--output_dir ../user_data/checkpoints_MAMI_stereotype/roberta_en/$i \
--learning_rate 5e-5

done

python predict.py \
--vote_model_paths ../user_data/checkpoints_MAMI_stereotype/roberta_en \
--predict_file ../DATA/test_new.csv \
--predict_result_file ../DATA/MAMI/pre_stereotype_ro1.csv

for ((i=0;i<5;i++));
do

python train.py \
--do_train \
--do_eval_during_train \
--max_seq_length 128 \
--model_name_or_path ../model/roberta_en \
--data_dir ../DATA/MAMI/Kfold_objectification/$i \
--output_dir ../user_data/checkpoints_MAMI_objectification/roberta_en/$i \
--learning_rate 5e-5

done

python predict.py \
--vote_model_paths ../user_data/checkpoints_MAMI_objectification/roberta_en \
--predict_file ../DATA/test_new.csv \
--predict_result_file ../DATA/MAMI/pre_objectification_ro1.csv

for ((i=0;i<5;i++));
do

python train.py \
--do_train \
--do_eval_during_train \
--max_seq_length 128 \
--model_name_or_path ../model/roberta_en \
--data_dir ../DATA/MAMI/Kfold_violence/$i \
--output_dir ../user_data/checkpoints_MAMI_violence/roberta_en/$i \
--learning_rate 5e-5

done

python predict.py \
--vote_model_paths ../user_data/checkpoints_MAMI_violence/roberta_en \
--predict_file ../DATA/test_new.csv \
--predict_result_file ../DATA/MAMI/pre_violence_ro1.csv















#for ((i=0;i<5;i++));
#do
#
#python train.py \
#--do_train \
#--do_eval_during_train \
#--max_seq_length 128 \
#--model_name_or_path ../model/bert_en \
#--data_dir ../data_memotion/reC_sarcastic/Kfold/$i \
#--output_dir ../user_data/checkpoints_mm/reC_sarcastic/bert_en/$i \
#--learning_rate 5e-5
#
#done
#
#
#for ((i=0;i<5;i++));
#do
#
#python train.py \
#--do_train \
#--do_eval_during_train \
#--max_seq_length 128 \
#--model_name_or_path ../model/bert_en \
#--data_dir ../data_memotion/reC_offensive/Kfold/$i \
#--output_dir ../user_data/checkpoints_mm/reC_offensive/bert_en/$i \
#--learning_rate 5e-5
#
#done


#for ((i=0;i<5;i++));
#do
#
#python train.py \
#--do_train \
#--do_eval_during_train \
#--max_seq_length 128 \
#--model_name_or_path ../model/bert_en \
#--data_dir ../data_memotion/reB_motivational2/Kfold/$i \
#--output_dir ../user_data/checkpoints_mm/reB_motivational2/bert_en/$i \
#--learning_rate 5e-5
#
#done
#
#python predict.py \
#--vote_model_paths ../user_data/checkpoints_MAMI/bert_en \
#--predict_file ../DATA/test_new.csv \
#--predict_result_file ../DATA/MAMI/pre1.csv

#python predict.py \
#--vote_model_paths ../user_data/checkpoints_mm/reC_sarcastic/bert_en \
#--predict_file ../data_memotion/new_test.csv \
#--predict_result_file ../data_memotion/test_reC_sarcastic.csv
#
#python predict.py \
#--vote_model_paths ../user_data/checkpoints_mm/reC_offensive/bert_en \
#--predict_file ../data_memotion/new_test.csv \
#--predict_result_file ../data_memotion/test_reC_offensive.csv

#python predict.py \
#--vote_model_paths ../user_data/checkpoints_mm/reB_motivational2/bert_en \
#--predict_file ../data_memotion/new_test.csv \
#--predict_result_file ../data_memotion/test_reB_motivational2.csv








#for ((i=0;i<5;i++));
#do
#
#python train.py \
#--do_train \
#--do_eval_during_train \
#--max_seq_length 128 \
#--model_name_or_path ../model/bert_en \
#--data_dir ../data_AAAI/Kfold/$i \
#--output_dir ../user_data/checkpoints_AAAI/bert_en/$i \
#--learning_rate 4e-5
#
#done

#for ((i=0;i<5;i++));
#do
#
#python train.py \
#--do_train \
#--do_eval_during_train \
#--max_seq_length 128 \
#--model_name_or_path ../model/roberta_en \
#--data_dir ../data_AAAI/Kfold/$i \
#--output_dir ../user_data/checkpoints_AAAI/roberta_en/$i \
#--learning_rate 4e-5
#
#done
#for ((i=0;i<5;i++));
#do
#
#python train.py \
#--do_train \
#--do_eval_during_train \
#--max_seq_length 128 \
#--model_name_or_path ../model/bert_en \
#--data_dir ../data_AAAI/Kfold/$i \
#--output_dir ../user_data/checkpoints_AAAI/bert_en_fl/$i \
#--learning_rate 4e-5
#
#done





#python predict.py \
#--vote_model_paths ../user_data/checkpoints_AAAI/bertweet_large \
#--predict_file ../data/new_val_newpoint.csv \
#--predict_result_file ../result_val_newpoint.csv



