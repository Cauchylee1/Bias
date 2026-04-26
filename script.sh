#data
PYTHONPATH=/root/autodl-tmp/BiasAlert python data/data_precessing/generate_passage_embeddings.py --model_name_or_path ./contriever --output_dir data/embeddings --passages data/retrieval/bias_doc_gender_race_religion_orientation.tsv --shard_id 0 --num_shards 1

python data/data_precessing/run_retrieval.py --data code/LLaMA-Factory/data/retrieval_outputs/test_data.json --passages data/retrieval/bias_doc_gender_race_religion_orientation.tsv --passages_embeddings "data/embeddings/*" --output_dir code/LLaMA-Factory/data/retrieval_outputs --output_name test_data_rag.json --model_name_or_path ./contriever --n_docs 5
#train
cd /root/autodl-tmp/BiasAlert/code/LLaMA-Factory && export OMP_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python src/train_bash.py --stage sft --do_train --model_name_or_path /root/autodl-tmp/models/Llama-2-7b-chat-hf --dataset bias_intruc --dataset_dir /root/autodl-tmp/BiasAlert/code/LLaMA-Factory/data/retrieval_outputs --template default --finetuning_type lora --lora_target all --output_dir /root/autodl-tmp/BiasAlert/code/LLaMA-Factory/saves/LLaMA2-7B-BiasAlert/lora/sft --overwrite_cache --overwrite_output_dir --cutoff_len 1024 --preprocessing_num_workers 16 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_steps 100 --eval_steps 1000 --evaluation_strategy steps  --learning_rate 5e-5 --num_train_epochs 10.0  --val_size 0.1 --plot_loss --fp16 --lora_rank 16 --weight_decay 0.05

#inference
python generate_responses.py --prompt /root/autodl-tmp/BiasAlert/data/prompt.txt --model /root/autodl-tmp/models/Llama-2-7b-chat-hf --adapter /root/autodl-tmp/BiasAlert/code/LLaMA-Factory/saves/LLaMA2-7B-BiasAlert/lora/sft --dataset /root/autodl-tmp/BiasAlert/code/LLaMA-Factory/data/retrieval_outputs/test_data_rag.json --save_path code/LLaMA-Factory/data/retrieval_outputs/pred --batch_size 16 --load_8bit

#eval
python eval_metrics.py --file_path code/LLaMA-Factory/data/retrieval_outputs/pred/test_data_rag_Llama-2-7b-chat-hf_prompt.json --save_path ./code/LLaMA-Factory/data/retrieval_outputs/pred/eval_metrics.json


Number of unequal elements (usable subset): 552others: 0.00,Efficacy Score: Accuracy: 0.82, Precision(biased): 0.90, Recall(biased): 0.74, F1(biased): 0.81,Classification Score: 0.85,Attribution Score: 0.93,Over-Safety Score (OS, usable ratio): 1.00,

