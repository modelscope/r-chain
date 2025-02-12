nproc_per_node=8
batch_size=128
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'modelscope/MathR-32B-Distill:clean' \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr $batch_size / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --deepspeed zero3