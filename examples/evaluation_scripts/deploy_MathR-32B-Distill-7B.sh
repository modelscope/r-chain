# replace --model with your own checkpoint path
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model /output/Qwen2.5-7B-Instruct/vxx-xxxx-xxxx/checkpoint-xxx \
    --infer_backend vllm \
    --served_model_name MathR-32B-Distill-Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.9 \
    --port 8802 \
