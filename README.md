## R-Chain: A lightweight reasoning data processing toolkit for r1/o1-like models.

Inspired by reasoning models like DeepSeek-R1, we aim to systematically reproduce the distillation process of the DeepSeek-R1 model for mathematical tasks. This effort involves two key steps and outcomes:

1. **Dataset Generation**: Create mathematical distillation datasets, [MathR](https://www.modelscope.cn/datasets/modelscope/MathR) and [MathR-32B-Distill](https://www.modelscope.cn/datasets/modelscope/MathR-32B-Distill), which incorporate reasoning processes. These datasets will be generated using the DeepSeek-R1 and DeepSeek-R1-Distill-Qwen-32B models.
   
2. **SFT Training and Evaluation**: Use these two datasets to distill the Qwen2.5-7B-Instruction model, separately, to validate the effectiveness of the generated data.

### MathR and MathR-32B-Distill Dataset Construction
1. **Problem Selection**: We utilize source mathematical problems from publicly available dataset [NuminaMath-CoT](https://www.modelscope.cn/datasets/AI-MO/NuminaMath-CoT), which includes multiple source, such as amc_aime, math, gsm8k and others.

2. **Teacher Model Inference**: We generate responses for the source problems using the DeepSeek-R1 and DeepSeek-R1-Distill-Qwen-32B models via API inference. The instruction prompt `"Please reason step by step, and put your final answer within \boxed{}."` is employed to guide the model's output. After obtaining the `reasoning_content` and `content` from teacher models, we format them using the template `f'<think>{reasoning_content}</think>\n\n<answer>{content}</answer>'`. These formatted responses are then assembled into the standard `messages` format, making them ready for direct use in training. All data generated in this step is progressively uploaded to `raw` subsets of the MathR and MathR-32B-Distill datasets. Source code is on the way.

3. **Response Filtering**: Although the two teacher models already possess strong capabilities, their responses to challenging math problems still contain errors. To address this, we employ a rule-based filtering approach to filter the `raw` datasets. We have implemented different filtering strategies tailored to the various problems in [NuminaMath-CoT](https://www.modelscope.cn/datasets/AI-MO/NuminaMath-CoT), depending on the source of the questions. The filtered data is uploaded to the `clean` subsets of MathR and MathR-32B-Distill datasets. Source code is on the way.

### SFT Training and Evaluation
While progressively generating distillation datasets, we use [ms-swift](https://github.com/modelscope/ms-swift.git) and [evalscope](https://github.com/modelscope/evalscope.git) to perform supervised fine-tuning and evaluation on the Qwen2.5-7B-Instruct model to verify their effectiveness.
1. **Supervised Fine-Tuning**: [ms-swift](https://github.com/modelscope/ms-swift.git) has supported SFT of Qwen2.5-7B-Instruct on MathR and MathR-32B-Distill datasets, run `bash examples/train_scripts/train_MathR-Distill-7B.sh` perform training with 8 GPUs.
2. **Deployment**: After training, run `bash examples/evaluation_scripts/deploy_MathR-Distill-7B.sh` to deploy the model with a vllm backend.
3. **Evaluation**: After deployment, run `python examples/evaluation_scripts/eval_MathR_Distill_7B.py` to evaluate the model on the MATH-500 and GPQA-Diamond benchmarks. The evaluation metric is `Pass@1`. Each sample is generated five times and the result is the average of these five attempts.
