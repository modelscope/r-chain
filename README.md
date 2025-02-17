## R-Chain: A lightweight toolkit for distilling reasoning models

Inspired by reasoning models like DeepSeek-R1 series, we put together r-chain to systematically reproduce the distillation process of reasoning models like DeepSeek-R1, for various tasks including mathematical reasoning. This effort involves several key steps and outcomes:

1. **Dataset Curation**: Curate mathematical distillation datasets, [MathR](https://www.modelscope.cn/datasets/modelscope/MathR) and [MathR-32B-Distill](https://www.modelscope.cn/datasets/modelscope/MathR-32B-Distill), which incorporate reasoning processes. These datasets shall be generated using the DeepSeek-R1 and DeepSeek-R1-Distill-Qwen-32B models, respectively.
   
2. **Training and Evaluation**: Use the curated datasets to distill a smaller dense model, such as Qwen2.5-7B-Instruction, separately. Evaluate the resulting model on reaonsing datasets, to validate the effectiveness of the curated data.

### MathR and MathR-32B-Distill Dataset Construction
1. **Problem Selection**:  Utilize publicly available dataset such as [NuminaMath-CoT](https://www.modelscope.cn/datasets/AI-MO/NuminaMath-CoT), including problems of different kinds, such as amc_aime, math, gsm8k and others.

2. **Teacher Model Inference**: We generate responses from Teacher models such as DeepSeek-R1 and DeepSeek-R1-Distill-Qwen-32B. The instruction prompt `"Please reason step by step, and put your final answer within \boxed{}."` is employed to guide and solict output from Teacher Model. After obtaining the `reasoning_content` and `content` from teacher models, we format them using the template `f'<think>{reasoning_content}</think>\n\n<answer>{content}</answer>'`. These formatted responses are then assembled into standard `messages` format, making them ready for direct use in training. All data generated in this step is progressively uploaded to `raw` subsets of the [MathR](https://www.modelscope.cn/datasets/modelscope/MathR) and [MathR-32B-Distill](https://www.modelscope.cn/datasets/modelscope/MathR-32B-Distill) datasets.

3. **Response Filtering**: Even with strong Teacher models such as DeepSeek-R1, their responses to challenging math problems may still contain errors. To address this, we employ a rule-based filtering approach to filter the `raw` datasets. We have implemented different filtering strategies tailored to the various problems in [NuminaMath-CoT](https://www.modelscope.cn/datasets/AI-MO/NuminaMath-CoT), depending on the source of the questions. The filtered data is uploaded to the `clean` subsets of MathR and MathR-32B-Distill datasets. 

### Tools for Training and Evaluation
**r-chain** is built upon existing tools such as [ms-swift](https://github.com/modelscope/ms-swift.git) and [evalscope](https://github.com/modelscope/evalscope.git) for performing supervised fine-tuning and evaluation, respectively.

#### Supervised Fine-Tuning: 
Training can be done with command 
```
bash examples/train_scripts/train_MathR-Distill-7B.sh
```
The script leverages [ms-swift](https://github.com/modelscope/ms-swift.git) and perform SFT on Qwen2.5-7B-Instruct with MathR and MathR-32B-Distill datasets. By default the training is configured to run on 8 GPUs, you may modify the script for various configurations.

### Deployment: 
Once the model is trained, you may deploy it to a vllm backend via 
```
bash examples/evaluation_scripts/deploy_MathR-Distill-7B.sh
```
This facilitates model evaluation later.
##### Evaluation：
The modle may be evaluated with [evalscope](https://github.com/modelscope/evalscope.git) with the following script:
```
python examples/evaluation_scripts/eval_MathR_Distill_7B.py
```
By default it evaulates on MATH-500 and GPQA-Diamond benchmarks, wiht evaluation metric being `Pass@1`. Each sample is generated five times and the result is the average of these five attempts.
