
R-Chain: A lightweight reasoning data processing toolkit for r1/o1-like models.

## SFT training and evaluation
We use [ms-swift](https://github.com/modelscope/ms-swift.git) and [evalscope](https://github.com/modelscope/evalscope.git) to perform SFT on the `Qwen2.5-7B-Instruct` model to verify the effectiveness of the `MathR` and `MathR-32B-Distill` datasets.
* Run `bash examples/train_scripts/train_MathR-Distill-7B.sh` to do SFT.
* After training, run `bash examples/evaluation_scripts/deploy_MathR-Distill-7B.sh` to deploy the model.
* After deployment, run `python examples/evaluation_scripts/eval_MathR_Distill_7B.py` to evaluate the model.