import os
from evalscope.collections.sampler import WeightedSampler
from evalscope.collections.schema import CollectionSchema, DatasetInfo
from evalscope.utils.io_utils import dump_jsonl_data

schema = CollectionSchema(name='DeepSeekDistill', datasets=[
            CollectionSchema(name='Math', datasets=[
                    DatasetInfo(name='math_500', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
                    DatasetInfo(name='gpqa', weight=1, task_type='math', tags=['en'],  args={'subset_list': ['gpqa_diamond'], 'few_shot_num': 0}),
            ])
        ])

print(schema.to_dict())
print(schema.flatten())

#  get the mixed data
mixed_data = WeightedSampler(schema).sample(100000)  # set a large number to ensure all datasets are sampled
os.makedirs('outputs', exist_ok=True)
dump_jsonl_data(mixed_data, 'outputs/MathR-Distill-Qwen2.5-7B-Instruct.jsonl')

from evalscope import TaskConfig, run_task
#  start the task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='MathR-Distill-Qwen2.5-7B-Instruct',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,
    datasets=[
        'data_collection',
    ],
    dataset_args={
        'data_collection': {
            'local_path': 'outputs/MathR-Distill-Qwen2.5-7B-Instruct.jsonl'
        }
    },
    eval_batch_size=16,
    repeat=5,  # num of samples per request
    generation_config={
        'max_tokens': 28000,  # avoid exceed max length
        'temperature': 0.6,
        'top_p': 0.95,
    },
)

run_task(task_cfg=task_cfg)
