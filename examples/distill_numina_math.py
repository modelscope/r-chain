import os
import time

from common.file_util import dump_jsonl_data, jsonl_to_list
from common.openai_api import OpenaiAPI
from math_distillation.process_numina_math import NuminaMath


def load_data(path, feature: str = None):
    """
    Load data from jsonl file.

    :param path: input file path, jsonl format.
    :param feature:  Filter samples with column `source` in the NuminaMath dataset, e.g. 'aops_forum', 'olympiads', 'amc_aime', ...
        default is None, which means no filter.
    :return: list of sample dict
    """
    return jsonl_to_list(path, feature)


def process_numina_data(api_client:OpenaiAPI, in_file: str, out_file: str, feature: str = None, batch_size: int = 256, max_workers: int = 8):

    if not api_client:
        raise ValueError('API client must be provided.')

    numina_data_list = load_data(in_file, feature)

    # Check if the output file exists, and reuse the existing ids.
    exists_ids_list = []
    if os.path.exists(out_file):
        numina_output_list = jsonl_to_list(out_file)
        exists_ids_list = [item['id'] for item in numina_output_list]

    exists_ids_list = set(exists_ids_list)

    numina_data_list = [item for item in numina_data_list if item['id'] not in exists_ids_list]
    print(f'After removing existing ids, got {len(numina_data_list)} examples to process.')

    # Process data in batches
    batches = []
    for i in range(0, len(numina_data_list), batch_size):
        batch = numina_data_list[i:i + batch_size]
        batches.append(batch)

    del numina_data_list

    # Initialize the NuminaMath process
    numina_math = NuminaMath(api_client)

    t1 = time.time()
    for idx, batch_list in enumerate(batches):
        print(f'Processing batch {idx} ...')

        results = numina_math.process_all(batch_list, max_workers=max_workers)
        print(f'Got {len(results)} results from service.')

        print(f'Dump results of batch-{idx} to output file: {out_file}')
        dump_jsonl_data(results, out_file, dump_mode='append')

        print(f'>>Cumulative time cost: {time.time() - t1}')



if __name__ == '__main__':

    # Example: Distill the reasoning process for the Numina Math dataset using DeepSeek-R1 model on Alibaba cloud Bailian LLMs platform.
    # Reference: https://www.aliyun.com/product/bailian

    numina_input_file = 'YOUR_NUMINA_MATH_DATASET.jsonl'
    numina_conversations_out = 'results/YOUR_NUMINA_MATH_DATASET_deepseek_r1_results.jsonl'
    # You may use any OpenAI-API compatible service
    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    # you may set the api_key here or use environment variable `OPENAI_API_KEY`
    api_key = 'YOUR_API_KEY'

    client = OpenaiAPI(
        model='deepseek-r1',  # or 'deepseek-r1-distill-qwen-32b'
        base_url=base_url,
        api_key=api_key,
        stream=True,
    )

    process_numina_data(
        client,
        in_file=numina_input_file,
        out_file=numina_conversations_out,
        # filter by `source` column,
        # e.g. 'aops_forum', 'olympiads', 'amc_aime', ... default is `None` to select all samples.
        feature='aops_forum'
    )
