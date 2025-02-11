import os
import jsonlines as jsonl


def jsonl_to_list(jsonl_file, feature: str = None):
    res_list = []
    with jsonl.open(jsonl_file, mode='r') as reader:
        for line in reader.iter(type=dict, allow_none=True, skip_invalid=False):
            if feature:
                if feature == line['source']:
                    res_list.append(line)
                else:
                    continue
            else:
                res_list.append(line)
    print(f'Got {len(res_list)} examples from {jsonl_file}')
    return res_list


def dump_jsonl_data(data_list, jsonl_file, dump_mode='overwrite'):
    if not jsonl_file:
        raise ValueError('output file must be provided.')

    jsonl_file = os.path.expanduser(jsonl_file)

    if not isinstance(data_list, list):
        data_list = [data_list]

    if dump_mode == 'overwrite':
        dump_mode = 'w'
    elif dump_mode == 'append':
        dump_mode = 'a'
    with jsonl.open(jsonl_file, mode=dump_mode) as writer:
        writer.write_all(data_list)
