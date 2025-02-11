from common.openai_api import OpenaiAPI
from common.thread_util import thread_executor


def get_prompt_template(question: str):
    # NOTE: only for math problems
    # Reference: https://github.com/deepseek-ai/DeepSeek-R1
    prompt_template: str = '\nPlease reason step by step, and put your final answer within \boxed{}.'

    return f'{question}{prompt_template}'


def get_system_prompt():
    # Note: unused for calling the API
    # Reference: https://github.com/deepseek-ai/DeepSeek-R1
    return (
        'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
        'The assistant first thinks about the reasoning process in the mind and then provides the user '
        'with the answer. The reasoning process and answer are enclosed '
        'within <think> </think> and <answer> </answer> tags, respectively, '
        'i.e., <think> reasoning process here </think> <answer> answer here </answer>.'
    )


class NuminaMath(object):

    def __init__(self, api_client: OpenaiAPI, **kwargs):

        self.api_client = api_client

    def process_all(self, data_list, max_workers=8):

        @thread_executor(max_workers=max_workers)
        def call_api_parallel(sample_d: dict):
            # sample_d keys: id, source, problem, solution, messages

            prompt: str = get_prompt_template(sample_d['problem'])
            origin_messages = sample_d.get('messages', [])

            try:
                resp = self.api_client.predict(prompt)
            except Exception as e:
                print(
                    f'Got error when calling API with data id: {sample_d["id"]}, error message: {e}'
                )
                return None

            reasoning_content: str = resp['reasoning_content']
            content: str = resp['content']

            resp_content: str = f'<think>{reasoning_content}</think>\n\n<answer>{content}</answer>'

            messages = [{
                'role': 'system',
                'content': get_system_prompt()
            }, {
                'role': 'user',
                'content': prompt
            }, {
                'role': 'assistant',
                'content': resp_content
            }]

            # rename the `messages` to `origin_messages`
            sample_d['messages'] = messages
            sample_d['origin_messages'] = origin_messages
            sample_d['model'] = self.api_client.model
            sample_d['generation_config'] = {}  # TODO: add generation config
            sample_d['usage'] = resp['usage']

            return sample_d

        results = call_api_parallel(data_list)
        return [item for item in results if item]
