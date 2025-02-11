import os

from openai import OpenAI


class OpenaiAPI(object):

    def __init__(self, model: str, base_url: str, api_key: str = None, **kwargs):

        self.model = model

        self.client = OpenAI(
            api_key=api_key if api_key else os.getenv('OPENAI_API_KEY'),
            base_url=base_url,
            **kwargs
        )

    def predict(self, prompt: str, stream=True):
        """
        Call OpenAI API to get the completion of the prompt

        :param prompt: prompt for the question
        :param stream: stream the response or not, default is True
        :return:
            response: dict, {'reasoning_content': str, 'content': str, 'usage': dict}
        """

        if stream:
            stream_args: dict = dict(
                stream_options=dict(
                    include_usage=True,
                )
            )
        else:
            stream_args = {}

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # {'role': 'system', 'content': get_system_prompt()},
                {'role': 'user', 'content': prompt}
            ],
            stream=stream,
            **stream_args,
        )

        if stream:
            reasoning_content = ""
            answer_content = ""
            usage: dict = {}

            for chunk in completion:
                chunk_choices = chunk.choices
                if chunk.usage:
                    usage = chunk.usage.to_dict()
                if len(chunk_choices) == 0:
                    continue

                reasoning_chunk = chunk_choices[0].delta.reasoning_content if hasattr(chunk_choices[0].delta, 'reasoning_content') else ""
                answer_chunk = chunk_choices[0].delta.content
                if reasoning_chunk:
                    reasoning_content += reasoning_chunk
                elif answer_chunk:
                    answer_content += answer_chunk

            resp = dict(
                reasoning_content=reasoning_content,
                content=answer_content,
                usage=usage,
            )
        else:
            resp = dict(
                reasoning_content=completion.choices[0].message.reasoning_content,
                content=completion.choices[0].message.content,
                usage=completion.usage.to_dict(),
            )

        return resp