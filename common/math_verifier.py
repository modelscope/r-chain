import logging
import re

from common.openai_api import OpenaiAPI
from common.thread_util import thread_executor

RESPONSE_PREFIX = 'Final Score:'
DEFAULT_VERIFY_PROMPT_TEMPLATE: str = (
    f'Check the similarity of the following two answers and provide a score. '
    f'The response must be concise and omit intermediary steps, '
    f'with the score ranging from [0, 1]. The final format should be, '
    f'for example: {RESPONSE_PREFIX}0.2, {RESPONSE_PREFIX}0.5, {RESPONSE_PREFIX}0.8, '
    f'and so on.')

logger = logging.getLogger(__name__)


def extract_boxed_answer(text):
    # refer: https://github.com/project-numina/aimo-progress-prize/blob/main/kaggle-solution.ipynb

    def last_boxed_only_string(text):
        idx = text.rfind('\\boxed')
        if idx < 0:
            idx = text.rfind('\\fbox')
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(text):
            if text[i] == '{':
                num_left_braces_open += 1
            if text[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            return None
        return text[idx:(right_brace_idx + 1)]

    def remove_boxed(boxed):
        left = '\\boxed{'
        try:
            assert boxed[:len(left)] == left
            assert boxed[-1] == '}'
            length = len(left)
            return boxed[length:-1]
        except Exception:
            return None

    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    answer = remove_boxed(boxed)
    return answer


def normalize_answer(answer):
    match = re.search(r'(.*?)Problem:', answer, flags=re.S)
    if match:
        answer = match.group(1)
    subs = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''),
            (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
            ('\\text{and}', ','), ('\\text{m}', '\\text{}'), ('\\le', '<')]
    remove = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f', '%'
    ]
    sub_patterns = [
        r'(\\text\{)(.*?)(\})', r'(\\textbf\{)(.*?)(\})',
        r'(\\overline\{)(.*?)(\})', r'(\\boxed\{)(.*)(\})'
    ]
    split_patterns = [
        r'finalansweris(.*)', r'answer?is:?(.*)', r'oxed\{(.*?)\}',
        r'\$(.*?)\$'
    ]
    for before, after in subs:
        answer = answer.replace(before, after)
    for expr in remove:
        answer = answer.replace(expr, '')
    for pattern in sub_patterns:
        answer = re.sub(pattern, '\\2', answer)
    for pattern in split_patterns:
        if len(re.findall(pattern, answer)) > 0:
            answer = re.findall(pattern, answer)[-1]
    answer = answer.strip()
    if 'rac' in answer and '\\frac' not in answer:
        answer = answer.replace('rac', '\\frac')
    answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', answer)
    answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', answer)
    answer = answer.replace('$', '')
    if answer.replace(',', '').isdigit():
        answer = answer.replace(',', '')
    return answer


def extract_solution_option(text):
    # i.e. (D)\\df\\frac{4ac-b^2}{4a}
    pattern = r'\((.*?)\)'
    match = re.search(pattern, text)

    if match:
        option = match.group(1)
    else:
        option = -1
    return option


def extract_answer_entry(content: str, check_last_n_chars: int = 100):
    region_to_check = content[-check_last_n_chars:]

    if ('answer is' in region_to_check or '\\boxed' in region_to_check):
        if 'boxed' in region_to_check:
            try:
                answer = normalize_answer(
                    extract_boxed_answer(region_to_check))
            except Exception:
                answer = None
        else:
            answer = normalize_answer(region_to_check)

        return answer


class MathVerifier:
    """
    Verify math answers with the following methods:
    - rule-based verification
    - model API-based verification, support multiple models voting
    """

    def __init__(self,
                 api_clients: OpenaiAPI,
                 prompt_template: str = DEFAULT_VERIFY_PROMPT_TEMPLATE,
                 **kwargs):
        self.api_clients = api_clients
        if len(self.api_clients) == 0:
            logger.warning(
                'Warning: No judge models or APIs provided. The rule-based verification will be used.'
            )

        self.prompt_template = prompt_template

    @staticmethod
    def extract_verify_score_from_response(response: str) -> float:
        reward = response.strip().split(f'{RESPONSE_PREFIX}')[-1]
        return float(reward)

    @staticmethod
    def extract_answer_from_response(text: str) -> str:
        # pattern = r'<answer>(.*?)<\/answer>'
        pattern = r'<think>(.*?)<\/think>'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            answer_content = match.group(1)
        else:
            answer_content = None

        return answer_content

    @staticmethod
    def preprocess_pair(solution: str, pred_content: str):
        solution = solution.split('\n')[-1].strip()

        pred_content = MathVerifier.extract_answer_from_response(pred_content)
        pred_content = pred_content.strip().split(
            '**Final Answer**')[-1].strip() if pred_content else None

        return solution, pred_content

    def verify_pair_with_judge_models(self, solution: str, pred_content: str):

        solution, pred_content = self.preprocess_pair(solution, pred_content)

        if not solution or not pred_content:
            return 0.0

        prompt: str = f'{self.prompt_template}\n\nANSWER1:\n{solution}\n\nANSWER2:\n{pred_content}'
        scores = []

        for client in self.api_clients:
            resp_d: dict = client.predict(prompt)
            score: float = self.extract_verify_score_from_response(
                resp_d['content'])
            scores.append(score)

        if len(scores) == 0:
            return 0.0

        # Average the scores from multiple models
        avg_score = sum(scores) / len(scores)  # todo

        return avg_score

    @staticmethod
    def verify_pair_with_rule_based(solution: str, pred_content: str):
        solution = extract_answer_entry(solution)
        pred_content = extract_answer_entry(pred_content)

        return 1.0 if solution == pred_content else 0.0

    def verify(self, data_list, max_workers=8):
        """
        Verify the math answers in parallel with multiple judge models(APIs).

        :param data_list: samples to verify,
            each sample is a dict with keys: id, source, problem, solution, messages
        :param max_workers: max workers for parallel processing
        :return:
            verified samples with column `match_score` added
        """

        @thread_executor(max_workers=max_workers)
        def call_verify_parallel(sample_d: dict):
            score: float = 0.0

            try:
                solution: str = sample_d['solution']
                pred_content: str = sample_d['messages'][-1]['content']

                score = self.verify_pair_with_rule_based(
                    solution, pred_content)

                # If rule-based extraction missed the answer,
                # try to judge the (solution, prediction) pair with models or APIs to recall verification results
                score = self.verify_pair_with_judge_models(
                    solution, pred_content) if score == 0.0 else score

            except Exception as e:
                logger.error(
                    f'Error: Got error when calling API with data id: {sample_d["id"]}, message: {e}'
                )

            sample_d['match_score'] = score

            return sample_d

        results = call_verify_parallel(data_list)

        return [item for item in results if item is not None]


if __name__ == '__main__':

    math_verifier = MathVerifier(api_clients=[], )

    # Test the verification method
    data_list = [{
        'id':
        1,
        'source':
        'test',
        'problem':
        'What is 1/2?',
        'solution':
        'The answer is \\boxed{\\frac{1}{2}}. Problem: What is 1/2?',
        'messages': [{
            'content':
            'The answer is \\boxed{\\frac{1}{3}}. Problem: What is 1/3?'
        }]
    }, {
        'id':
        2,
        'source':
        'test',
        'problem':
        'What is 1/3?',
        'solution':
        'The answer is \\boxed{\\frac{1}{3}}. Problem: What is 1/3?',
        'messages': [{
            'content':
            'The answer is \\boxed{\\frac{1}{3}}. Problem: What is 1/3?'
        }]
    }]
    verified_data = math_verifier.verify(data_list)

    import json
    print(json.dumps(verified_data, ensure_ascii=False, indent=4))
