import os

from dotenv import load_dotenv
from smolagents.local_python_executor import LocalPythonExecutor
from smolagents import CodeAgent, HfApiModel, Tool


custom_executor = LocalPythonExecutor(["numpy"])

## ------------------------------------------------------##
def run_capture_exception(command: str):
    try:
        custom_executor(harmful_command)

    except Exception as e:
        print("ERROR:\n", e)

## ------------------------------------------------------##
!echo Bad command

## ------------------------------------------------------##
harmful_command = "!echo Bad command"

run_capture_exception(harmful_command)

## ------------------------------------------------------##
[
    're',
    'queue',
    'random',
    'statistics',
    'unicodedata',
    'itertools',
    'math',
    'stat',
    'time',
    'datetime',
    'collections',
    'numpy'
]

## ------------------------------------------------------##
harmful_command = """
                import os
                exit_code = os.system("echo Bad command")
                """

run_capture_exception(harmful_command)

## ------------------------------------------------------##
harmful_command = """
                import random
                random._os.system('echo Bad command')
                """

run_capture_exception(harmful_command)

## ------------------------------------------------------##
harmful_command = """
                while True:
                    pass
                """

run_capture_exception(harmful_command)

## ------------------------------------------------------##
custom_executor = LocalPythonExecutor(["PIL"])

harmful_command = """
                from PIL import Image

                img = Image.new('RGB', (100, 100), color = 'blue')

                i = 0
                while i < 10000:
                    img.save('simple_image_{i}.png')
                    i += 1
                """
# custom_executor(harmful_command)

## ------------------------------------------------------##
load_dotenv()

E2B_API_KEY = os.getenv("E2B_API_KEY")

## ------------------------------------------------------##
model = HfApiModel()

class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
                "Visits a webpage at the given url and reads its content as a markdown string.\
                Use this to browse webpages."
                )

    inputs = {
            "url" : {
                    "type" : "string",
                    "description" : "The url of the webpage to visit.",
                    }
            }

    output_type = "string"

    def __init__(self, max_output_length: int = 40000):
        super().__init__()
        self.max_output_length = max_output_length

    def forward(self, url: str) -> str:
        try:
            import re
            import requests

            from markdownify import markdownify
            from requests.exceptions import RequestException
            from smolagents.utils import truncate_content

        except ImportError as e:
            raise ImportError(
                            "You must install packages `markdownify` and `requests` \
                            to run this tool:\
                            for instance run `pip install markdownify requests`."
                            ) from e

        try:
            response = requests.get(url, timeout = 20)
            response.raise_for_status()  # Raise an exception for bad status codes
            markdown_content = markdownify(response.text).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."

        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"

        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

agent = CodeAgent(
                tools = [VisitWebpageTool()],
                model = model,
                executor_type = "e2b",
                executor_kwargs = {"api_key" : E2B_API_KEY},
                max_steps = 5
                )

## ------------------------------------------------------##
output = agent.run(
                "Give me one of the top github repos from organization huggingface."
                )

print("E2B executor result: ", output)
