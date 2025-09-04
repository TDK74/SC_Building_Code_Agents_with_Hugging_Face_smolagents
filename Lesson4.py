PROJECT_NAME = "Customer-Success"

## ------------------------------------------------------##
import os
import phoenix as px
import pandas as pd
import json

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from smolagents import HfApiModel, CodeAgent
from smolagents import tool
from typing import Dict


tracer_provider = register(
                        project_name = PROJECT_NAME,
                        #endpoint = get_phoenix_endpoint() + "v1/traces"
                        endpoint = os.getenv('DLAI_LOCAL_URL').format(port = '6006') + "v1/traces"
                        )

SmolagentsInstrumentor().instrument(tracer_provider = tracer_provider)

## ------------------------------------------------------##
load_dotenv()
login(os.getenv('HF_API_KEY'))

## ------------------------------------------------------##
model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", provider = "together")
model([{"role" : "user", "content" : "Hello!"}])

## ------------------------------------------------------##
print(os.environ.get('DLAI_LOCAL_URL').format(port = '6006'))

## ------------------------------------------------------##
agent = CodeAgent(model = model, tools = [])

## ------------------------------------------------------##
agent.run("What is the 100th Fibonacci number?")

## ------------------------------------------------------##
print(os.environ.get('DLAI_LOCAL_URL').format(port = '6006'))

## ------------------------------------------------------##
menu_prices = {"crepe nutella" : 1.50, "vanilla ice cream" : 2, "maple pancake" : 1.}

ORDER_BOOK = {}

@tool
def place_order(quantities: Dict[str, int], session_id: int) -> None:
    """Places a pre-order of snacks.
    Args:
        quantities: a dictionary with names as keys and quantities as values
        session_id: the id for the client session
    """
    global ORDER_BOOK
    assert isinstance(quantities, dict), "Incorrect type for the input dictionary!"
    assert [key in menu_prices for key in quantities.keys()],\
            f"All food names should be within {menu_prices.keys()}"

    ORDER_BOOK[session_id] = quantities


@tool
def get_prices(quantities: Dict[str, int]) -> str:
    """Gets price for certain quantities of ice cream.
    Args:
        quantities: a dictionary with names as keys and quantities as values
    """
    assert isinstance(quantities, dict), "Incorrect type for the input dictionary!"
    assert [key in menu_prices for key in quantities.keys()],\
            f"All food names should be within {menu_prices.keys()}"

    total_price = sum([menu_prices[key] * value for key, value in quantities.items()])

    return (
            f"Given the current menu prices:\n{menu_prices}\n"
            f"The total price for your order would be: ${total_price}"
            )

## ------------------------------------------------------##
order_agent = CodeAgent(
                        tools = [place_order, get_prices],
                        model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct",
                                           provider = "together")
                        )

## ------------------------------------------------------##
order_agent.run(
                "Could I come and collect one crepe nutella?",
                additional_args = {"session_id" : 192}
                )

## ------------------------------------------------------##
client_requests = [
                ("Could I come and collect one crepe nutella?", "place_order"),
                ("What would be the price for 1 crêpe nutella + 2 pancakes?", "get_prices"),
                ("How did you start your ice-cream business?", None),
                ("What's the weather at the Louvre right now?", None),
                ("I'm not sure if I should order. I want a vanilla ice cream, \
                but if it's more expensive than $1, I don't want it. \
                If it's below, I'll order it, please.", "place_order")
                ]

## ------------------------------------------------------##
for request in client_requests:
    order_agent.run(
                    request[0],
                    additional_args = {"session_id" : 0, "menu_prices" : menu_prices}
                    )

## ------------------------------------------------------##
spans = px.Client().get_spans_dataframe(project_name = PROJECT_NAME)
spans.head(20)

## ------------------------------------------------------##
agents = spans[spans['span_kind'] == 'AGENT'].copy()
agents['task'] = agents['attributes.input.value'].apply(
                                                        lambda x: json.loads(x).get('task') \
                                                            if isinstance(x, str) else None
                                                        )

tools = spans.loc[
                spans['span_kind'] == 'TOOL',
                ["attributes.tool.name", "attributes.input.value", "context.trace_id"]
                ].copy()

tools_per_task = agents[
                        ["name", "start_time", "task", "context.trace_id"]
                        ].merge(
                                tools,
                                on = "context.trace_id",
                                how = "left",
                                )

tools_per_task.head()

## ------------------------------------------------------##
def score_request(expected_tool: str, tool_calls: list):
    if expected_tool is None:
        return tool_calls == set(["final_answer"])

    else:
        return expected_tool in tool_calls

results = []

for request, expected_tool in client_requests:
    tool_calls = set(tools_per_task.loc[tools_per_task["task"] == request,
                                        "attributes.tool.name"].tolist())

    results.append(
                    {
                    "request" : request,
                    "tool_calls_performed" : tool_calls,
                    "is_correct" : score_request(expected_tool, tool_calls)
                    }
                )

pd.DataFrame(results)

