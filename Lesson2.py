import os
import io
import IPython.display as display
import base64
import warnings
import pandas as pd
import numpy as np

from PIL import Image
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from smolagents import tool
from smolagents import HfApiModel, CodeAgent
from smolagents import ToolCallingAgent
from helper import get_huggingface_token


warnings.filterwarnings("ignore")
_ = load_dotenv()
login(os.environ['HF_API_KEY'])

## ------------------------------------------------------##
suppliers_data = {
                "name": [
                        "Montreal Ice Cream Co",
                        "Brain Freeze Brothers",
                        "Toronto Gelato Ltd",
                        "Buffalo Scoops",
                        "Vermont Creamery",
                        ],
                "location": [
                            "Montreal, QC",
                            "Burlington, VT",
                            "Toronto, ON",
                            "Buffalo, NY",
                            "Portland, ME",
                            ],
                "distance_km" : [120, 85, 400, 220, 280],
                "canadian" : [True, False, True, False, False],
                "price_per_liter" : [1.95, 1.91, 1.82, 2.43, 2.33],
                "tasting_fee" : [0, 12.50, 30.14, 42.00, 0.20],
                }

data_description = """Suppliers have an additional tasting fee:\
                    that is a fixed fee applied to each order to taste the ice cream."""
suppliers_df = pd.DataFrame(suppliers_data)

suppliers_df

## ------------------------------------------------------##
def calculate_daily_supplier_price(row):
    order_volume = 30

    product_cost = row["price_per_liter"] * order_volume

    trucks_needed = np.ceil(order_volume / 300)
    cost_per_km = 1.20
    transport_cost = row["distance_km"] * cost_per_km * trucks_needed

    tariff = product_cost * np.pi / 50 * row["canadian"]

    total_cost = product_cost + transport_cost + tariff + row["tasting_fee"]

    return total_cost


suppliers_df["daily_price"] = suppliers_df.apply(calculate_daily_supplier_price, axis = 1)

display(suppliers_df)

suppliers_df = suppliers_df.drop("daily_price", axis = 1)

## ------------------------------------------------------##
@tool
def calculate_transport_cost(distance_km: float, order_volume: float) -> float:
    """
    Calculate transportation cost based on distance and order size.
    Refrigerated transport costs $1.2 per kilometer and has a capacity of 300 liters.

    Args:
        distance_km: the distance in kilometers
        order_volume: the order volume in liters
    """
    trucks_needed = np.ceil(order_volume / 300)
    cost_per_km = 1.20

    return distance_km * cost_per_km * trucks_needed


@tool
def calculate_tariff(base_cost: float, is_canadian: bool) -> float:
    """
    Calculates tariff for Canadian imports. Returns the tariff only, not the total cost.
    Assumes tariff on dairy products from Canada is worth 2 * pi / 100, approx 6.2%

    Args:
        base_cost: the base cost of goods, not including transportation cost.
        is_canadian: wether the import is from Canada.
    """
    if is_canadian:
        return base_cost * np.pi / 50

    return 0

## ------------------------------------------------------##
calculate_transport_cost.description

## ------------------------------------------------------##
model = HfApiModel("Qwen/Qwen2.5-72B-Instruct", provider = "together", temperature = 0.1)

## ------------------------------------------------------##
agent = CodeAgent(
                model = model,
                tools = [calculate_transport_cost, calculate_tariff],
                max_steps = 10,
                additional_authorized_imports = ["pandas", "numpy"],
                verbosity_level = 2
                )

agent.logger.console.width = 66

## ------------------------------------------------------##
agent.run(
        """Can you get me the transportation cost for 50 liters
        of ice cream over 10 kilometers?"""
        )

## ------------------------------------------------------##
task = """Here is a dataframe of different ice cream suppliers.
        Could you give me a comparative table (as a dataframe) of the total
        daily price for getting daily ice cream delivery from each of them,
        given that we need exactly 30 liters of ice cream per day? Take
        into account transportation cost and tariffs.
        """

agent.logger.level = 1

agent.run(
        task,
        additional_args = {"suppliers_data" : suppliers_df,
                           "data_description" : data_description},
        )

## ------------------------------------------------------##
model = HfApiModel("Qwen/Qwen2.5-72B-Instruct", temperature = 0.6)

agent = ToolCallingAgent(
                        model = model,
                        tools = [calculate_transport_cost, calculate_tariff],
                        max_steps = 20,
                        )

agent.logger.console.width = 66

## ------------------------------------------------------##
output = agent.run(
                    task,
                    additional_args = {"suppliers_data" : suppliers_df,
                                    "data_description" : data_description},
                    )

print(output)
