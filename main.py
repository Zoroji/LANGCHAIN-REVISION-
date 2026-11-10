import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool("locate_user", description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "ABC123":
            return "Dehradun"
        case "XYZ456":
            return "London"
        case "HJKL111":
            return "Paris"
        case _:
            return "Unknown"


@tool("get_weather", description="Get the current weather for a given city")
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    data = response.json()
    current = data["current_condition"][0]

    return {
        "city": city,
        "temperature_celsius": float(current["temp_C"]),
        "temperature_fahrenheit": float(current["temp_F"]),
        "humidity": float(current["humidity"]),
        "condition": current["weatherDesc"][0]["value"]
    }


model = init_chat_model(
    model="groq:openai/gpt-oss-20b",
    temperature=0.7)

checkpointer  = InMemorySaver()


agent = create_agent(
    model=model,
    tools=[locate_user, get_weather],
    system_prompt="You are a weather assistant who cracks jokes while being helpful. If the user doesn't mention a city, use locate_user to find it.",
    context_schema=Context,
    # response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id': '1'}}

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What's the weather today?"}
        ]
    },
    config=config,
    context=Context(user_id="ABC123")
)

print(response["messages"][-1].content)