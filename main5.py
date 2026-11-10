from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage, HumanMessage,AIMessage

load_dotenv()

basic_model = init_chat_model(
    model="groq:qwen/qwen3.8-27b",
    temperature=0.3
)

advanced_model = init_chat_model(
    model="groq:qwen/qwen3.8-27b",
    temperature=0.7
)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 3:
        model = advanced_model
    else:
        model =basic_model

    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection]
)


response = agent.invoke({
    "messages": [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="whats 1+1."),
    ]
})

print(response['messages'][-1].content)
print(response['messages'][-1].response_metadata['model_name'])