from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest,ModelResponse, dynamic_prompt

load_dotenv()

@dataclass
class Context:
    user_role:str


@dynamic_prompt
def user_role_prompt(request:ModelRequest) ->str:
    user_role = request.runtime.context.user_role

    base_prompt = 'You are an helpful assistant'

    match user_role:
        case 'expert':
            return base_prompt + 'provide detail and technical explanations.'
        case 'beginner':
            return base_prompt + ' keep basic explanation and simple.'
        case 'child':
            return base_prompt + 'Explain everything as if you are talking to a 5 year old'
        case _:
            return base_prompt

agent  = create_agent(
    model="groq:qwen/qwen3.8-27b",
    middleware=[user_role_prompt],
    context_schema=Context,
)

response = agent.invoke({
    'messages':[{'role':'user','content':'Explain PCA.'}],
},context=Context(user_role='child')
)

print(response)