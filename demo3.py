from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

load_dotenv()

@dataclass
class Context:
    user_role: str 

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful and excellent assistant"

    match user_role:
        case "expert":
            return f'{base_prompt} Provide detail technical responses.'
        case "regular":
            return f'{base_prompt} Keep your explanation simple and basic'
        case "beginner":
            return f'{base_prompt} Explain using easy words as you are talking to a dummy'
        case _:
            return base_prompt

agent = create_agent(
    model = 'ollama:llama3.2',
    middleware = [user_role_prompt],
    context_schema = Context
)

response = agent.invoke({
    'message': [{'role': 'user', 'content': 'Explain RAG'}]
}, context = Context(user_role='beginner'))

print(response)