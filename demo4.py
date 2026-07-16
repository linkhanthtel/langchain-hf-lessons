from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

basic_model_call = init_chat_model(model='ollama:llama3.2')
advanced_model_call = init_chat_model(model='ollama:ornith')

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state['messages'])

    if message_count < 3:
        model = basic_model_call
    else:
        model = advanced_model_call

    request.model = model

    return handler(request)

agent = create_agent(model=basic_model_call, middleware=[dynamic_model_selection])

response = agent.invoke({
    'message': [
        SystemMessage('You are a help AI assistant'),
        HumanMessage('What is a car?')
    ]
})

print(response['message'][-1].content)
print(response['message'][-1].response_metadata['model_name'])