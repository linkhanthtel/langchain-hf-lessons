import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description='Return weather information of a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
    model = 'ollama:llama3.2',
    tools = [get_weather],
    system_prompt = 'You are a helpful assistant who always care about environmental issues.'
)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like in Singapore?'}
    ]
})

# print(response)
print(response["messages"][-1].content)

