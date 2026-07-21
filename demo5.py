from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic being researched")
    summary: str = Field(description="A clear, concise summary of the topic")
    sources: list[str] = Field(description="Useful sources or references")
    tools_used: list[str] = Field(description="Tools used, or empty list if none")


agent = create_agent(
    model="ollama:llama3.2",
    tools=[],
    system_prompt=(
        "You are a research assistant that helps users learn about a topic.\n"
        "When answering:\n"
        "1. Identify the main topic\n"
        "2. Write a clear, concise summary\n"
        "3. List useful sources (real URLs or well-known references when possible)\n"
        "4. List any tools you used (or an empty list if none)"
    ),
    response_format=ResearchResponse,
)

raw_response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the capital of Myanmar?"}
        ]
    }
)

structured = raw_response.get("structured_response")
if structured is not None:
    print("Topic:", structured.topic)
    print("Summary:", structured.summary)
    print("Sources:", structured.sources)
    print("Tools used:", structured.tools_used)
else:
    print(raw_response["messages"][-1].content)