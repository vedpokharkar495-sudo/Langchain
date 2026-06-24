

# Integration with anthropic API

from langchain_anthropic import ChatAnthropic
from typing_extensions import Annotated

model = ChatAnthropic(model="claude-haiku-4-5-20251001")


def get_weather(
    location: Annotated[str, ..., "Location as city and state."]
) -> str:
    """Get the weather at a location."""
    return "It's sunny."


model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("Which city is hotter today: LA or NY?")
response.content