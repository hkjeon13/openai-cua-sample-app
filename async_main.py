import argparse
import asyncio
import dotenv
from agent.async_agent import AsyncAgent
from computers import LocalPlaywrightComputer
import openai
import typing
import os

dotenv.load_dotenv()

class AsyncOpenAiModel:
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def __call__(
            self,
            messages:typing.List[typing.Dict],
            tools:typing.List[typing.Dict[str, typing.Any]],
            model: str = "computer-use-preview"
    ) -> typing.Coroutine[str]:

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
        )

        return response.choices[0].message.text


async def main(args:argparse.Namespace):
    model = AsyncOpenAiModel(api_key=os.getenv("OPENAI_API_KEY"))

    with LocalPlaywrightComputer() as computer:
        agent = AsyncAgent(computer=computer, model=model)
        query = [{"role": "user", "content": args.query}]
        output_items = await agent.run_full_turn(query, debug=True, show_images=True)

    print(output_items)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="How do you think about the climate change?")
    asyncio.run(main(parser.parse_args()))
