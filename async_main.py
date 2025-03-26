import argparse
import asyncio
import dotenv
from agent.async_agent import AsyncAgent
from async_computers import LocalPlaywrightComputer
import openai
import typing
import os
import httpx


dotenv.load_dotenv()

class AsyncOpenAiModel:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def __call__(
        self,
        inputs: typing.List[typing.Dict],
        tools: typing.List[typing.Dict[str, typing.Any]],
        model: str = "computer-use-preview"  # Updated to a chat model
    ) -> typing.Any:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "input": inputs,
                    "tools": tools,
                    "truncation": "auto",
                },
            )
            print("##RAW: ",response.text)
            response.raise_for_status()
        return response.json()


async def main(args:argparse.Namespace):
    model = AsyncOpenAiModel(api_key=os.getenv("OPENAI_API_KEY"))

    async with LocalPlaywrightComputer() as computer:
        agent = AsyncAgent(computer=computer, model=model)
        query = [{"role": "user", "content": args.query}]
        output_items = await agent.run_full_turn(query)

    print(output_items)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="Collect the youtube video titles of the latest videos from the channel 'OpenAI'")
    asyncio.run(main(parser.parse_args()))
