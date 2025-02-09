from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


class NaiveAgent:
    def answer(self, question: str) -> MathReasoning:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful math tutor. Answer the question.",
                },
                {"role": "user", "content": question},
            ],
            response_format=MathReasoning,
        )
        return completion.choices[0].message.parsed
