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


class AnswerAgent:
    def answer(self, question: str, meta_knowledge: str = "") -> MathReasoning:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
                },
                {
                    "role": "system",
                    "content": f"Here is the key concepts that you learned from your experience: {meta_knowledge}",
                },
                {"role": "user", "content": question},
            ],
            response_format=MathReasoning,
        )
        return completion.choices[0].message.parsed
