from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


class CritiqueResponse(BaseModel):
    feedback: str
    confidence_score: float


class CritiqueAgent:
    def critique_step(
        self, question: str, steps: list, step_index: int
    ) -> CritiqueResponse:
        # Build the message with all steps up to the current step for logical context
        all_steps = "\n".join(
            [
                f"Step {i+1}: {step.explanation} -> {step.output}"
                for i, step in enumerate(steps[: step_index + 1])
            ]
        )

        critique_message = f"Evaluate the reasoning and correctness of the following steps in solving this math problem: {question}\n{all_steps}"

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a critical reviewer of math solutions. Provide feedback and a confidence score (0 to 1) for each step.",
                },
                {"role": "user", "content": critique_message},
            ],
            response_format=CritiqueResponse,
        )
        return completion.choices[0].message.parsed
