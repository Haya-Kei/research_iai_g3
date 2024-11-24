from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


class MetaKnowledge(BaseModel):
    meta_knowledge: str = Field(
        description="Please briefly summarize the conversation. Sort the information from top to bottom based on the importance of the information."
    )


class MetaAgent:
    def generate_metaKnowledge(
        self, context: str, meta_knowledge: str
    ) -> MetaKnowledge:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                    ### Instructions
                                    You are highly skilled in metacognition. You excel at learning from others' feedback to improve yourself. Follow the steps below to generate new **meta_knowledge**.

                                    ### Steps
                                    1. Understand the **##context** and the **##meta_knowledge** that you have gained from past experiences.
                                    2. If **meta_knowledge** is empty, generate new **meta_knowledge** based on **##context**.
                                    3. Referring to the **##context**, update **meta_knowledge** if necessary.
                                    4. If no updates are needed, output the existing **meta_knowledge** as is.
                                    5. After completing these steps, output only the final **meta_knowledge**.

                                    ### Notes
                                    - Organize **meta_knowledge** in descending order of importance.
                                    - Summarize in concise bullet points.

                                    ### context
                                        {context}

                                    ### meta_knowledge
                                        {meta_knowledge}

                                    ### meta_knowledge (revised)
                    """,
                },
            ],
            response_format=MetaKnowledge,
        )
        return completion.choices[0].message.parsed
