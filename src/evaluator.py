from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


class Score(BaseModel):
    score: int = Field(
        description="Evaluate the given problem and its corresponding answer, and grade the response based on its correctness and completeness from 0 to 100"
    )


class Evaluator:
    def evaluator(
        self,
        question,
        steps,
        final_output: str,
        answer_correct: str,
        explanation,
    ) -> Score:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                    ### Instructions  
                                        You are an evaluator responsible for grading answers based on their correctness and completeness. Your task is to assess the final output by comparing it with the expected correct answer and the given problem.

                                    ### Steps  
                                        1. Given the question, review the **final_output**, which is the submitted answer or solution.  
                                        2. Compare the final output with the **answer_correct**, which represents the expected correct answer.  
                                        3. Assign a **score from 0 to 100** based on:  
                                        - **Correctness**: Does the final output match the correct answer?  
                                        - **Completeness**: Is the answer fully elaborated, covering all necessary steps?  
                                        - **Clarity**: Is the reasoning and logic clear and easy to follow?  

                                    ### Notes  
                                        - A **perfect score (100)** should be given only if the answer is entirely correct, well-structured, and complete.  
                                        - A **score of 0** should be given if the answer is entirely incorrect or missing.  
                                        - Partial scores should be assigned based on how much of the solution is correct and well-formed.

                                    ### question
                                        {question}

                                    ### steps
                                        {steps}

                                    ### final_output
                                        {final_output}

                                    ### answer_correct
                                        {answer_correct}

                                    ### explanation
                                        {explanation}
                    """,
                },
            ],
            response_format=Score,
        )
        return completion.choices[0].message.parsed.score
