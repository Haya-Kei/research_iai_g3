from src.answer_agent import AnswerAgent
from src.critique_agent import CritiqueAgent
from src.feedback_loop import feedback_loop
from src.meta_agent import MetaAgent

question = """
Problem Statement

Let  T(n)  be a transformation function defined on a positive integer  n  as follows:


T(n) =
\\begin{cases}
\\frac{n}{2} & \\text{if } n \\text{ is even}, \\\\
3n + 1 & \\text{if } n \\text{ is odd}.
\\end{cases}


Define the sequence  \\{ a_k \\}  such that  a_0 = n  and  a_{k+1} = T(a_k) . The Collatz Conjecture asserts that for every positive integer  n , there exists a finite integer  k  such that  a_k = 1 .

Prove or disprove the Collatz Conjecture by addressing the following:
"""


if __name__ == "__main__":
    #  question = input(
    #      "Enter a math question (e.g., Solve for x in the equation x^3 - 6x^2 + 11x - 6 = 0):\n> "
    #  )

    question = question

    answer_agent = AnswerAgent()
    critique_agents = [CritiqueAgent() for _ in range(3)]
    meta_agent = MetaAgent()

    final_answer = feedback_loop(answer_agent, critique_agents, meta_agent, question)
    print("\nFinal Answer:", final_answer)
