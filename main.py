import pandas as pd
from tqdm import tqdm  # For progress bar
from src.answer_agent import AnswerAgent
from src.critique_agent import CritiqueAgent
from src.feedback_loop import feedback_loop
from src.meta_agent import MetaAgent
from src.evaluator import Evaluator
from src.naive_agent import NaiveAgent

# Load the dataset
dataset = pd.read_csv("./gpqa_main.csv")

if __name__ == "__main__":
    answer_agent = AnswerAgent()
    critique_agents = [CritiqueAgent() for _ in range(3)]
    meta_agent = MetaAgent()
    evaluator = Evaluator()
    naive_agent = NaiveAgent()

    # List to store results
    results = []

    # Display a blue progress bar with tqdm (colour argument available in Python 3.8+ with a recent tqdm version)
    for iteration, row in tqdm(
        dataset.iterrows(),
        total=len(dataset),  # Use the full dataset length
        desc="Processing",
        colour="blue",  # Set progress bar color to blue
    ):
        question = row["Question"]
        correct_answer = row["Correct Answer"]
        explanation = row["Explanation"]

        ans = naive_agent.answer(question)
        steps = ans.steps
        final_answer = ans.final_answer

        #  # Generate the final answer through the feedback loop
        #  steps, final_answer = feedback_loop(
        #      answer_agent, critique_agents, meta_agent, question
        #  )

        # Evaluation
        score = evaluator.evaluator(
            question, steps, final_answer, correct_answer, explanation
        )

        # Store the result as a dictionary
        results.append(
            {
                "question": question,
                "final_answer": final_answer,
                "correct_answer": correct_answer,
                "score": score,
            }
        )

        # Print progress log (consider using tqdm.write() to prevent overlapping with the progress bar)
        print(f"\rIteration {iteration + 1}/{len(dataset)} - Score: {score:.2f}")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Display the summary of results
    print("\n===== Results Summary =====")
    print(results_df.head())  # Display the first few rows
    print(f"Average Score: {results_df['score'].mean():.2f}")

    # Save results as a CSV file if needed
    results_df.to_csv("benchmark_results.csv", index=False)
