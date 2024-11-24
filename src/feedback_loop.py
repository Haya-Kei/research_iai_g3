def feedback_loop(
    answer_agent, critique_agents, meta_agent, question, confidence_threshold=0.8
):
    solution = answer_agent.answer(question)
    iteration = 1
    meta_knowledge = ""

    while True:
        print(f"\nIteration {iteration}: Evaluating Solution")
        all_feedback = []
        all_confidence = []

        # Verbose output: Show the initial solution steps
        print("\nInitial Solution Steps:")
        for i, step in enumerate(solution.steps):
            print(f"  Step {i + 1}: {step.explanation} -> {step.output}")

        # Evaluate each step
        for index, step in enumerate(solution.steps):
            step_feedback = []
            step_confidences = []
            print(f"\nEvaluating Step {index + 1}:")
            print(f"  Explanation: {step.explanation}")
            print(f"  Output: {step.output}")

            # Each critique agent evaluates the current step
            for agent_number, agent in enumerate(critique_agents, start=1):
                critique = agent.critique_step(question, solution.steps, index)
                step_feedback.append(critique.feedback)
                step_confidences.append(critique.confidence_score)

                # Verbose output: Individual critique agent's feedback
                print(f"    Critique Agent {agent_number}:")
                print(f"      - Feedback: {critique.feedback}")
                print(f"      - Confidence Score: {critique.confidence_score:.2f}")

            # Calculate average confidence for this step
            step_confidence = sum(step_confidences) / len(step_confidences)
            print(f"  Average Confidence for Step {index + 1}: {step_confidence:.2f}")
            all_confidence.append(step_confidence)

            # Display feedback if confidence is below threshold
            if step_confidence < confidence_threshold:
                print(f"  Feedback for Step {index + 1} (confidence below threshold):")
                for feedback in step_feedback:
                    print(f"    - {feedback}")
                all_feedback.extend(step_feedback)

        # Check if all steps meet confidence threshold
        if all(conf >= confidence_threshold for conf in all_confidence):
            print("\nSolution is sufficiently confident. Final Answer:")
            print(solution.final_answer)
            return solution.final_answer

        # If not confident, provide feedback and re-ask
        feedback_text = "\n".join(all_feedback)
        refined_question = (
            f"{question}\nConsider the following feedback:\n{feedback_text}"
        )
        meta_knowledge = meta_agent.generate_metaKnowledge(
            feedback_text, meta_knowledge
        )
        print(f"Meta knowledge:\n{meta_knowledge}\n")
        solution = answer_agent.answer(refined_question, meta_knowledge)

        iteration += 1
        print("\n" + "=" * 40 + "\n")
