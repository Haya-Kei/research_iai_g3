def feedback_loop(
    answer_agent, critique_agents, meta_agent, question, confidence_threshold=0.8
):
    solution = answer_agent.answer(question)
    iteration = 1
    meta_knowledge = ""

    while True:
        all_feedback = []
        all_confidence = []

        # Evaluate each step
        for index, step in enumerate(solution.steps):
            step_feedback = []
            step_confidences = []

            # Each critique agent evaluates the current step
            for agent_number, agent in enumerate(critique_agents, start=1):
                critique = agent.critique_step(question, solution.steps, index)
                step_feedback.append(critique.feedback)
                step_confidences.append(critique.confidence_score)

                # Verbose output: Individual critique agent's feedback

            # Calculate average confidence for this step
            step_confidence = sum(step_confidences) / len(step_confidences)

            # Display feedback if confidence is below threshold
            if step_confidence < confidence_threshold:
                all_feedback.extend(step_feedback)

        # Check if all steps meet confidence threshold
        if all(conf >= confidence_threshold for conf in all_confidence):
            return solution.steps, solution.final_answer

        # If not confident, provide feedback and re-ask
        feedback_text = "\n".join(all_feedback)
        refined_question = (
            f"{question}\nConsider the following feedback:\n{feedback_text}"
        )
        meta_knowledge = meta_agent.generate_metaKnowledge(
            feedback_text, meta_knowledge
        )
        solution = answer_agent.answer(refined_question, meta_knowledge)

        iteration += 1
