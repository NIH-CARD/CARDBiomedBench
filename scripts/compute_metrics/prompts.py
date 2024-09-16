
def biomedical_grading_prompt(question: str, gold_res: str, pred_res: str) -> str:
    """
    Generate a prompt for BioScore grading.
    """
    prompt = (
        "### Scoring Instructions for Evaluating Analyst Responses\n\n"
        "**Objective:** Evaluate an analyst's response against a gold standard.\n\n"
        "**Scoring Criteria:**\n"
        "- **Exact Match:** 3 points for an exact or equally accurate response.\n"
        "- **Close Match:** 2 points for a very close response with minor inaccuracies.\n"
        "- **Partial Match:** 1 point for a partially accurate response with significant omissions.\n"
        "- **Irrelevant Information (Harmless):** Deduct 0.5 points for harmless irrelevant information.\n"
        "- **Irrelevant Information (Distracting):** Deduct 1 point for distracting irrelevant information.\n"
        "- **No Match:** 0 points for no match.\n"
        "- **Not Knowing Response:** -1 point for stating lack of knowledge or abstaining. An example of this scenario is when Analyst Response says \'There are various studies, resources or databases on this topic that you can check ... but I do not have enough information on this topic.\'\n\n"
        "**Scoring Process:**\n"
        "1. **Maximum Score:** 3 points per question.\n"
        "2. **Calculate Score:** Apply criteria to evaluate the response.\n\n"
        f"**Question:** {question}\n"
        f"**Golden Answer:** {gold_res}\n"
        f"**Analyst Response:** {pred_res}\n"
        "## Your grading\n"
        "Using the scoring instructions above, grade the Analyst Response return only the numeric score on a scale from 0.0-3.0. If the response is stating lack of knowledge or abstaining, give it -1.0."
    )
    return prompt

def biomedical_grading_prompt_old(question: str, gold_res: str, pred_res: str) -> str:
    """
    Generate a prompt for LLM grading.

    Parameters:
    - question (str): The question being evaluated.
    - gold_res (str): The reference (gold) response.
    - pred_res (str): The predicted response.

    Returns:
    - str: The grading prompt.
    """
    prompt = (
        "Scoring Instructions for Evaluating Analyst Responses\n"
        "Objective: The goal is to evaluate the performance of an analyst's response to a question compared to a gold standard response.\n"
        "Scoring Criteria:\n"
        "Exact or Equal Match: Award 3 points if the analyst's response exactly matches the gold standard response or gives an equally accurate one.\n"
        "Close Match: Award 2 points if the analyst's response is very close to the gold standard response but may contain minor inaccuracies or omissions.\n"
        "Partial Match: Award 1 point if the analyst's response partially matches the gold standard response, but significant information is missing or inaccurate.\n"
        "Irrelevant Information Penalty (Harmless): Deduct 0.5 points if the analyst's response includes irrelevant but relatively harmless information.\n"
        "Irrelevant Information Penalty (Distracting): Deduct 1 point if the analyst's response includes hallucinated or totally off-base irrelevant information that significantly distracts from the quality of the response.\n"
        "No Match: Give a score of 0 if the analyst's response doesn't match the gold standard response at all.\n"
        "Not Knowing Response: Assign a score of -1.0 if the analyst responds that it doesn't know the answer or abstains to answer based on its knowledge.\n"
        "Scoring Process:\n"
        "Define Maximum Achievable Score: The maximum amount of points for an answer to a question is 3 points.\n"
        "Calculate Score: Apply the scoring criteria to evaluate the analyst's response to the question.\n"
        "Error Handling: Anticipate potential errors or edge cases during the scoring process and handle them appropriately.\n\n"
        f"Question: {question}\n"
        f"Golden Answer: {gold_res}\n"
        f"Analyst Response: {pred_res}\n"
        "Grade the analyst response using the scoring instructions above, return only the numeric score on a scale from 0.0-3.0 or -1.0.\n"
    )
    return prompt