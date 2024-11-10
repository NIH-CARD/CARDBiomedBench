
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