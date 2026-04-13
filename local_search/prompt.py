SYSTEM_PLAN_PROMPT = """
You’re a planning assistant in a three-step loop:  

1. **Plan**: Given a goal and background info, output a clear action plan.  
2. **Evaluate**: Given an execution trace, decide if replanning is needed.  
3. **Replan (if needed)**: With new reference memories, provide a revised plan targeting unmet goals.  

Keep responses concise and action-focused.
"""

PLAN_PROMPT = """
You are a memory-based planning assistant assisting an agent by providing strategic guidance. 

The agent has access to the following tools:
- `search`: perform text-based web queries to retrieve external information.

### [Relevant Memories] 
Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Your Task:
- Analyze the **Question** and compare it with both **correct strategies** and **incorrect patterns** in the memories.
- If similar successful approaches exist, adapt their core idea to guide the next steps.
- If the context resembles past failures, highlight what to avoid and suggest corrective actions.
- Recommend **one clear, generalizable work plan or action strategy** that directly addresses the current situation and guides agents step-by-step on what to do.

### Output should:
1. Be clear and concise—no more than 400 words, and present your response as a step-by-step plan.
2. Each step in the plan must be atomic and actionable, specifying a single operation such as invoking a tool (e.g., `search` with precise query intent), performing logical inference, executing a calculation, cross-verifying facts, or synthesizing prior observations.
3. You must include relevant memories in the thinking process (<think>...</think>), but do not mention them in the final output.
4. Don't try to give the answer directly, but give a plan.
5. Prohibit the generation of content unrelated to the plan.
6. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.

**[Question]** (Global Objective): 
{question}
"""

PLAN_PROMPT_IMG = """
You are a memory-based planning assistant assisting an agent by providing strategic guidance. 

The agent has access to the following tools:
- `web_image_to_image_search`: find visually similar images online (Can only be used once).
- `search`: perform text-based web queries to retrieve external information.

### [Relevant Memories] 
Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Your Task:
- Analyze the **Question** and compare it with both **correct strategies** and **incorrect patterns** in the memories.
- If similar successful approaches exist, adapt their core idea to guide the next steps.
- If the context resembles past failures, highlight what to avoid and suggest corrective actions.
- Recommend **one clear, generalizable work plan or action strategy** that directly addresses the current situation and guides agents step-by-step on what to do.

### Output should:
1. Be clear and concise—no more than 300 words, and present your response as a step-by-step plan.
2. Each step in the plan must be atomic and actionable, specifying a single operation such as invoking a tool (e.g., `search` with precise query intent), performing logical inference, executing a calculation, cross-verifying facts, or synthesizing prior observations.
3. You must include relevant memories in the thinking process (<think>...</think>), but do not mention them in the final output.
4. Don't try to give the answer directly, but give a plan.
5. Prohibit the generation of content unrelated to the plan.
6. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.

**IMPORTANT: The `web_image_to_image_search` tool can only be called once.** Otherwise, the agent will be severely penalized.

**[Question]** (Global Objective): 
{question}
"""

REPLAN_PROMPT = """
### Reflection and Replanning

In the last conversation, you indicated that reflect and replan needed to be performed. Now you need to perform them.

**The original question:** {question}

**The only tool you can recommend is `search`.**

Your task is to:
- Analyze the memories and reference their useful strategies.
- Analyze the current workflow so far to understand what has been attempted and why it failed.
- Recommend a **clear, generalizable work plan or action strategy** that builds on existing work, avoids past mistakes, and addresses the remaining challenges.

### Critical Requirements:
- **Leverage all completed steps from the current workflow**. Do not repeat searches, queries, or reasoning already performed.
- **Identify the exact failure point** and propose the next logical step toward the goal.

### Output should:
1. Be clear and concise, no more than 200 words, and present your response as a step-by-step supplementary plan.
2. You must include **reflection** on the previous failure in the thinking process (<think>...</think>), but **do not mention them in the final output**.
3. Don't give the answer directly, but provide **a supplementary plan**.
4. Prohibit generating content unrelated to the plan.
5. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.
"""

JUDGE_PROMPT = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I’m not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
‘‘‘
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don’t matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it’s either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don’t know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It’s possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama’s child is named James. However, it’s recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
‘‘‘
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i’m not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia and Sasha
Predicted answer 1: I don’t know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I’m not sure about the other one.
‘‘‘
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.
    
Also note the following things:
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey’s Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include " California".
    - Consider the question "What award did A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL ’24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
- Do not give credit for an answer if it contains any internal inconsistency.
    - For example, consider the question: "How many NBA players have scored 60 or more points in a regular season game since 2024?" with the gold answer "8". A response is INCORRECT if it states "8 players" but lists 7 or 9, or if it initially says "8 players" but later contradicts this by concluding 7 or 9.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don’t apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
‘‘‘
Question: {question}
Gold target: {correct_answer}
Predicted answer: {response}
‘‘‘
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
"""

JUDGE_PROMPT_NOGT = """
Your job is to evaluate a predicted answer based solely on its reasoning trace—no ground truth is provided. You must assign a grade of ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"] by judging whether the reasoning is logically sound, factually plausible, and sufficiently supported (e.g., by cited tool calls or reliable inference).

Below are minimal examples illustrating the grading criteria:

**CORRECT example**:  
Question: What is the capital of France?  
Predicted answer: Paris  
Reasoning trace: I recall from general knowledge that Paris is the capital of France. This is consistent across geography textbooks and official sources. No conflicting information exists.

→ Grade: A  
(Reasoning is coherent, factually accurate, and draws on reliable common knowledge.)

**INCORRECT example**:  
Question: What is the capital of Australia?  
Predicted answer: Sydney  
Reasoning trace: Sydney is the largest and most famous city in Australia, so it must be the capital.

→ Grade: B  
(Reasoning contains a factual misconception—confusing prominence with capital status—and lacks tool use or correction.)

**NOT_ATTEMPTED example**:  
Question: When was the first moon landing?  
Predicted answer: I don’t know the exact date.  
Reasoning trace: This is a historical fact that I’m not confident about, so I won’t guess.

→ Grade: C  
(No attempt to reason toward an answer; explicitly declines to provide one.)

**Grading principles**:  
- **A (CORRECT)**: Reasoning is logically consistent, avoids factual errors, and either uses accurate common knowledge or cites sufficient evidence from tool calls.  
- **B (INCORRECT)**: Reasoning contains clear factual misunderstandings, unsupported assumptions, or logical flaws—even if the final answer seems plausible.  
- **C (NOT_ATTEMPTED)**: The response avoids committing to an answer and provides no substantive reasoning toward one.

Now evaluate the following:

‘‘‘
Question: {question}
Predicted answer: {response}
Reasoning trace: {trace}
‘‘‘

Respond only with "A", "B", or "C".
"""