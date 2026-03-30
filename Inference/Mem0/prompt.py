# SYSTEM_PROMPT = """You are a helpful assistant. You can call functions to assist with the user query. Important: You must call only one function at a time. After each function call, wait for the execution result before making the next function call if needed.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Searches for relevant information based on a query using web search.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"The search query\"}}, \"required\": [\"query\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n"""

DIRECT_PROMPT="""Based on the image, answer the question. Here is the question and image:
<image>"""

USER_PROMPT_IMAGE_SHORT="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think (Plan)**
* **This is the starting point for every turn.**
* Analyze the user's query and all available information (including previous observations) carefully.
* **Evaluate the query's difficulty and nature.** Determine if the question can be answered *directly* or if it *requires external information* (e.g., facts, real-time data).
* Formulate a plan. Your plan must decide on **one** of two courses of action:
    1.  **Call a tool:** If your evaluation shows you **need more information** (e.g., for complex, factual, or real-time questions).
    2.  **Provide a final answer:** If your evaluation shows you have **sufficient information** (e.g., for simple questions, or tasks that don't require external data).
* Your entire reasoning process must be enclosed in `<think>...</think>` tags.

**Step 2: Act (Tool Call)**
* **Execute this step ONLY if your Step 1 plan was to call a tool.**
* Call the **one single tool** decided upon in your plan.
* The tool call must be enclosed in `<tool_call>...</tool_call>` tags.
* **Important: If you call a tool, you must STOP and wait for the observation. Do NOT proceed to Step 4.**

**Step 3: Observe (Tool Output)**
* **You will only enter this step after a tool call.**
* You will receive the tool's output (observation).
* After receiving the output, you **MUST** go back to **Step 1 (Think)** to analyze the new information and decide the next step (e.g., call another tool or provide the final answer).

**Step 4: Answer (Final Response)**
* **Execute this step ONLY if your Step 1 plan was to provide a final answer.**
* In your **Step 1 Think block**, you must have already synthesized all information and planned the content of your response.
* Formulate the **clear, focused, and concise user-facing final answer** based on that plan.
* Your final answer must be enclosed in `<answer>...</answer>` tags.
{mem_prompt}
Here is the question and image:
<image>"""


USER_PROMPT_IMAGE_NEW="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think (Plan)**
* **This is the starting point for every turn.**
* Analyze the user's query and all available information (including previous observations) carefully.
* Formulate a plan. Your plan must decide on **one** of two courses of action:
    1.  **Call a tool:** If you need more information.
    2.  **Provide a final answer:** If you have sufficient information.
* Your entire reasoning process must be enclosed in `<think>...</think>` tags.

**Step 2: Act (Tool Call)**
* **Execute this step ONLY if your Step 1 plan was to call a tool.**
* Call the **one single tool** decided upon in your plan.
* The tool call must be enclosed in `<tool_call>...</tool_call>` tags.
* **Important: If you call a tool, you must STOP and wait for the observation. Do NOT proceed to Step 4.**

**Step 3: Observe (Tool Output)**
* **You will only enter this step after a tool call.**
* You will receive the tool's output (observation).
* After receiving the output, you **MUST** go back to **Step 1 (Think)** to analyze the new information and decide the next step (e.g., call another tool or provide the final answer).

**Step 4: Answer (Final Response)**
* **Execute this step ONLY if your Step 1 plan was to provide a final answer.**
* In your **Step 1 Think block**, you must have already synthesized all information and planned the content of your response.
* Formulate the **comprehensive, detailed, and user-facing final answer** based on that plan.
* Your final answer must be enclosed in `<answer>...</answer>` tags.
{mem_prompt}
Here is the question and image:
<image>
"""

USER_PROMPT_TEXT_NEW="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think (Plan)**
* **This is the starting point for every turn.**
* Analyze the user's query and all available information (including previous observations) carefully.
* Formulate a plan. Your plan must decide on **one** of two courses of action:
    1.  **Call a tool:** If you need more information.
    2.  **Provide a final answer:** If you have sufficient information.
* Your entire reasoning process must be enclosed in `<think>...</think>` tags.

**Step 2: Act (Tool Call)**
* **Execute this step ONLY if your Step 1 plan was to call a tool.**
* Call the **one single tool** decided upon in your plan.
* The tool call must be enclosed in `<tool_call>...</tool_call>` tags.
* **Important: If you call a tool, you must STOP and wait for the observation. Do NOT proceed to Step 4.**

**Step 3: Observe (Tool Output)**
* **You will only enter this step after a tool call.**
* You will receive the tool's output (observation).
* After receiving the output, you **MUST** go back to **Step 1 (Think)** to analyze the new information and decide the next step (e.g., call another tool or provide the final answer).

**Step 4: Answer (Final Response)**
* **Execute this step ONLY if your Step 1 plan was to provide a final answer.**
* In your **Step 1 Think block**, you must have already synthesized all information and planned the content of your response.
* Formulate the **comprehensive, detailed, and user-facing final answer** based on that plan.
* Your final answer must be enclosed in `<answer>...</answer>` tags.
{mem_prompt}
Here is the question:
"""




USER_PROMPT_IMAGE="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think**
- **This is the starting point for every turn.**
- Analyze the user's query carefully.
- Break down the problem and formulate a plan.
- Your entire reasoning process must be enclosed in `<think>...</think>` tags.

**Step 2: Act (Tool Call)**
- If your plan requires information you don't have, call **one single tool**.
- The tool call must be enclosed in `<tool_call>...</tool_call>` tags.
- If you can answer without tools, you must skip this step and proceed directly to Step 4.
- **Important: If you called a tool in this round, you absolutely must NOT provide a final answer in this round. You must wait for the tool's result before providing any answer.**

**Step 3: Observe and Think Again**
- **This is the critical step where you receive the tool's output and then decide what to do next.**
- After a tool call, you will receive the tool's output (observation).
- You **MUST** start a new thought process in `<think>...</think>` tags to analyze this output.
- Based on the analysis, you decide:
    - **A) Is the information insufficient?** -> Formulate the next step and go back to **Step 2** to call another tool.
    - **B) Is the information sufficient?** -> Proceed to **Step 4** to generate the final answer.

**Step 4: Final Think**
- **This is the final thinking step before generating the final answer.**
- Synthesize all available information (including the original user query, image analysis, and all tool observations).
- Plan and structure the core points and content of the final answer.
- **You must enclose this thinking process in `<think>...</think>` tags as well.**

**Step 5: Answer**
- **Execute this step only after completing the "Final Think" step.**
- Based on the plan from Step 4, formulate the final, user-facing answer.
- Your final answer must be enclosed in `<answer>...</answer>` tags, without detailed illustrations, such as <answer>beijing</answer>.
{mem_prompt}
Here is the question and image:
<image>
"""

USER_PROMPT_TEXT="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think**
- **This is the starting point for every turn.**
- Analyze the user's query carefully.
- Break down the problem and formulate a plan.
- Your entire reasoning process must be enclosed in `<think>...</think>` tags.

**Step 2: Act (Tool Call)**
- If your plan requires information you don't have, call **one single tool**.
- The tool call must be enclosed in `<tool_call>...</tool_call>` tags.
- If you can answer without tools, you must skip this step and proceed directly to Step 4.
- **Important: If you called a tool in this round, you absolutely must NOT provide a final answer in this round. You must wait for the tool's result before providing any answer.**

**Step 3: Observe and Think Again**
- **This is the critical step where you receive the tool's output and then decide what to do next.**
- After a tool call, you will receive the tool's output (observation).
- You **MUST** start a new thought process in `<think>...</think>` tags to analyze this output.
- Based on the analysis, you decide:
    - **A) Is the information insufficient?** -> Formulate the next step and go back to **Step 2** to call another tool.
    - **B) Is the information sufficient?** -> Proceed to **Step 4** to generate the final answer.

**Step 4: Final Think**
- **This is the final thinking step before generating the final answer.**
- Synthesize all available information (including the original user query, image analysis, and all tool observations).
- Plan and structure the core points and content of the final answer.
- **You must enclose this thinking process in `<think>...</think>` tags as well.**

**Step 5: Answer**
- **Execute this step only after completing the "Final Think" step.**
- Based on the plan from Step 4, formulate the final, user-facing answer.
- Your final answer must be enclosed in `<answer>...</answer>` tags,tmux set -g mouse on .
{mem_prompt}
Here is the question :
"""

USER_PROMPT_Search_r1="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think**
- **This is the starting point for every turn.**
- Analyze the user's query carefully.
- Break down the problem and formulate a plan.
- Your entire reasoning process must be enclosed in `<think>...</think>` tags.

**Step 2: Act (Tool Call)**
- If your plan requires information you don't have, call **one single tool**.
- The tool call must be enclosed in `<search> ...</search>` tags.
- If you can answer without tools, you must skip this step and proceed directly to Step 4.
- **Important: If you called a tool in this round, you absolutely must NOT provide a final answer in this round. You must wait for the tool's result before providing any answer.**

**Step 3: Observe and Think Again**
- **This is the critical step where you receive the tool's output and then decide what to do next.**
- After a tool call, you will receive the tool's output (observation).
- You **MUST** start a new thought process in `<think>...</think>` tags to analyze this output.
- Based on the analysis, you decide:
    - **A) Is the information insufficient?** -> Formulate the next step and go back to **Step 2** to call another tool.
    - **B) Is the information sufficient?** -> Proceed to **Step 4** to generate the final answer.

**Step 4: Final Think**
- **This is the final thinking step before generating the final answer.**
- Synthesize all available information (including the original user query, image analysis, and all tool observations).
- Plan and structure the core points and content of the final answer.
- **You must enclose this thinking process in `<think>...</think>` tags as well.**

**Step 5: Answer**
- **Execute this step only after completing the "Final Think" step.**
- Based on the plan from Step 4, formulate the final, user-facing answer.
- Your final answer must be enclosed in `<answer>...</answer>` tags, without detailed illustrations, such as <answer>beijing</answer>.
{mem_prompt}
Here is the question :
"""


SYSTEM_PROMPT_IMAGE  = """You are a helpful assistant. You can call functions to assist with the user query.
Important: You must call only one function at a time. After each function call, wait for the execution result before making the next function call if needed.
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Searches the web for relevant information based on the given query.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "description": "A list of complete textual queries. Each query must clearly state the topic without referring to images."}}, "required": ["query_list"]}}}
{"type": "function", "function": {"name": "web_image_to_image_search", "description": "**IMPORTANT: This tool can only be called once.** Searches for relevant images based on the original image using web search.", "parameters": {"type": "object", "properties": {"img_idx": {"type": "string", "description": "A placeholder parameter. The value must always be '0'."}}, "required": ["img_idx"]}}}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

SYSTEM_PROMPT_TEXT  = """You are a helpful assistant. You can call functions to assist with the user query.
Important: You must call only one function at a time. After each function call, wait for the execution result before making the next function call if needed.
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Searches the web for relevant information based on the given query.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "description": "A list of complete textual queries. Each query must clearly state the topic."}}, "required": ["query_list"]}}}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

#mmsearch
#{"type": "function", "function": {"name": "image_search", "description": "**IMPORTANT: This tool can only be called once.** Searches for relevant images based on the original image using web search."}}

#ours
#{"type": "function", "function": {"name": "search", "description": "Searches the web for relevant information based on the given query.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "description": "A list of complete textual queries. Each query must clearly state the topic without referring to images."}}, "required": ["query_list"]}}}
#{"type": "function", "function": {"name": "web_image_to_image_search", "description": "**IMPORTANT: This tool can only be called once.** Searches for relevant images based on the original image using web search.", "parameters": {"type": "object", "properties": {"img_idx": {"type": "string", "description": "A placeholder parameter. The value must always be '0'."}}, "required": ["img_idx"]}}}


EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


USER_PROMPT_NEW = """# Workflow and Output Format You must follow these steps in order. In every conversation turn, you start from Step 1.

Step 1: Think

This is the starting point for every turn.

Analyze the user's query carefully.

Break down the problem and formulate a plan.

You must also formulate an initial <hypothesis>...</hypothesis> based on your internal knowledge. This hypothesis is your "current best guess" which you will try to validate or refine.

Your entire reasoning process must be enclosed in <think>...</think> tags.

Example: <think> User is asking about X. My internal knowledge suggests Y. <hypothesis>The answer is Y.</hypothesis> To confirm this, I need to search for Z. </think>

Step 2: Act (Tool Call)

If your plan requires information you need to validate, refine, or complete your <hypothesis>, call one single tool.

The tool call must be enclosed in <tool_call>...</tool_call> tags.

If your <hypothesis> from Step 1 is already complete and you are confident in it without needing tools, you must skip this step and proceed directly to Step 4.

Important: If you called a tool in this round, you absolutely must NOT provide a final answer in this round. You must wait for the tool's result before providing any answer.

Step 3: Observe and Think Again

This is the critical step where you receive the tool's output and then decide what to do next.

After a tool call, you will receive the tool's output (observation).

You MUST start a new thought process in <think>...</think> tags to analyze this output.

Critically: You MUST explicitly compare the observation with your previous <hypothesis> and formulate a new, updated <hypothesis> within this <think> block.

Based on the analysis of your new hypothesis, you decide:

A) Is the new hypothesis still insufficient or unverified? -> Formulate the next step (e.g., "I need to search for W to verify this new detail") and go back to Step 2 to call another tool.

B) Is the new hypothesis sufficient and verified? -> Proceed to Step 4 to generate the final answer.

Example: <think> Observation: Search results for Z confirm Y. My previous hypothesis was <hypothesis>The answer is Y.</hypothesis>. The observation supports it. <hypothesis>The answer is Y, confirmed by search Z.</hypothesis> This hypothesis is now sufficient. </think>

Step 4: Final Think

This is the final thinking step before generating the final answer.

Synthesize all available information (including the original user query, and all tool observations), focusing on your final, most recent <hypothesis>.

Plan and structure the core points and content of the final answer based on this verified hypothesis.

You must enclose this thinking process in <think>...</think> tags as well.

Step 5: Answer

Execute this step only after completing the "Final Think" step.

Based on the plan from Step 4 (which is built from your final hypothesis), formulate the final, user-facing answer.

Your final answer must be enclosed in <answer>...</answer> tags, without detailed illustrations, such as <answer>beijing</answer>. Here is the question : """

USER_PROMPT_BC ="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think**
- **This is the starting point for every turn.**
- Analyze the user's query carefully.
- Break down the problem and formulate a plan.
- When the task requires browsing, comparison, or synthesis (browsecomp-type), you must plan for **multiple tool calls** and **multi-step reasoning**, not just one round of search.
- Think in detail about what information you need from different sources, and how you will compare or combine them.
- Your entire reasoning process must be enclosed in `<think>...</think>` tags.
- Spend enough time thinking through the structure and reasoning path; your thought should be **long, analytical, and detailed**.

**Step 2: Act (Tool Call)**
- If your plan requires information you don't have, call **one single tool**.
- For browsecomp-type tasks, you are expected to **call tools multiple times**, iteratively refining your search or focusing on different aspects of the topic.
- The tool call must be enclosed in `<tool_call>...</tool_call>` tags.
- If you can answer without tools, you must skip this step and proceed directly to Step 4.
- **Important: If you called a tool in this round, you absolutely must NOT provide a final answer in this round. You must wait for the tool's result before providing any answer.**

**Step 3: Observe and Think Again**
- **This is the critical step where you receive the tool's output and then decide what to do next.**
- After a tool call, you will receive the tool's output (observation).
- You **MUST** start a new thought process in `<think>...</think>` tags to analyze this output.
- Reflect deeply on whether the observation covers all key aspects.
- If the task is comparative or open-ended, you should expect to need **several iterations** of this step, using multiple tools or refined searches.
- Based on the analysis, you decide:
    - **A) Is the information insufficient?** -> Formulate the next step and go back to **Step 2** to call another tool.
    - **B) Is the information sufficient?** -> Proceed to **Step 4** to generate the final answer.
- Your reasoning here should be **comprehensive and multi-perspective**, not short or superficial.

**Step 4: Final Think**
- **This is the final thinking step before generating the final answer.**
- Synthesize all available information (including the original user query, image analysis, and all tool observations).
- For browsecomp tasks, carefully **compare**, **contrast**, and **evaluate** information from multiple sources.
- Plan and structure the core points and content of the final answer.
- **You must enclose this thinking process in `<think>...</think>` tags as well.**
- Take your time to produce a well-structured synthesis before writing the answer.

**Step 5: Answer**
- **Execute this step only after completing the "Final Think" step.**
- Based on the plan from Step 4, formulate the final, user-facing answer.
- Your final answer must be enclosed in `<answer>...</answer>` tags, without detailed illustrations, such as <answer>beijing</answer>.
Here is the question :
"""


#{"type": "function", "function": {"name": "search", "description": "Searches for relevant information based on a query using web search.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}}

#{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}

# {"type": "function", "function": {"name": "search", "description": "Searches for relevant information based on a query using web search.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}}
#{"type": "function", "function": {"name": "image_search", "description": "Searches for relevant images based on the image using web search.", "parameters": {"type": "object", "properties": {"img_idx": {"type": "number", "description": "The index of the image (starting from 0)"}}, "required": ["img_idx"]}}}

# {"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": ["string", "array"], "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit"}, "goal": {"type": "string", "description": "The goal of the visit for webpage(s)"}}, "required": ["url", "goal"]}}}
# {"type": "function", "function": {"name": "image_search", "description": "Searches for relevant images based on cache data. Returns both text descriptions and actual images.", "parameters": {"type": "object", "properties": {"image_url": {"type": "string", "description": "Query image URL or identifier"}, "cache_id": {"type": "string", "description": "Cache ID corresponding to the subfolder name in cache folder"}}, "required": ["image_url", "cache_id"]}}}