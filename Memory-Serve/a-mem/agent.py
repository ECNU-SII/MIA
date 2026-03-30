import json
import os
import json
import argparse
from flask import Flask, request, jsonify

from memory import AgenticMemorySystem, LLMController

agent = None

app = Flask(__name__)

class advancedMemAgent:
    def __init__(self, model, backend, retrieve_k, temperature_c5, sglang_host="http://localhost", sglang_port=30000):
        self.memory_system = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port
        )
        self.retriever_llm = LLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
    
    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)
    
    def retrieve_memory(self, content, k=3):
        return self.memory_system.find_related_memories_raw(content, k=k)
    
    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}
                
                But please do not directly output {{"keywords": "keyword1, keyword2, keyword3"}}."""

        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response


    def memory_context(self, question: str) -> str:
        """Generate answer for a question given the conversation context."""
        keywords = self.generate_query_llm(question)
        context = self.retrieve_memory(keywords, k=self.retrieve_k)
        return context
    
    
    def answer_question(self, question: str) -> str:
        """Generate answer for a question given the conversation context."""
        keywords = self.generate_query_llm(question)
        context = self.retrieve_memory(keywords, k=self.retrieve_k)
        user_prompt = f"Based on the context: {context}, answer the following question. {question}"
        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}, temperature=self.temperature_c5
        )
        return response, user_prompt, context


# agent = advancedMemAgent(
#     model='qwen3-32b',
#     backend='sglang',
#     retrieve_k=3,
#     temperature_c5=0.5
# )

# memory_ids = []
# memory_ids.append(agent.add_memory("Neural networks are composed of layers of neurons that process information."))
# memory_ids.append(agent.add_memory("Data preprocessing involves cleaning and transforming raw data for model training."))
# memory_ids.append(agent.add_memory("Backpropagation is a key algorithm used to train neural networks by updating weights based on error gradients."))
# memory_ids.append(agent.add_memory("Feature engineering helps models learn more effectively by selecting and transforming relevant input features."))
# memory_ids.append(agent.add_memory("Deep learning models typically require large amounts of labeled data for supervised training."))
# memory_ids.append(agent.add_memory("Normalization and standardization are common preprocessing steps to stabilize model training."))
# memory_ids.append(agent.add_memory("Overfitting occurs when a model learns training data too well but fails to generalize to unseen data."))
# memory_ids.append(agent.add_memory("Convolutional neural networks are especially effective for image and spatial data processing."))
# memory_ids.append(agent.add_memory("Attention mechanisms allow models to focus on the most relevant parts of the input data."))
# memory_ids.append(agent.add_memory("Data augmentation improves model robustness by artificially increasing training data diversity."))
# memory_ids.append(agent.add_memory("Loss functions quantify the difference between model predictions and ground truth labels."))
# memory_ids.append(agent.add_memory("Unsupervised learning discovers patterns in data without explicit labels."))
# print("\nQuerying for related memories...")

# prediction, user_prompt, raw_context = agent.answer_question(question="What is the normalization and standardization?")
# print(prediction)
# print(user_prompt)


@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    global agent
    del agent
    agent = advancedMemAgent(
        model='qwen',
        backend='openai',
        retrieve_k=3,
        temperature_c5=0.5
    )
    return jsonify({"status": "success", "message": "Agent reinitialized"})



@app.route('/add_memory', methods=['POST'])
def add_memory():
    try:
        data = request.get_json()
        question = data['question']
        messages = data['messages'][1:]
        content = f"### Question: {question}\n"
        j = 1
        for message in messages:
            if message["role"] == "assistant":
                content += f"### Round {j}:\n"
                content += f"#### Agent Reasoning and Tool Call:\n{message['content']}\n"
                j += 1
            else:
                content += f"#### Tool Call Return Results:\n{message['content']}\n"
        agent.add_memory(content)
        return jsonify({"message": "Memory added"})
    except:
        return jsonify({"message": "false"})


@app.route('/memory_context', methods=['POST'])
def memory_context():
    data = request.get_json()
    question = data['question']
    context = agent.memory_context(question)
    return jsonify({"context": context})



if __name__ == '__main__':
    agent = advancedMemAgent(
        model='qwen',
        backend='openai',
        retrieve_k=3,
        temperature_c5=0.5
    )
    app.run(host='0.0.0.0', port=5000)