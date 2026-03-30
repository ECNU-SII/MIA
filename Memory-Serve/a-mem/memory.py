import json
import os
import requests
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Literal, Any
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI



class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass


class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt", api_key: Optional[str] = None):
        self.model = "qwen"
        self.client = OpenAI(base_url="http://localhost:8002/v1", api_key="EMPTY")



    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model="qwen",
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            max_tokens=10000
        )
        return response.choices[0].message.content


class SGLangController(BaseLLMController):
    def __init__(self, model: str = "qwen3-32b", sglang_host: str = "http://localhost", sglang_port: int = 30000):
        self.model = model
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.base_url = f"{sglang_host}:{sglang_port}"

    def _generate_empty_value(self, schema_type: str) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"])

        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            json_schema = response_format.get("json_schema", {}).get("schema", {})
            json_schema_str = json.dumps(json_schema)

            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": 1000,
                    "json_schema": json_schema_str  # SGLang expects JSON schema as string
                }
            }

            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                # SGLang returns the generated text in 'text' field
                generated_text = result.get("text", "")
                return generated_text
            else:
                print(f"SGLang server returned status {response.status_code}: {response.text}")
                raise Exception(f"SGLang server error: {response.status_code}")

        except Exception as e:
            print(f"SGLang completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)


class LLMController:
    def __init__(self,
                 backend: Literal["openai", "sglang"] = "sglang",
                 model: str = "qwen3-32b",
                 api_key: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        self.llm = OpenAIController(model=model, api_key=api_key)
        # if backend == "openai":
        #     self.llm = OpenAIController(model=model, api_key=api_key)
        # elif backend == "sglang":
        #     self.llm = SGLangController(model=model, sglang_host=sglang_host, sglang_port=sglang_port)
        # else:
        #     raise ValueError(f"Backend {backend} not supported")


class MemoryNote:
    """Basic memory unit with metadata"""
    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None):
        self.content = content

        if llm_controller and any(param is None for param in [keywords, context, tags]):
            analysis = self.analyze_content(content, llm_controller)
            # print("analysis:", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.tags = tags or []

    def analyze_content(self, content: str, llm_controller: LLMController) -> Dict:
        """Analyze content to extract keywords, context, and other metadata"""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content

        response = llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "context": {
                            "type": "string",
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                    },
                    "required": ["keywords", "context", "tags"],
                    "additionalProperties": False
                },
                "strict": True
            }
        })

        try:
            response_cleaned = response.strip()
            if not response_cleaned.startswith('{'):
                start_idx = response_cleaned.find('{')
                if start_idx != -1:
                    response_cleaned = response_cleaned[start_idx:]
            if not response_cleaned.endswith('}'):
                end_idx = response_cleaned.rfind('}')
                if end_idx != -1:
                    response_cleaned = response_cleaned[:end_idx + 1]
            analysis = json.loads(response_cleaned)

            return analysis

        except json.JSONDecodeError as e:
            print(f"JSON parsing error in analyze_content: {e}")


class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer("/your_path/models/all-MiniLM-L6-v2")
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}

    def add_documents(self, documents: List[str]):
        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.append(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx +idx

    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        if not self.corpus:
            return []
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices


class AgenticMemorySystem:
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "qwen3-32b",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000
                 ):
        self.memories = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, sglang_host, sglang_port)
        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents(["content:" + note.content + " context:" + note.context + " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)])
        # if evo_label:
        #     self.evo_cnt += 1
        #     if self.evo_cnt % self.evo_threshold == 0:
        #         self.consolidate_memories()
        return note.id

    def process_memory(self, note: MemoryNote) -> bool:
        """Process a memory note and return an evolution label"""
        try:
            neighbor_memory, indices = self.find_related_memories(note.content, k=2)
            prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories=neighbor_memory,neighbor_number=len(indices))
        except:
            prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories="", neighbor_number=0)
        # print("prompt_memory: ", prompt_memory)
        response = self.llm_controller.llm.get_completion(prompt_memory, response_format={"type": "json_schema", "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "should_evolve": {
                                "type": "boolean",
                            },
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "suggested_connections": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                }
                            },
                            "new_context_neighborhood": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "tags_to_update": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "new_tags_neighborhood": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        },
                        "required": ["should_evolve","actions","suggested_connections","tags_to_update","new_context_neighborhood","new_tags_neighborhood"],
                        "additionalProperties": False
                    },
                    "strict": True
                }}
        )
        try:
            # print("response: ", response, type(response))
            response_cleaned = response.strip()
            print(response_cleaned)
            if not response_cleaned.startswith('{'):
                start_idx = response_cleaned.find('{')
                if start_idx != -1:
                    response_cleaned = response_cleaned[start_idx:]
            if not response_cleaned.endswith('}'):
                end_idx = response_cleaned.rfind('}')
                if end_idx != -1:
                    response_cleaned = response_cleaned[:end_idx+1]
            response_json = json.loads(response_cleaned)
        # print("response_json", response_json, type(response_json))
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # print(f"Raw response: {response}")
            # Return default values for failed parsing
            # return False, note
        should_evolve = response_json['should_evolve']
        if should_evolve:
            actions = response_json['actions']
            for action in actions:
                if action == "strengthen":
                    suggest_connections = response_json['suggested_connections']
                    new_tags = response_json["tags_to_update"]
                    note.links.extend(suggest_connections)
                    note.tags = new_tags
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json['new_context_neighborhood']
                    new_tags_neighborhood = response_json['new_tags_neighborhood']
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    # print("indices", indices)
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        temp_memory_idx = indices[i]
                        temp_note = noteslist[temp_memory_idx]
                        temp_note.tags = tag
                        temp_note.context = context
                        self.memories[notes_id[temp_memory_idx]] = temp_note
        return should_evolve, note

    def find_related_memories(self, query: str, k: int=5) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += "memory index:" + str(i) + "\t talk start time:" + all_memories[i].timestamp + "\t memory content: " + all_memories[i].content + "\t memory context: " + all_memories[i].context + "\t memory keywords: " + str(all_memories[i].keywords) + "\t memory tags: " + str(all_memories[i].tags) + "\n"
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int=2) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return "", []
        
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        j = 0
        for i in indices:
            memory_str +=  "memory content: " + all_memories[i].content + "memory context: " + all_memories[i].context + "memory keywords: " + str(all_memories[i].keywords) + "memory tags: " + str(all_memories[i].tags) + "\n"
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                # if all_memories[neighbor].content not in memory_str:
                memory_str += "memory content: " + all_memories[neighbor].content + "memory context: " + all_memories[neighbor].context + "memory keywords: " + str(all_memories[neighbor].keywords) + "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                if j >= k:
                    return memory_str
                j += 1

