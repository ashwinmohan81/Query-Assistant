from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import httpx

import config


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...

    def parse_query_response(self, text: str) -> dict:
        """Extract query and explanation from LLM response."""
        sql_match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL)
        gql_match = re.search(r"```graphql\s*(.*?)```", text, re.DOTALL)
        query = ""
        query_type = "sql"
        if sql_match:
            query = sql_match.group(1).strip()
            query_type = "sql"
        elif gql_match:
            query = gql_match.group(1).strip()
            query_type = "graphql"
        else:
            lines = text.strip().split("\n")
            code_lines = [
                l for l in lines
                if any(kw in l.upper() for kw in ["SELECT", "FROM", "WHERE", "JOIN", "GROUP", "ORDER", "LIMIT", "query ", "{", "}"])
            ]
            if code_lines:
                query = "\n".join(code_lines)

        explanation_match = re.search(
            r"(?:explanation|description|note)[:\s]*(.*?)(?:```|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        if not explanation:
            non_query = text
            for block in re.findall(r"```.*?```", text, re.DOTALL):
                non_query = non_query.replace(block, "")
            explanation = non_query.strip()

        return {"query": query, "query_type": query_type, "explanation": explanation}


class OllamaProvider(LLMProvider):
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]


class BedrockProvider(LLMProvider):
    def __init__(self):
        import boto3
        self.client = boto3.client(
            "bedrock-runtime", region_name=config.BEDROCK_REGION
        )
        self.model_id = config.BEDROCK_MODEL_ID

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        })
        resp = self.client.invoke_model(modelId=self.model_id, body=body)
        result = json.loads(resp["body"].read())
        return result["content"][0]["text"]


class MockProvider(LLMProvider):
    """Returns template-based responses for testing without any model."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "```sql\nSELECT * FROM mock_table WHERE 1=1 LIMIT 10;\n```\n\n"
            "This is a mock response. Connect an LLM provider (Ollama or Bedrock) "
            "for real query generation."
        )


def get_provider() -> LLMProvider:
    providers = {
        "ollama": OllamaProvider,
        "bedrock": BedrockProvider,
        "mock": MockProvider,
    }
    cls = providers.get(config.LLM_PROVIDER)
    if cls is None:
        raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")
    return cls()
