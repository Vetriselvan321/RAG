"""LLM generation with pluggable free providers.

Supported providers:
  gemini — Google Gemini via google-genai.
  groq   — Groq cloud via groq SDK.

For deployed BYOK usage, the API key can be supplied per request.
"""

import os
import re
from typing import Iterator

from .config import GEMINI_API_KEY, LLM_MODEL

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


class Generator:

    def __init__(
        self,
        model: str = LLM_MODEL,
        provider: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider or LLM_PROVIDER
        self.model = model

        # Use the key supplied for this request.
        # If no key was supplied, fall back to the local .env key.
        if self.provider == "groq":
            key = api_key or GROQ_API_KEY

            if not key:
                raise ValueError(
                    "Groq API key not set. Add your key in Settings."
                )

            from groq import Groq

            self._client = Groq(api_key=key)

        elif self.provider == "gemini":
            key = api_key or GEMINI_API_KEY

            if not key:
                raise ValueError(
                    "Gemini API key not set. Add your key in Settings."
                )

            from google import genai

            self._client = genai.Client(api_key=key)

        else:
            raise ValueError(
                f"Unknown LLM provider '{self.provider}'. "
                "Choose 'gemini' or 'groq'."
            )

    def _get_groq_model(self) -> str:
        """Return a valid Groq model name."""

        if (
            "llama" in self.model
            or "mixtral" in self.model
            or "gemma" in self.model
            or self.model.startswith("openai/")
        ):
            return self.model

        return "openai/gpt-oss-20b"

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
    ) -> str:
        """Generate a grounded answer from query + retrieved chunks."""

        context = "\n\n---\n\n".join([
            f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
            for c in context_chunks
        ])

        prompt = f"""Answer the question using ONLY the context below.
If the context does not contain the answer, say so honestly.
Cite sources by name when you use them.

Context:
{context}

Question: {query}

Answer:"""

        if self.provider == "groq":
            response = self._client.chat.completions.create(
                model=self._get_groq_model(),
                messages=[{"role": "user", "content": prompt}],
            )

            return response.choices[0].message.content

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text

    def generate_with_citations(
        self,
        query: str,
        context_chunks: list[dict],
    ) -> "CitedAnswer":
        """Generate a grounded answer with inline [N] citation markers."""

        from .models import CitedAnswer

        context_parts = [
            f"[{i}] [Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
            for i, c in enumerate(context_chunks, 1)
        ]

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Answer the question using ONLY the context below.
After each fact you use, add an inline citation like [1] or [2] that refers to the context number.
If the context does not contain the answer, say so honestly.

Context:
{context}

Question: {query}

Answer (with inline citations):"""

        if self.provider == "groq":
            response = self._client.chat.completions.create(
                model=self._get_groq_model(),
                messages=[{"role": "user", "content": prompt}],
            )

            answer_text = response.choices[0].message.content

        else:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            answer_text = response.text

        cited_nums = sorted(
            set(
                int(n)
                for n in re.findall(r"\[(\d+)\]", answer_text)
            )
        )

        citations = [
            context_chunks[n - 1]
            for n in cited_nums
            if 1 <= n <= len(context_chunks)
        ]

        return CitedAnswer(
            answer=answer_text,
            citations=citations,
            query=query,
        )

    def generate_stream(
        self,
        query: str,
        context_chunks: list[dict],
    ) -> Iterator[str]:
        """Stream the answer token-by-token."""

        context_parts = [
            f"[{i}] [Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
            for i, c in enumerate(context_chunks, 1)
        ]

        context = (
            "\n\n---\n\n".join(context_parts)
            if context_parts
            else "(No context retrieved)"
        )

        prompt = f"""Answer the question using ONLY the context below.
After each fact you use, add an inline citation like [1] or [2] that refers to the context number.
If the context does not contain the answer, say so honestly.

Context:
{context}

Question: {query}

Answer (with inline citations):"""

        if self.provider == "groq":
            stream = self._client.chat.completions.create(
                model=self._get_groq_model(),
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content

                if delta:
                    yield delta

        else:
            for chunk in self._client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
            ):
                if chunk.text:
                    yield chunk.text