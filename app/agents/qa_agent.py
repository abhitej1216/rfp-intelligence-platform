from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator

from dotenv import load_dotenv

from app.agents.state import AgentState

load_dotenv()
logger = logging.getLogger(__name__)


class QAAgent:
    def answer(self, question: str, state: AgentState) -> AgentState:
        snippets, citations, raw_evidence = self.collect_evidence(state.evidence, question)
        if not snippets:
            state.final_output = {
                "question": question,
                "answer": "Not found in documents.",
                "rephrased_answer": "Not found in documents.",
                "raw_evidence": "",
                "sources": [],
                "rephrase_source": "none",
            }
            return state

        rephrased_parts: list[str] = []
        rephrase_source = "offline"
        for event in self.iter_rephrase_events(question, snippets):
            if event["type"] == "token":
                rephrased_parts.append(event["text"])
            elif event["type"] == "source":
                rephrase_source = event["source"]

        rephrased = self._normalize_rephrased("".join(rephrased_parts), question, snippets, rephrase_source)
        if rephrase_source == "gemini" and rephrased.startswith("I could not find a direct answer"):
            rephrase_source = "gemini+offline"

        blended = self._blend_answer(rephrased, snippets)
        state.final_output = {
            "question": question,
            "answer": blended,
            "rephrased_answer": rephrased,
            "raw_evidence": raw_evidence,
            "sources": citations,
            "rephrase_source": rephrase_source,
        }
        return state

    def collect_evidence(
        self,
        evidence: list[dict],
        question: str,
        *,
        max_snippets: int = 5,
    ) -> tuple[list[str], list[dict], str]:
        question_terms = {term.casefold() for term in re.findall(r"[A-Za-z0-9]+", question) if len(term) > 3}
        ranked = sorted(
            evidence,
            key=lambda item: sum(term in item.get("text", "").casefold() for term in question_terms),
            reverse=True,
        )
        citations: list[dict] = []
        evidence_snippets: list[str] = []
        for item in ranked:
            text = " ".join(item.get("text", "").split())
            if not text:
                continue
            metadata = item.get("metadata") or {}
            evidence_snippets.append(text[:600])
            citations.append(
                {
                    "file": metadata.get("source_filename", "Unknown source"),
                    "page": metadata.get("page_number"),
                    "chunk_id": item.get("chunk_id"),
                }
            )
            if len(evidence_snippets) == max_snippets:
                break

        raw_evidence = "\n\n".join(
            f"[{idx}] {snippet}" for idx, snippet in enumerate(evidence_snippets, start=1)
        )
        return evidence_snippets, citations, raw_evidence

    def iter_rephrase_events(self, question: str, evidence_snippets: list[str]) -> Iterator[dict[str, str]]:
        """Yield {"type":"token","text":...} and a final {"type":"source","source":...}."""
        streamed = False
        for token in self._stream_with_llm(question, evidence_snippets):
            streamed = True
            yield {"type": "token", "text": token}
        if streamed:
            yield {"type": "source", "source": "gemini"}
            return

        offline = self._rephrase_offline(question, evidence_snippets)
        # Soft stream offline text so the UI can animate tokens.
        words = offline.split(" ")
        for idx, word in enumerate(words):
            piece = word if idx == 0 else f" {word}"
            yield {"type": "token", "text": piece}
        yield {"type": "source", "source": "offline"}

    def _normalize_rephrased(
        self,
        rephrased: str,
        question: str,
        evidence_snippets: list[str],
        rephrase_source: str,
    ) -> str:
        cleaned = " ".join(rephrased.split()).strip()
        if cleaned.casefold() not in {"not found in documents.", "not found in documents"}:
            return cleaned or "I could not find an answer to that question in the retrieved documents."

        offline = self._rephrase_offline(question, evidence_snippets)
        if offline.strip().casefold() not in {"not found in documents.", "not found in documents"}:
            detail = offline
            if detail.startswith("Regarding "):
                detail = detail.split(":", 1)[-1].strip()
            return (
                "I could not find a direct answer to that question in the retrieved documents. "
                f"Closest related detail from the chunks: {detail}"
            )
        return "I could not find an answer to that question in the retrieved documents."

    def _blend_answer(self, rephrased: str, evidence_snippets: list[str]) -> str:
        context_block = "\n".join(f"- {snippet}" for snippet in evidence_snippets[:3])
        return (
            "ANSWER:\n"
            f"{rephrased}\n\n"
            "SUPPORTING CHUNKS:\n"
            f"{context_block}"
        )

    def _api_key(self) -> str | None:
        key = (os.getenv("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
        if not key or key.startswith("your_"):
            return None
        return key

    def _model_candidates(self) -> list[str]:
        preferred = (os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview").strip()
        fallbacks = [
            preferred,
            "gemini-3-flash-preview",
            "gemini-3.8-flash",
            "gemini-flash-latest",
        ]
        ordered: list[str] = []
        for model in fallbacks:
            if model and model not in ordered:
                ordered.append(model)
        return ordered

    def _build_prompt(self, question: str, evidence_snippets: list[str]) -> str:
        context = "\n\n".join(f"[{idx}] {snippet}" for idx, snippet in enumerate(evidence_snippets, start=1))
        return (
            "You are answering a question about procurement / RFP documents.\n"
            "Use ONLY the evidence below. Do not invent facts that are not supported.\n"
            "Always write a natural-language answer in 2-4 sentences that directly addresses the question.\n"
            "If the exact fact is present, state it clearly (include dates, times, amounts, names).\n"
            "If the exact fact is missing, say that it was not found, then briefly summarize the most relevant related facts from the evidence.\n"
            "Do not return only chunk text. Do not refuse with a single terse phrase unless there is truly no useful evidence.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{context}\n\n"
            "Rephrased answer:"
        )

    def _stream_with_llm(self, question: str, evidence_snippets: list[str]) -> Iterator[str]:
        api_key = self._api_key()
        if not api_key:
            logger.warning("GEMINI_API_KEY missing; using offline rephrase.")
            return
        try:
            from google import genai
        except ImportError:
            logger.warning("google-genai not installed; using offline rephrase.")
            return

        prompt = self._build_prompt(question, evidence_snippets)
        client = genai.Client(api_key=api_key)
        for model_name in self._model_candidates():
            try:
                stream = client.models.generate_content_stream(model=model_name, contents=prompt)
                produced = False
                for chunk in stream:
                    text = getattr(chunk, "text", None) or ""
                    if not text:
                        continue
                    produced = True
                    yield text
                if produced:
                    return
            except Exception as exc:  # noqa: BLE001 - try next model
                logger.warning("Gemini stream model %s failed (%s).", model_name, type(exc).__name__)
                continue
        logger.warning("Gemini streaming failed; using offline rephrase.")

    def _rephrase_offline(self, question: str, evidence_snippets: list[str]) -> str:
        question_terms = {term.casefold() for term in re.findall(r"[A-Za-z0-9]+", question) if len(term) > 2}
        stems = {term[: max(4, len(term) - 3)] if len(term) > 5 else term for term in question_terms}
        sentences: list[str] = []
        for snippet in evidence_snippets:
            parts = re.split(r"(?<=[.!?])\s+", snippet)
            sentences.extend(part.strip() for part in parts if part.strip())

        if not sentences:
            return "Not found in documents."

        def score(sentence: str) -> int:
            tokens = set(re.findall(r"[A-Za-z0-9]+", sentence.casefold()))
            exact = sum(1 for term in question_terms if term in tokens)
            fuzzy = sum(1 for stem in stems if any(token.startswith(stem) for token in tokens))
            date_bonus = 2 if re.search(
                r"\b(?:\d{1,2}[:/]\d{2}|\d{4}|january|february|march|april|may|june|july|august|september|october|november|december)\b",
                sentence.casefold(),
            ) else 0
            return exact * 3 + fuzzy + date_bonus

        ranked = sorted(sentences, key=score, reverse=True)
        top = [sentence for sentence in ranked if score(sentence) > 0][:2]
        if not top:
            top = ranked[:1]

        focus = question.rstrip("?").strip()
        joined = " ".join(top)
        return f"Regarding \"{focus}\": {joined}"
