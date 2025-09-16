from __future__ import annotations

"""
helixenv — Verifiers environment for Helix concepts (MCQ and optional open-answer)

Patterned after bixbench/hle to align with verifiers usage.
"""

import json
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset
from verifiers import Parser
from verifiers.types import Messages


# Small built‑in MCQ bank (gold at index 0 before any shuffling)
MCQ_ITEMS: list[dict[str, Any]] = [
    {
        "id": "q-af-incidence",
        "question": "In Helix, what does the incidence matrix B_k represent at depth k?",
        "options": [
            "How child regions refine parent regions between depths k-1 and k",
            "A Gram matrix of feature vectors at depth k",
            "Fourier coefficients of the activation function",
            "A confusion matrix for predicted vs. true labels",
        ],
        "explanation": "B_k encodes parent→child refinement between consecutive depths.",
    },
    {
        "id": "q-cp-v",
        "question": "Helix constructs a unital CP map Φ(X)=V* X V. Which stable form does V use?",
        "options": [
            "V = D(τ_{k-1})^{-1/2} B_k D(τ_k)^{1/2}",
            "V = B_k^T",
            "V = D(τ_k) B_k D(τ_{k-1})",
            "V = (B_k B_k^T)^{1/2}",
        ],
        "explanation": "Stable scaling with τ prevents divide-by-zero and ill-conditioning.",
    },
    {
        "id": "q-ulam-gap",
        "question": "In Helix’s Ulam PF discretization, what is the reported spectral gap?",
        "options": [
            "1 − |λ₂(P)|, eigenvalues of P^T",
            "|λ₁(P)| − |λ₂(P)|, eigenvalues of P",
            "Sum of top‑2 singular values of P",
            "Trace(P) divided by grid size",
        ],
        "explanation": "Gap uses magnitudes of eigenvalues of P^T, taking 1 − |λ₂|.",
    },
    {
        "id": "q-mass-consistency",
        "question": "At depth k, mass consistency checks which quantity?",
        "options": [
            "‖τ_{k-1} − B_k τ_k‖₁",
            "‖V V* − I‖_F",
            "‖Φ(I) − I‖_F",
            "‖B_k − I‖₁",
        ],
        "explanation": "Mass at k−1 should equal aggregated mass from k via B_k.",
    },
    {
        "id": "q-parents",
        "question": "Why prefer parent pointers over dense B_k?",
        "options": [
            "O(n) memory and fast indexed aggregation",
            "Avoids using numpy entirely",
            "Improves SGD convergence",
            "Enables GPU backprop optimization",
        ],
        "explanation": "Parent pointers compress sparsity; B_k is built on demand.",
    },
    {
        "id": "q-cp-checks",
        "question": "Which diagnostics sanity‑check the CP construction?",
        "options": [
            "Unitality Φ(I)=I, PSD preservation, and coisometry proxy",
            "Only spectral norm of V",
            "Only det(V)",
            "Frobenius norm of B_k",
        ],
        "explanation": "Φ(I)=I, PSD for random PSDs, and ‖V V* − I‖_F as proxy.",
    },
    {
        "id": "q-anisotropy",
        "question": "Helix’s cumulative anisotropy proxy comes from?",
        "options": [
            "Column sums of cumulative product B₁ … B_k",
            "Row sums of V",
            "det(B_k)",
            "rank(B_k)",
        ],
        "explanation": "Multiply incidences across depths, then sum child columns.",
    },
]


# Parsers (mirroring bixbench helpers)
def extract_mcq_answer(text: str) -> str | None:
    """Extract MCQ answer letter from response. Falls back to option text.

    Returns:
        - 'A'/'B'/'C'/'D'/'E' when clearly indicated
        - Otherwise returns the normalized free text (to be handled by the scorer)
    """
    if not text:
        return None

    raw = text.strip()
    up = raw.upper()
    # Normalize common unicode punctuation to improve regex matching
    up = up.replace("’", "'")

    # Common refusal markers -> E
    if re.search(r"\b(I\s+DON['’]?T\s+KNOW|DON['’]?T\s+KNOW|IDK|UNSURE|NOT\s+SURE)\b", up):
        return "E"

    # Start-of-line single letter followed by common punctuation
    m = re.search(r"^\s*([A-E])(?=[\).:\s])", up)
    if m:
        return m.group(1)

    # Pattern like "Option A"
    m = re.search(r"\bOPTION\s*([A-E])\b", up)
    if m:
        return m.group(1)

    # Look for single letter at very start (fallback)
    if len(up) > 0 and up[0] in ["A", "B", "C", "D", "E"]:
        return up[0]

    # Pattern like "The answer is A" or "Answer: C"
    m = re.search(r"\b(?:ANSWER\s+IS|ANSWER[:\s])\s*([ABCDE])\b", up)
    if m:
        return m.group(1)

    # Intentionally avoid overly-loose fallbacks that may misparse
    # incidental letters (e.g., "As an AI...") as answers.

    # Fall back to raw text (used by scorer for fuzzy/normalized matching)
    return raw


def extract_open_answer(text: str) -> str | None:
    """Extract open-ended answer from response."""
    if not text:
        return None
    return text.strip()


def load_environment(
    mode: str = "zero_shot",
    answer_mode: str = "mcq",
    system_prompt: str | None = None,
    use_think: bool = False,
    shuffle_options: bool = True,
    with_refusal: bool = False,
    seed: int = 42,
    max_episodes: int | None = None,
    max_turns: int = 10,
    **env_kwargs,
) -> vf.Environment:
    """Load Helix MCQ/open environment following bixbench structure."""

    # Build a small in‑memory dataset and wrap as HF Dataset for consistency
    items = MCQ_ITEMS.copy()
    if max_episodes is not None:
        try:
            items = items[: max(0, min(max_episodes, len(items)))]
        except Exception:
            items = items[:max_episodes]

    def stable_shuffle(opts: list[str], qid: str, gold_idx: int = 0) -> tuple[list[str], int]:
        if not opts:
            return opts, 0
        try:
            import hashlib
            h = int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16)
        except Exception:
            h = 0
        rng = random.Random(seed + h)
        idxs = list(range(len(opts)))
        rng.shuffle(idxs)
        new_gold = idxs.index(gold_idx) if 0 <= gold_idx < len(idxs) else 0
        return [opts[i] for i in idxs], new_gold

    def transform_example(example: dict[str, Any]) -> dict[str, Any]:
        qid = str(example.get("id", ""))
        question_text = str(example.get("question", "")).strip()
        options: list[str] = list(example.get("options", []) or [])
        gold_index = 0
        if with_refusal:
            options = options + ["I don't know"]
        if shuffle_options:
            options, gold_index = stable_shuffle(options, qid, gold_index)

        if answer_mode == "mcq":
            if options:
                question_text += "\n\nOptions:\n" + "".join(
                    f"{chr(65+i)}. {opt}\n" for i, opt in enumerate(options[:5])
                )
            gold_letter = chr(65 + gold_index) if gold_index < 5 else "A"
            answer_data = {
                "gold": options[gold_index] if options else "",
                "options": options,
                "gold_index": gold_index,
                "gold_letter": gold_letter,
                "question_id": qid,
                "explanation": example.get("explanation", ""),
            }
        else:
            # open‑answer uses the gold text directly (first option)
            answer_data = {
                "gold": (options[0] if options else ""),
                "question_id": qid,
                "explanation": example.get("explanation", ""),
            }

        return {
            "question": question_text,
            "answer": json.dumps(answer_data),
            "task": f"helixenv-{mode}",
            "info": {"id": qid},
        }

    base = Dataset.from_list(items)
    eval_dataset = base.map(transform_example).select_columns(["question", "answer", "task", "info"])

    # Parser selection
    if answer_mode == "mcq":
        extract_fn = extract_mcq_answer
        default_prompt = (
            "You are answering multiple‑choice questions about the Helix toolkit. "
            "Respond with a single letter A, B, C, or D. If an 'E. I don't know' "
            "option is shown, you may answer E. Output only the letter."
        )
    else:
        extract_fn = extract_open_answer
        default_prompt = (
            "Answer succinctly in one or two sentences. Do not include qualifiers like 'I think'."
        )
    system_prompt = system_prompt or default_prompt

    parser = vf.ThinkParser(extract_fn) if use_think else Parser(extract_fn)

    # Scoring helpers (match bixbench normalization/fuzzy fallback)
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()

    def _best_option_match(pred_text: str, options: list[str]) -> int:
        from difflib import SequenceMatcher

        pred_n = _normalize(pred_text)
        best_idx, best_score = -1, 0.0
        for i, opt in enumerate(options):
            score = SequenceMatcher(None, pred_n, _normalize(opt)).ratio()
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx if best_score >= 0.8 else -1

    def score_completion(completion: Messages, answer: Any, **kwargs) -> float:
        # Parse answer payload
        if isinstance(answer, str):
            try:
                data = json.loads(answer)
            except Exception:
                data = {}
        else:
            data = answer if isinstance(answer, dict) else {}

        prediction = parser.parse_answer(completion)
        if prediction is None:
            return 0.0

        if answer_mode == "mcq":
            gold_letter = (data.get("gold_letter") or "").upper()
            options = data.get("options", [])
            gold_text = data.get("gold", "")

            pred_str = str(prediction)
            pred_up = pred_str.strip().upper()

            # Letter match path
            if pred_up in {"A", "B", "C", "D", "E"}:
                return 1.0 if gold_letter and pred_up == gold_letter else 0.0

            # Textual fallback: exact normalized gold or fuzzy option mapping
            if gold_text and options:
                if _normalize(pred_str) == _normalize(gold_text):
                    return 1.0
                idx = _best_option_match(pred_str, options)
                if idx >= 0 and gold_letter in {"A", "B", "C", "D", "E"}:
                    return 1.0 if idx == (ord(gold_letter) - ord("A")) else 0.0

            return 0.0

        # open‑answer: simple normalized exact match
        gold_text = data.get("gold", "")
        return 1.0 if _normalize(str(prediction)) == _normalize(str(gold_text)) else 0.0

    rule_rubric = vf.Rubric(funcs=[score_completion], weights=[1.0], parser=parser)

    if mode == "agentic":
        class HelixAgenticEnv(vf.MultiTurnEnv):
            def __init__(self, *args, max_turns: int = 10, **kwargs):
                super().__init__(*args, **kwargs)
                self._max_turns = max_turns

            def is_completed(self, messages: Messages, state: dict, **kwargs) -> bool:
                assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
                return len(assistant_msgs) >= self._max_turns

            def env_response(self, messages: Messages, state: dict, **kwargs) -> tuple[list, dict]:
                return [], state

        env = HelixAgenticEnv(
            dataset=eval_dataset,
            max_turns=max_turns,
            message_type="chat",
            rubric=rule_rubric,
            parser=parser,
            system_prompt=system_prompt,
            **env_kwargs,
        )
    else:
        env = vf.SingleTurnEnv(
            dataset=eval_dataset,
            parser=parser,
            rubric=rule_rubric,
            system_prompt=system_prompt,
            **env_kwargs,
        )

    return env
