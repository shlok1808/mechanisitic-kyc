"""S2 paraphrase pass: rewrite rendered vignettes for surface diversity.

The paraphraser MUST be a different model family from the subject models (Gemma-2,
Llama-3.1) so its style does not self-correlate with the interp experiments (design-doc
§5.3). Default backend = Gemini Flash; drop-in = GPT-4o-mini.

Both tiers are paraphrased by the SAME backend so the only systematic difference between
explicit and implicit vignettes is the banned-lexicon constraint (the RQ1 confound
control). For implicit vignettes the result is re-scanned for banned terms and retried;
if it still leaks after `max_retries`, we fall back to the clean rendered template
(tagged paraphrase_model="template-only").

Results are cached to disk keyed by content, so reruns are reproducible and never re-bill
the API despite the API being non-deterministic.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from templates import find_banned

_SYSTEM = (
    "You rewrite a first-person description a person gives of themselves and their "
    "finances. Keep every concrete fact (ages, numbers, timelines, family, job, money "
    "amounts, how they reacted to past market moves). Change only wording and sentence "
    "flow so it reads naturally and a little differently. Return ONLY the rewritten "
    "paragraph, first person, no preamble."
)
_IMPLICIT_RULE = (
    " Hard rule: do NOT use any of these words or variants of them: {banned}. Convey the "
    "same meaning through plain life narrative instead."
)


class ParaphraseClient:
    """Backend-agnostic paraphraser with a disk cache and banned-term retries."""

    def __init__(self, backend, model, banned_words, banned_regex, cache_dir,
                 temperature=0.7, max_retries=4, max_workers=8, enabled=True):
        self.backend = backend
        self.model = model
        self.banned_words = banned_words
        self.banned_regex = banned_regex
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # lazy

    # -- backends ----------------------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return
        if self.backend == "gemini":
            import google.generativeai as genai
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY not set")
            genai.configure(api_key=key)
            self._client = genai.GenerativeModel(self.model, system_instruction=_SYSTEM)
        elif self.backend == "openai":
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = OpenAI()
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

    def _call(self, prompt):
        self._ensure_client()
        for attempt in range(4):
            try:
                if self.backend == "gemini":
                    import google.generativeai as genai
                    r = self._client.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.temperature),
                    )
                    return r.text.strip()
                else:  # openai
                    r = self._client.chat.completions.create(
                        model=self.model, temperature=self.temperature,
                        messages=[{"role": "system", "content": _SYSTEM},
                                  {"role": "user", "content": prompt}],
                    )
                    return r.choices[0].message.content.strip()
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    # -- caching -----------------------------------------------------------------
    def _cache_path(self, mode, text):
        h = hashlib.sha1(
            f"{self.backend}|{self.model}|{mode}|{text}".encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    # -- public ------------------------------------------------------------------
    def paraphrase_one(self, text, mode):
        """Return (final_text, paraphrase_model). `mode` in {explicit, implicit}."""
        if not self.enabled:
            return text, "template-only"

        cp = self._cache_path(mode, text)
        if cp.exists():
            d = json.loads(cp.read_text())
            return d["text"], d["model"]

        prompt = text
        if mode == "implicit":
            prompt = text + _IMPLICIT_RULE.format(banned=", ".join(self.banned_words))

        out, model = None, self.model
        for _ in range(self.max_retries):
            cand = self._call(prompt)
            if mode == "implicit" and find_banned(cand, self.banned_regex):
                prompt = (text + _IMPLICIT_RULE.format(banned=", ".join(self.banned_words))
                          + " You used a forbidden word last time. Try again, avoiding ALL of them.")
                continue
            out = cand
            break
        if out is None:  # never produced a clean implicit rewrite -> use the clean template
            out, model = text, "template-only"

        cp.write_text(json.dumps({"text": out, "model": model}))
        return out, model

    def paraphrase_batch(self, jobs):
        """jobs: list of (text, mode). Returns list of (final_text, model), order-preserved."""
        if not self.enabled:
            return [(t, "template-only") for t, _ in jobs]
        results = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.paraphrase_one, t, m): i for i, (t, m) in enumerate(jobs)}
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results
