import os
import json
import random
import time
from typing import Optional

from groq import Groq

SYSTEM_PROMPT = """You are a fraud threat-intelligence analyst working on a red-team fraud \
simulation system. You propose new, structurally distinct fraud scenarios for a synthetic \
UPI/Indian-digital-payments transaction simulator. You always respond with strict JSON only \
- no preamble, no markdown code fences, no commentary."""

PROMPT_TEMPLATE = """Given this F3 taxonomy excerpt:
{tactic_json}

And this list of already-generated scenarios (do not duplicate their mechanism):
{existing_scenario_ids}

{miss_context}

And this transaction schema your scenario must manipulate fields from:
{schema_json}

Propose ONE new fraud scenario that:
1. Maps to exactly one F3 tactic + technique not yet covered (or, if all base techniques are
   covered, a genuinely harder/novel variant of a previously-missed technique).
2. Specifies which data fields it manipulates (must be real fields from the schema above,
   including nested deviceDetails fields, or the join-only Identity/behavioral fields listed
   under _identity_join_fields / _behavioral_join_fields).
3. Specifies HOW it manipulates them - describe an aggregation-level anomaly (a pattern across
   multiple transactions or fields), not a single-field anomaly like "amount is very high."
4. Is structurally distinct from all existing scenarios listed above (different
   feature-manipulation signature - different field combination AND different manipulation_type,
   or the same combination applied in a genuinely different mechanism).

Output STRICT JSON matching exactly this shape, nothing else:
{{
  "scenario_name": "...",
  "f3_tactic": "...",
  "f3_technique": "...",
  "mechanism_description": "...",
  "fields_manipulated": ["...", "..."],
  "manipulation_type": "identity|behavioral|network|channel|ai_specific",
  "novelty_tag": "..."
}}"""


class ScenarioProposer:
    # Changed default model to a standard Groq powerhouse for JSON
    def __init__(self, f3_taxonomy_path: str, schema_path: str, model: str = "openai/gpt-oss-120b"):
        with open(f3_taxonomy_path) as f:
            self.f3_taxonomy = json.load(f)
        with open(schema_path) as f:
            self.schema = json.load(f)
        self.model = model
        
        # Automatically picks up the GROQ_API_KEY environment variable
        self.client = Groq()

    def _pick_tactic_excerpt(self) -> dict:
        return random.choice(self.f3_taxonomy["tactics"])

    def propose(
        self,
        existing_scenario_ids: list[str],
        miss_explanations: Optional[list[str]] = None,
    ) -> dict:
        tactic_excerpt = self._pick_tactic_excerpt()

        miss_context = ""
        if miss_explanations:
            joined = "\n".join(f"- {m}" for m in miss_explanations[-10:])
            miss_context = (
                "The Blue (Defend) model recently MISSED these scenario patterns for the "
                "following plain-language reasons - propose something that specifically "
                "targets one of these gaps if it fits a real F3 technique:\n" + joined
            )

        prompt = PROMPT_TEMPLATE.format(
            tactic_json=json.dumps(tactic_excerpt, indent=2),
            existing_scenario_ids=json.dumps(existing_scenario_ids),
            miss_context=miss_context,
            schema_json=json.dumps(self.schema, indent=2),
        )

        max_retries = 4
        for attempt in range(max_retries):
            try:
                # Updated to Groq's Chat Completions API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    # Forces the model to output a valid JSON object
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                
                raw_text = response.choices[0].message.content.strip()
                break  
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      [API overloaded/error, pausing for 5s... Attempt {attempt+1}/{max_retries}]")
                    time.sleep(5) # Groq recovers faster, 5s is usually enough
                else:
                    raise e

        # The JSON mode guarantees a JSON structure, but we still parse and validate keys
        try:
            scenario = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON: {raw_text[:300]}") from e

        required_keys = {
            "scenario_name", "f3_tactic", "f3_technique", "mechanism_description",
            "fields_manipulated", "manipulation_type", "novelty_tag",
        }
        missing = required_keys - scenario.keys()
        if missing:
            raise ValueError(f"LLM response missing required keys: {missing}")

        return scenario

if __name__ == "__main__":
    # Ensure you have run: export GROQ_API_KEY="your_api_key_here"
    proposer = ScenarioProposer(
        f3_taxonomy_path="config/f3_taxonomy.json",
        schema_path="config/upi_schema.json",
    )
    result = proposer.propose(existing_scenario_ids=[])
    print(json.dumps(result, indent=2))