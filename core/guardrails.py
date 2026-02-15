"""
Aegis Guardrails — NeMo Guardrails integration for chat safety filtering.

Wraps the config/ directory (config.yml + rails.co) to provide input/output
validation gates for the chat interface. Guardrails check that user messages
stay on-topic (document redaction) and bot responses are safe/professional.

The guardrails do NOT own the LLM call — they act as validation filters.
The actual RAG retrieval + LLM generation happens separately via DocumentRAG.
"""

import os
import re
import yaml
from importlib import import_module
from pathlib import Path
from typing import Optional, Tuple

from nemoguardrails import RailsConfig, LLMRails


# Path to the config directory containing config.yml and rails.co
_CONFIG_DIR = Path(__file__).parent.parent / "config"

# Singleton instance
_guardrails_instance: Optional["AegisGuardrails"] = None


class AegisGuardrails:
    """
    NeMo Guardrails wrapper for Aegis chat safety.

    Uses the existing config/config.yml and config/rails.co to:
    1. Check user input against policy (blocks off-topic, bypass attempts)
    2. Check bot output against policy (blocks hallucinations, unprofessional content)
    3. Handle simple dialog flows (greetings, process explanations) via Colang

    Usage:
        guardrails = get_guardrails()
        allowed, refusal = await guardrails.check_input("What SSNs are in this doc?")
        if not allowed:
            return refusal  # blocked by policy

        # ... do RAG + LLM call ...

        safe, sanitized = await guardrails.check_output(bot_response)
    """

    def __init__(self, config_path: Optional[str] = None):
        config_dir = config_path or str(_CONFIG_DIR)
        self.config = RailsConfig.from_path(config_dir)
        self.rails = LLMRails(self.config)

        # Load the self-check prompts from config.yml for lightweight validation
        config_yml = Path(config_dir) / "config.yml"
        with open(config_yml, "r") as f:
            raw_config = yaml.safe_load(f)

        self._input_check_prompt = None
        self._output_check_prompt = None
        for prompt in raw_config.get("prompts", []):
            if prompt.get("task") == "self_check_input":
                self._input_check_prompt = prompt["content"]
            elif prompt.get("task") == "self_check_output":
                self._output_check_prompt = prompt["content"]

    async def check_input(self, user_message: str) -> Tuple[bool, str]:
        """
        Check if a user message complies with the input policy.

        Uses the full NeMo Guardrails pipeline (Colang flows + self_check_input).
        If the message is off-topic or a bypass attempt, guardrails will return
        a refusal response from the defined Colang bot responses.

        Args:
            user_message: The raw user chat message.

        Returns:
            (allowed, message) tuple:
            - (True, "") if the message is on-topic and allowed
            - (False, refusal_text) if blocked by guardrails
        """
        try:
            result = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_message}]
            )

            response_text = result.get("content", "") if isinstance(result, dict) else str(result)

            # Detect if guardrails blocked the message by checking for refusal patterns
            # from rails.co bot response definitions
            refusal_patterns = [
                "focused specifically on document redaction",
                "designed to help with document redaction only",
                "I can only help with analyzing documents",
                "continue to focus on that task",
            ]

            is_blocked = any(pattern in response_text for pattern in refusal_patterns)

            if is_blocked:
                return False, response_text

            return True, ""

        except Exception as e:
            # If guardrails fail, allow the message through (fail-open for usability)
            print(f"  [guardrails] Input check failed: {e}")
            return True, ""

    async def check_output(self, bot_response: str) -> Tuple[bool, str]:
        """
        Check if a bot response complies with the output policy.

        Uses the self_check_output prompt to validate the response.
        If the response is blocked, returns a sanitized fallback.

        Args:
            bot_response: The generated bot response text.

        Returns:
            (safe, message) tuple:
            - (True, original_response) if the response passes
            - (False, sanitized_response) if blocked
        """
        if not self._output_check_prompt:
            return True, bot_response

        try:
            # Use the self_check_output prompt directly with the LLM
            check_prompt = self._output_check_prompt.replace(
                "{{ bot_response }}", bot_response
            )

            # Use the rails' LLM to check the output
            result = await self.rails.generate_async(
                messages=[
                    {"role": "system", "content": check_prompt},
                    {"role": "user", "content": "Should this response be blocked?"},
                ]
            )

            response_text = result.get("content", "") if isinstance(result, dict) else str(result)

            # "Yes" means the response should be blocked
            if response_text.strip().lower().startswith("yes"):
                return False, (
                    "I can only provide information about document redaction "
                    "and privacy. Let me know how I can help with your document."
                )

            return True, bot_response

        except Exception as e:
            # Fail-open: if output check fails, return the original response
            print(f"  [guardrails] Output check failed: {e}")
            return True, bot_response

    async def generate_simple(self, user_message: str) -> Optional[str]:
        """
        Generate a response using only Colang dialog flows (no RAG context).

        Useful for simple exchanges like greetings, process explanations,
        and off-topic refusals that don't need document context.

        Args:
            user_message: The user message.

        Returns:
            The guardrails-generated response, or None if no Colang flow matched.
        """
        try:
            result = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_message}]
            )
            response_text = result.get("content", "") if isinstance(result, dict) else str(result)
            return response_text if response_text.strip() else None
        except Exception:
            return None


def get_guardrails(config_path: Optional[str] = None) -> AegisGuardrails:
    """
    Get or create the singleton AegisGuardrails instance.

    The guardrails are expensive to initialize (loads Colang flows, compiles
    dialog trees), so we cache a single instance for the server lifetime.
    """
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = AegisGuardrails(config_path)
    return _guardrails_instance
