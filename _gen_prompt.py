"""One-shot generator: extracts the Saraiki system prompt from the notebook
into assistant/prompt.py without hand-retyping (which would corrupt the
Saraiki-specific characters ݙ ڄ ڳ ٻ ڃ ڦ ۏ)."""
import json
import os

nb = json.load(open("saraiki_speech_assistant.ipynb", encoding="utf-8"))
src = "".join(nb["cells"][3]["source"])

start = src.index("You are a native-level")
mid = src.index("USER MESSAGE:")
rest = src[mid:]
close_start = rest.index("Now understand")
close_end = rest.index('"""')

rules = src[start:mid].rstrip()
closing = rest[close_start:close_end].rstrip()

module = '''"""Saraiki system prompt, lifted verbatim from saraiki_speech_assistant.ipynb.

Do not retype the Saraiki example characters by hand - they are easy to mangle.
Regenerate from the notebook instead.
"""

SARAIKI_RULES = """\\
{rules}
"""

CLOSING_INSTRUCTION = """\\
{closing}
"""


def build_prompt(user_message: str) -> str:
    """Assemble the full Gemini prompt for one turn of conversation."""
    return (
        SARAIKI_RULES
        + "\\n\\nUSER MESSAGE:\\n"
        + user_message.strip()
        + "\\n\\n"
        + CLOSING_INSTRUCTION
    )
'''.format(rules=rules, closing=closing)

os.makedirs("assistant", exist_ok=True)
with open("assistant/prompt.py", "w", encoding="utf-8") as fh:
    fh.write(module)

print("wrote assistant/prompt.py")
print("rules chars:", len(rules), "closing chars:", len(closing))
