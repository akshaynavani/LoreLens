"""Push the bundled prompt templates to LangFuse prompt management.

Usage: python scripts/push_prompts.py [label ...]   (default label: LORELENS_PROMPT_LABEL)

After pushing, edit prompts in the LangFuse UI, then compare versions with
`lorelens eval --langfuse-dataset lorelens-qa --run-name <prompt-version>`.
"""

from __future__ import annotations

import sys

from lorelens.observability import push_default_prompts

if __name__ == "__main__":
    labels = sys.argv[1:] or None
    for name in push_default_prompts(labels):
        print(f"pushed {name}")
