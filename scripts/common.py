"""Shared paths and small helpers. Importing this also puts the repo root on sys.path,
so every script can `import mock_llm` no matter which folder it is launched from."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def path(*parts):
    """Absolute path under the repository root."""
    return os.path.join(ROOT, *parts)


RAW = path("data", "raw")
CLEAN = path("data", "clean")
POLICIES = path("data", "policies")
OUTPUTS = path("outputs")
STATE = path("state")
DB = path("data", "instamart.db")


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def md_table(df, floatfmt="{:,.2f}"):
    """Render a DataFrame as a GitHub-flavoured Markdown table (tabulate is not an allowed library)."""
    def cell(v):
        if isinstance(v, float):
            return "" if v != v else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, rule] + body)


def section(title):
    print(f"\n=== {title} ===")
