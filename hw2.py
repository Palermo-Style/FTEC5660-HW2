#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


VERIFIER_PROMPT = """\
You verify one CV against the candidate's LinkedIn and Facebook profiles and
report how reliable the CV is. The tools search and read those profiles.

The CV text is extracted from a PDF. Some CVs arrive as plain lines and some as
a markdown table whose words are split across many cells; read either form as
one CV.

Step 1 - find the right person.
Names are shared by many people, and a search that finds no exact match falls
back to unrelated people, so never trust a result just because the name matches.
Search LinkedIn with the CV's name, the INDUSTRY taken from the CV headline (a
headline like "Finance Professional" means industry "Finance"), and the CV's
CITY as the location. Pass the city on its own ("Munich"), never
"Munich, Germany" - the location filter matches a bare city or country and
returns nothing for anything else. If that returns nothing, retry without the
industry and then without the location.
Read the profiles of the remaining candidates and keep the one whose companies,
school and graduation year match the CV. A result with match_type "fuzzy", or
one whose jobs and education share nothing with the CV, is somebody else - drop
it. If no candidate matches the CV's employers and school, the CV cannot be
validated, so score it low.

Step 2 - check these fields one at a time and state the result of each.
  - name
  - city
  - every job in the CV: company, title together with seniority, start year,
    end year
  - every education entry: degree, school, field, graduation year
  - skills: write out the profile's skill list, then check EVERY skill the CV
    lists against it one by one. A skill the CV lists that the profile does not
    have is a discrepancy, even when all the other skills match.
Job descriptions, the headline and the hometown are never discrepancies.

Step 3 - decide.
Wording differences are not discrepancies: "Bachelor of Science" is "BSc",
"UI/UX Design" is "UI/UX", a CV title "Senior Engineer" matches a profile entry
"Engineer" whose seniority is "senior", and a CV listing fewer skills than the
profile is fine.
A discrepancy is at least one false claim: an inflated job title, a shifted
employment or graduation year, an upgraded degree, a school or employer the
profile does not have, a wrong city, or a skill the candidate does not have.
One discrepancy is enough.

Give a short justification, then one final line and nothing after it:

SCORE: <a number between 0 and 1>

Use 0.9-1.0 when the CV matches, and 0.0-0.1 when you found a discrepancy.
"""

_SCORE_RE = re.compile(r"score\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_ANY_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _read_score(message: Any) -> float | None:
    """Return the final score from the agent's last message, if it gave one."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "\n".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
        )
    text = str(content)
    match = _SCORE_RE.search(text)
    candidates = [match.group(1)] if match else _ANY_NUMBER_RE.findall(text)
    for raw in reversed(candidates):
        try:
            value = float(raw)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
    return None


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    ### YOUR CODE HERE
    from langchain.agents import create_agent
    from langchain_deepseek import ChatDeepSeek

    # No api_key argument: ChatDeepSeek reads DEEPSEEK_API_KEY from the
    # environment, which load_env_file() has already filled from .env.
    model = ChatDeepSeek(model=MODEL_NAME, temperature=0, timeout=300, max_retries=3)
    return create_agent(model, tools, system_prompt=VERIFIER_PROMPT)


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    ### YOUR CODE HERE
    limit = asyncio.Semaphore(3)

    async def one(name: str, text: str) -> tuple[str, float]:
        async with limit:
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": f"CV file: {name}\n\n{text}"}]}
                )
                score = _read_score(result["messages"][-1])
                if score is None:
                    print(f"warning: {name}: no score in the reply, using 0.5")
            except Exception as error:  # one bad CV must not lose the whole run
                print(f"warning: {name}: {type(error).__name__}: {error}")
                score = None
            final = 0.5 if score is None else score
            print(f"{name}: {final:.2f}")
            return name, final

    return dict(await asyncio.gather(*(one(name, text) for name, text in cvs.items())))


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
