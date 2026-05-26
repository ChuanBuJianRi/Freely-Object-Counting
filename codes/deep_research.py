"""Run OpenAI Deep Research queries from the command line.

Usage:
    python codes/deep_research.py "你的研究问题"
    echo "你的研究问题" | python codes/deep_research.py
    python codes/deep_research.py --file path/to/prompt.md

Reads configuration from the project-root `.env` file (see `.env.example`).
Saves the final report and intermediate steps under `tmp/deep-research/<timestamp>/`.

Reference: https://platform.openai.com/docs/guides/deep-research
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    print("[!] 缺少依赖 python-dotenv，请先运行: pip install python-dotenv openai", file=sys.stderr)
    raise

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    print("[!] 缺少依赖 openai (>=1.40)，请先运行: pip install --upgrade openai", file=sys.stderr)
    raise


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tmp" / "deep-research"


SYSTEM_PROMPT = (
    "You are a senior CS research assistant for the GOC (Freely Object Counting) project. "
    "Produce a rigorous, citation-rich answer. Prefer primary sources (papers on arXiv, "
    "official repos, benchmark pages). Include a short TL;DR, then a structured deep dive, "
    "then a list of references with URLs."
)


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        print(f"[!] 未找到 {env_path}，请先复制 .env.example 为 .env 并填写。", file=sys.stderr)
        sys.exit(2)
    load_dotenv(env_path)
    if not os.getenv("OPENAI_API_KEY"):
        print("[!] .env 中 OPENAI_API_KEY 为空，请先配置。", file=sys.stderr)
        sys.exit(2)


def build_client() -> OpenAI:
    kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.getenv("OPENAI_ORG_ID"):
        kwargs["organization"] = os.environ["OPENAI_ORG_ID"]
    if os.getenv("OPENAI_PROJECT_ID"):
        kwargs["project"] = os.environ["OPENAI_PROJECT_ID"]
    return OpenAI(**kwargs)


def build_tools() -> list[dict]:
    tools: list[dict] = []
    if os.getenv("DEEP_RESEARCH_WEB_SEARCH", "true").lower() == "true":
        tools.append({"type": "web_search_preview"})
    if os.getenv("DEEP_RESEARCH_CODE_INTERPRETER", "true").lower() == "true":
        tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
    return tools


def run(query: str) -> Path:
    client = build_client()
    model = os.getenv("OPENAI_DEEP_RESEARCH_MODEL", "o4-mini-deep-research")
    tools = build_tools()
    timeout = int(os.getenv("DEEP_RESEARCH_TIMEOUT", "1800"))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "query.md").write_text(query, encoding="utf-8")

    print(f"[i] 模型: {model}")
    print(f"[i] 工具: {[t['type'] for t in tools]}")
    print(f"[i] 输出目录: {out_dir.relative_to(ROOT)}")
    print(f"[i] 提交任务中（deep research 通常耗时 5-30 分钟，使用 background 模式轮询）...")

    resp = client.responses.create(
        model=model,
        background=True,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": query}]},
        ],
        reasoning={"summary": "auto"},
        tools=tools,
    )
    rid = resp.id
    print(f"[i] response id = {rid}")
    (out_dir / "response_id.txt").write_text(rid, encoding="utf-8")

    start = time.time()
    while True:
        resp = client.responses.retrieve(rid)
        status = resp.status
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>5}s] status = {status}")
        if status in {"completed", "failed", "cancelled", "incomplete"}:
            break
        if elapsed > timeout:
            print(f"[!] 超时 {timeout}s，取消任务...")
            client.responses.cancel(rid)
            break
        time.sleep(15)

    (out_dir / "response.json").write_text(
        json.dumps(resp.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if status == "completed":
        text_parts: list[str] = []
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                for c in item.content:
                    if getattr(c, "type", None) == "output_text":
                        text_parts.append(c.text)
        final = "\n\n".join(text_parts).strip() or "(空)"
        report_path = out_dir / "report.md"
        report_path.write_text(
            f"# Deep Research 报告\n\n**模型**: {model}  \n**时间**: {stamp}\n\n## 问题\n\n{query}\n\n## 回答\n\n{final}\n",
            encoding="utf-8",
        )
        print(f"\n[✓] 完成，报告: {report_path.relative_to(ROOT)}")
        return report_path

    print(f"\n[x] 未完成，状态: {status}。详见 response.json", file=sys.stderr)
    sys.exit(1)


def parse_args() -> str:
    ap = argparse.ArgumentParser(description="OpenAI Deep Research CLI")
    ap.add_argument("query", nargs="*", help="研究问题（也可用 --file 或 stdin）")
    ap.add_argument("--file", "-f", type=Path, help="从文件读取问题")
    args = ap.parse_args()
    if args.file:
        return args.file.read_text(encoding="utf-8").strip()
    if args.query:
        return " ".join(args.query).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    ap.error("请通过参数、--file 或 stdin 提供问题")


if __name__ == "__main__":
    load_env()
    q = parse_args()
    if not q:
        print("[!] 问题为空", file=sys.stderr)
        sys.exit(2)
    run(q)
