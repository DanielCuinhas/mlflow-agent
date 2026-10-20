"""
Experiment-analysis agent: LangGraph + the read-only MLflow MCP server.

Setup:
    pip install langgraph langchain-anthropic langchain-openai langchain-mcp-adapters mcp \
        pydantic-settings python-dotenv "mlflow>=3"
Run:
    python agent.py                       # interactive chat
    python agent.py "Summarize the runs"  # one-shot
Select provider/model with --provider and --model, or use MODEL_PROVIDER and provider-specific settings.
Env (optional): MLFLOW_TRACKING_URI
Optional tracing: export LANGSMITH_TRACING=true LANGSMITH_API_KEY=...
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()  # load .env if present

from config import settings


HERE = Path(__file__).parent
RECURSION_LIMIT = 30  # max graph steps per turn (each tool round is ~2 steps)

SYSTEM_TEMPLATE = """You are an ML experiment analyst. You have read-only access to an MLflow \
tracking server through tools. Help the user understand their training runs and decide what to try next.

Rules:
- Ground every claim in tool results. Never invent metrics, params, or run names.
- Query efficiently: use search_runs for overviews and filters, compare_runs for differences, \
get_artifact_json only for the runs that matter. Avoid fetching every run's details.
- Treat metric differences smaller than the CV std (cv_*_std) as noise, and say so explicitly. \
Do not crown a winner when runs are statistically indistinguishable.
- Use gap_* metrics (train minus validation) to flag overfitting.
- Suggest next experiments as hypotheses: state the evidence, the proposed change, and what result \
would confirm or refute it. Stay inside the search space and constraints below.
- Never ask for or reference a final test set; it is intentionally held out.
- Be concise. Prefer a short table plus a few bullet points over long prose.

Problem context (from problem_context.md):
{context}
"""


def load_context() -> str:
    path = HERE / "problem_context.md"
    return path.read_text() if path.exists() else "(no problem_context.md found)"


def text_of(content) -> str:
    """ChatAnthropic content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def create_chat_model(provider: str | None = None, model: str | None = None):
    provider = provider or settings.model_provider

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install the Anthropic integration with `pip install langchain-anthropic`.") from exc

        api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to use the Anthropic provider.")
        return ChatAnthropic(
            model=model or os.environ.get("ANTHROPIC_MODEL", settings.model_name),
            api_key=api_key,
            max_tokens=4096,
            timeout=settings.http_timeout,
        )

    if provider in {"openai", "huggingface"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI-compatible integration with `pip install langchain-openai`.") from exc

        if provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI provider.")
            return ChatOpenAI(
                model=model or settings.openai_model_name,
                api_key=settings.openai_api_key,
                base_url=settings.model_base_url,
                timeout=settings.http_timeout,
            )

        if not settings.huggingface_api_key:
            raise RuntimeError("Set HUGGINGFACE_API_KEY to use the Hugging Face provider.")
        return ChatOpenAI(
            model=model or settings.huggingface_model_name,
            api_key=settings.huggingface_api_key,
            base_url=settings.model_base_url or "https://router.huggingface.co/v1",
            timeout=settings.http_timeout,
        )

    raise ValueError(f"Unsupported model provider: {provider}")


def build_graph(tools, system: str, provider: str | None = None, model: str | None = None):
    llm = create_chat_model(provider, model).bind_tools(tools)

    async def call_model(state: MessagesState):
        response = await llm.ainvoke([SystemMessage(content=system)] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "agent")
    # MemorySaver keeps conversation history per thread_id (in-process only).
    return builder.compile(checkpointer=MemorySaver())


async def ask(graph, question: str, config: dict) -> str:
    final = ""
    async for update in graph.astream(
        {"messages": [("user", question)]}, config, stream_mode="updates"
    ):
        for node, out in update.items():
            if node != "agent":
                continue
            for msg in out["messages"]:
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [tool] {tc['name']}({tc['args']})", file=sys.stderr)
                else:
                    final = text_of(msg.content)
    return final


async def main():
    parser = argparse.ArgumentParser(description="Chat with the MLflow experiment-analysis agent.")
    parser.add_argument(
        "--provider",
        choices=("anthropic", "openai", "huggingface"),
        help="Model provider (defaults to MODEL_PROVIDER or anthropic).",
    )
    parser.add_argument("--model", help="Model name (defaults to the selected provider's configured model).")
    parser.add_argument("question", nargs="*", help="Optional one-shot question; omit for interactive chat.")
    args = parser.parse_args()

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{Path('mlflow.db').resolve()}"
    )
    mcp_client = MultiServerMCPClient({
        "mlflow": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(HERE / "mcp_server.py")],
            "env": {**os.environ, "MLFLOW_TRACKING_URI": tracking_uri},
        }
    })
    config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": "main"}}

    # One persistent session, so the MCP server isn't respawned on every tool call.
    async with mcp_client.session("mlflow") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(
            tools,
            SYSTEM_TEMPLATE.format(context=load_context()),
            provider=args.provider,
            model=args.model,
        )

        if args.question:
            print(await ask(graph, " ".join(args.question), config))
            return

        print(f"Connected. Tools: {', '.join(t.name for t in tools)}")
        print("Ask about your runs ('exit' to quit).\n")
        while True:
            try:
                q = (await asyncio.to_thread(input, "you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            print("\n" + await ask(graph, q, config) + "\n")


if __name__ == "__main__":
    asyncio.run(main())