"""
Phase 2: The Autonomous Content Engine (LangGraph)
A state machine that: decides a topic → searches mock web → drafts a 280-char post.
Output is a strict JSON object: {"bot_id": "...", "topic": "...", "post_content": "..."}.
"""

import os
import json
import re
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# ─── Mock Search Tool ────────────────────────────────────────────────────────

MOCK_SEARCH_DB = {
    "crypto":     "Bitcoin hits new all-time high amid regulatory ETF approvals. Ethereum 2.0 staking yields surge.",
    "bitcoin":    "Bitcoin hits new all-time high amid regulatory ETF approvals.",
    "ai":         "OpenAI releases GPT-5; Anthropic Claude surpasses benchmarks. AI regulation debate heats up in EU.",
    "openai":     "OpenAI releases GPT-5 with real-time reasoning capabilities.",
    "tech":       "Apple surpasses $4T market cap. Google faces antitrust ruling in three countries simultaneously.",
    "elon":       "Elon Musk's xAI raises $6B Series B. Tesla FSD achieves Level 4 autonomy milestone.",
    "space":      "SpaceX Starship completes first crewed orbital mission. NASA Artemis moon landing confirmed for 2026.",
    "market":     "S&P 500 rallies 2.3% on strong jobs data. Fed signals two rate cuts in 2025.",
    "stock":      "Nvidia stock up 14% after blowout earnings. Magnificent 7 dominates Q1 returns.",
    "rate":       "Federal Reserve holds rates at 4.5%. Mortgage rates dip below 6% for first time since 2022.",
    "privacy":    "EU fines Meta €1.2B for GDPR violations. Apple launches encrypted cloud storage globally.",
    "climate":    "UN climate report: 2024 was hottest year on record. Renewables now cheaper than coal in 90% of markets.",
    "social":     "Instagram algorithm changes spark creator exodus to decentralized platforms.",
    "monopoly":   "DOJ files antitrust suit against Google Search. Meta faces forced divestiture of Instagram.",
    "regulation": "EU AI Act enforcement begins. OpenAI, Google face compliance audits.",
}

@tool
def mock_searxng_search(query: str) -> str:
    """
    Mock web search tool. Returns hardcoded recent news headlines based on keywords.
    Simulates a SearxNG search endpoint for the bot's research phase.
    """
    query_lower = query.lower()
    results = []
    for keyword, headline in MOCK_SEARCH_DB.items():
        if keyword in query_lower:
            results.append(headline)

    if results:
        return " | ".join(dict.fromkeys(results))   # deduplicate while preserving order
    return "No specific breaking news found. General tech discourse continues online."


# ─── Bot Personas ─────────────────────────────────────────────────────────────

BOT_PERSONAS = {
    "Bot_A": {
        "name": "Tech Maximalist",
        "system_prompt": (
            "You are Bot_A, an extreme tech optimist. You believe AI and crypto will solve "
            "ALL human problems. You are enthusiastic about Elon Musk, SpaceX, and the singularity. "
            "You dismiss regulatory concerns as fear-mongering. You post with energy and confidence. "
            "Your tweets are sharp, declarative, and slightly provocative."
        ),
    },
    "Bot_B": {
        "name": "Doomer / Skeptic",
        "system_prompt": (
            "You are Bot_B, a critical doomer. You believe late-stage capitalism and tech monopolies "
            "are destroying democracy and nature. You are suspicious of AI hype, billionaires, and "
            "surveillance tech. You cite real harms. You post with cold clarity and moral urgency."
        ),
    },
    "Bot_C": {
        "name": "Finance Bro",
        "system_prompt": (
            "You are Bot_C, a finance bro. You ONLY care about alpha, ROI, yield, and market signals. "
            "You speak in finance jargon: 'risk-adjusted returns', 'macro headwinds', 'sentiment play'. "
            "You view every world event as a trading opportunity. You post with bro-energy and conviction."
        ),
    },
}

# ─── LangGraph State ─────────────────────────────────────────────────────────

class BotState(TypedDict):
    bot_id: str
    messages: Annotated[list, add_messages]
    search_query: str
    search_results: str
    topic: str
    post_content: str


# ─── LLM Setup ───────────────────────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    return ChatGroq(model="llama3-8b-8192", temperature=0.8, api_key=api_key)


# ─── Node 1: Decide Search Query ─────────────────────────────────────────────

def node_decide_search(state: BotState) -> BotState:
    """LLM decides what topic to post about and formats a search query."""
    print(f"\n[Node 1 - Decide Search] Bot: {state['bot_id']}")
    llm = get_llm()
    persona = BOT_PERSONAS[state["bot_id"]]

    prompt = (
        f"{persona['system_prompt']}\n\n"
        "Decide ONE topic you want to post about today. "
        "Respond with ONLY a JSON object like: "
        '{"topic": "short topic label", "search_query": "search keywords for news"}'
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    print(f"[Node 1] LLM raw output: {raw}")

    # Safely extract JSON even if model wraps it in backticks
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    parsed = json.loads(match.group()) if match else {"topic": "technology", "search_query": "tech news"}

    print(f"[Node 1] Topic: {parsed['topic']} | Query: {parsed['search_query']}")
    return {**state, "topic": parsed["topic"], "search_query": parsed["search_query"]}


# ─── Node 2: Web Search ───────────────────────────────────────────────────────

def node_web_search(state: BotState) -> BotState:
    """Executes the mock_searxng_search tool."""
    print(f"\n[Node 2 - Web Search] Query: \"{state['search_query']}\"")
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"[Node 2] Results: {results}")
    return {**state, "search_results": results}


# ─── Node 3: Draft Post ───────────────────────────────────────────────────────

def node_draft_post(state: BotState) -> BotState:
    """LLM generates a 280-char opinionated post using persona + search context."""
    print(f"\n[Node 3 - Draft Post] Generating post for {state['bot_id']} …")
    llm = get_llm()
    persona = BOT_PERSONAS[state["bot_id"]]

    prompt = (
        f"{persona['system_prompt']}\n\n"
        f"Today's topic: {state['topic']}\n"
        f"Breaking news context: {state['search_results']}\n\n"
        "Write ONE highly opinionated tweet (max 280 characters) that reflects your personality. "
        "Respond with ONLY a JSON object: "
        '{"bot_id": "<bot_id>", "topic": "<topic>", "post_content": "<tweet text max 280 chars>"} '
        f"Use bot_id = \"{state['bot_id']}\"."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    print(f"[Node 3] LLM raw output: {raw}")

    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        parsed = json.loads(match.group())
    else:
        parsed = {
            "bot_id": state["bot_id"],
            "topic": state["topic"],
            "post_content": raw[:280],
        }

    # Enforce 280-char limit
    parsed["post_content"] = parsed["post_content"][:280]
    print(f"\n[Node 3] ✅ Final JSON output:\n{json.dumps(parsed, indent=2)}")
    return {**state, "post_content": parsed["post_content"]}


# ─── Build the Graph ──────────────────────────────────────────────────────────

def build_content_engine():
    graph = StateGraph(BotState)
    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search", node_web_search)
    graph.add_node("draft_post", node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_content_engine(bot_id: str) -> dict:
    """Run the LangGraph pipeline for a given bot and return the structured post."""
    if bot_id not in BOT_PERSONAS:
        raise ValueError(f"Unknown bot_id: {bot_id}. Choose from {list(BOT_PERSONAS.keys())}")

    app = build_content_engine()
    initial_state: BotState = {
        "bot_id": bot_id,
        "messages": [],
        "search_query": "",
        "search_results": "",
        "topic": "",
        "post_content": "",
    }
    final_state = app.invoke(initial_state)
    return {
        "bot_id": final_state["bot_id"],
        "topic": final_state["topic"],
        "post_content": final_state["post_content"],
    }


if __name__ == "__main__":
    for bot in ["Bot_A", "Bot_B", "Bot_C"]:
        print(f"\n{'='*60}")
        print(f" Running Content Engine for {bot}")
        print('='*60)
        result = run_content_engine(bot)
        print(f"\n📢 FINAL POST → {json.dumps(result, indent=2)}")
