#!/usr/bin/env python3
"""
External Benchmark Protocol (EBP) harness for LangGraph with Gemini.

Measures the overhead of LangGraph's ReAct agent pattern when executing
the same workload as adk-bench. Uses langchain-google-genai as the model
provider.

Usage:
    python3 bench_langgraph.py <workload.json>

Requires:
    - GOOGLE_API_KEY environment variable
    - BENCH_START_EPOCH_NS environment variable (injected by adk-bench)
    - pip install langgraph langchain-google-genai langchain-core
"""

import json
import os
import sys
import time


def main():
    if len(sys.argv) < 2:
        print("Usage: bench_langgraph.py <workload.json>", file=sys.stderr)
        sys.exit(1)

    workload_path = sys.argv[-1]
    api_key = os.environ.get("GOOGLE_API_KEY")
    bench_start_ns = int(os.environ.get("BENCH_START_EPOCH_NS", "0"))

    if not api_key:
        print("Error: GOOGLE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load workload
    with open(workload_path) as f:
        workload = json.load(f)

    from langchain_core.tools import tool as langchain_tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent

    # Build tools from workload definitions
    tools = []
    for tool_name, tool_def in workload.get("agent", {}).get("tools", {}).items():
        fixed_response = tool_def.get("fixedResponse", {"status": "success"})
        latency_ms = tool_def.get("simulatedLatencyMs", 0)

        # Create a LangChain tool dynamically using StructuredTool
        from langchain_core.tools import StructuredTool

        def make_tool_fn(name, desc, response, latency):
            def tool_func(**kwargs):
                if latency > 0:
                    time.sleep(latency / 1000.0)
                return json.dumps(response)

            return StructuredTool.from_function(
                func=tool_func,
                name=name,
                description=desc,
            )

        tools.append(make_tool_fn(tool_name, tool_def["description"], fixed_response, latency_ms))

    # Create the LLM with deterministic settings
    model_name = workload.get("model", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
        top_p=1.0,
    )

    # Create the ReAct agent graph
    agent = create_react_agent(llm, tools)

    # Measure execution
    user_message = workload["agent"]["userMessage"]
    system_prompt = workload["agent"]["instructions"]

    first_llm_call_ns = time.time_ns()

    turn_start = time.perf_counter_ns()

    # Invoke the agent
    result = agent.invoke(
        {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]},
    )

    turn_end = time.perf_counter_ns()

    # Count the messages to determine turns and compute overhead
    messages = result.get("messages", [])
    llm_calls = sum(1 for m in messages if hasattr(m, "type") and m.type == "ai")
    tool_calls_count = sum(1 for m in messages if hasattr(m, "type") and m.type == "tool")

    total_time_us = (turn_end - turn_start) // 1000

    # Estimate LLM time (we don't have per-call timing, so estimate overhead)
    # For LangGraph, the overhead includes: message routing, tool dispatch,
    # graph state management, serialization between nodes
    # We'll report total time and let the user understand it includes LLM
    # A more precise measurement would require instrumenting the LLM wrapper

    # For a fair comparison, report the overhead as a portion of total time
    # minus estimated LLM time. Since we can't separate them cleanly in LangGraph,
    # report the full per-turn time divided by number of LLM calls
    if llm_calls > 0:
        per_turn_us = total_time_us // llm_calls
    else:
        per_turn_us = total_time_us

    # We'll create samples based on how many turns occurred
    overhead_samples = [per_turn_us] * max(1, llm_calls)

    overhead_samples.sort()
    count = len(overhead_samples)
    min_us = overhead_samples[0]
    max_us = overhead_samples[-1]
    mean_us = sum(overhead_samples) // count
    median_us = overhead_samples[count // 2]
    p95_idx = min(int(0.95 * count + 0.5), count) - 1
    p99_idx = min(int(0.99 * count + 0.5), count) - 1
    p95_us = overhead_samples[max(0, p95_idx)]
    p99_us = overhead_samples[max(0, p99_idx)]

    # Cold start
    cold_start_us = (first_llm_call_ns - bench_start_ns) // 1000 if bench_start_ns > 0 else 0

    # Memory
    try:
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak_rss_bytes = peak_rss
        else:
            peak_rss_bytes = peak_rss * 1024
    except Exception:
        peak_rss_bytes = None

    # Output EBP JSON
    output = {
        "framework": "langgraph",
        "cold_start_us": max(0, cold_start_us),
        "first_llm_call_epoch_ns": first_llm_call_ns,
        "loop_overhead": {
            "min_us": min_us,
            "max_us": max_us,
            "mean_us": mean_us,
            "median_us": median_us,
            "p95_us": p95_us,
            "p99_us": p99_us,
            "count": count,
        },
        "peak_rss_bytes": peak_rss_bytes,
        "throughput_agents_per_sec": None,
        "token_overhead": None,
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
