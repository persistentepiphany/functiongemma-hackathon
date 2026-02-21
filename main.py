
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

# ── Persistent model singleton: init once, reuse across calls ──
_model = cactus_init(functiongemma_path)

# ── Training-matched system prompt ──────────────────────────────
# This is the EXACT trigger phrase from FunctionGemma's training data.
# The chat template places it in <start_of_turn>developer, immediately
# before <start_function_declaration> blocks.  Any extra instructions
# here are noise the 270M model wasn't trained on — strip them.
SYSTEM_PROMPT = (
    "You are a model that can do function calling with the following functions"
)

# ── Few-shot bank ───────────────────────────────────────────────
# One known-good single-tool call per common tool name.
# Used to prime the model with the exact output format it was
# trained to produce:
#   <start_function_call>call:NAME{key:<escape>val<escape>}<end_function_call>
# Cactus's chat template converts the tool_calls dict below into
# that format automatically.
_FEWSHOT = {
    "get_weather": (
        "Weather in Tokyo?",
        {"location": "Tokyo"},
    ),
    "set_alarm": (
        "Alarm for 7 AM.",
        {"hour": 7, "minute": 0},
    ),
    "send_message": (
        "Text Bob saying hi.",
        {"recipient": "Bob", "message": "hi"},
    ),
    "create_reminder": (
        "Remind me about lunch at noon.",
        {"title": "lunch", "time": "12:00 PM"},
    ),
    "search_contacts": (
        "Find Alice in contacts.",
        {"query": "Alice"},
    ),
    "play_music": (
        "Play jazz.",
        {"song": "jazz"},
    ),
    "set_timer": (
        "3 minute timer.",
        {"minutes": 3},
    ),
}


def _pick_fewshot(tools, user_msg):
    """Pick one few-shot example using a tool that ISN'T the likely target.

    Avoids the model copying example arguments instead of extracting
    from the real query.  Falls back to the first available match.
    """
    tool_names = [t["name"] for t in tools]
    available = [n for n in tool_names if n in _FEWSHOT]
    if not available:
        return []

    # Heuristic: skip tools whose name appears (partially) in the query
    lower_msg = user_msg.lower()
    safe = [n for n in available if n.split("_")[-1] not in lower_msg]
    pick = safe[0] if safe else available[-1]

    query, args = _FEWSHOT[pick]
    return [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {"name": pick, "arguments": args},
            }],
        },
    ]


def _optimize_tools(tools):
    """Shorten descriptions to <10 words; add enum/range hints to params."""
    out = []
    for t in tools:
        t = json.loads(json.dumps(t))  # deep copy
        # Truncate description
        words = t.get("description", "").split()
        if len(words) > 9:
            t["description"] = " ".join(words[:9])
        # Enrich parameter descriptions with type hints
        props = t.get("parameters", {}).get("properties", {})
        for pname, pdef in props.items():
            pdesc = pdef.get("description", "")
            ptype = pdef.get("type", "")
            low = pname.lower()
            if ptype == "integer" and "hour" in low:
                pdef["description"] = pdesc + " (0-23)" if pdesc else "Hour (0-23)"
            elif ptype == "integer" and "minute" in low:
                pdef["description"] = pdesc + " (0-59)" if pdesc else "Minute (0-59)"
            elif ptype == "integer" and "minutes" in low:
                pdef["description"] = pdesc + " (positive int)" if pdesc else "Minutes (positive int)"
            # Add enum hint if enum values exist
            if "enum" in pdef:
                vals = ", ".join(str(v) for v in pdef["enum"][:5])
                pdef["description"] = f"{pdesc} [{vals}]" if pdesc else f"[{vals}]"
        out.append(t)
    return out


def _build_messages(messages, tools):
    """Assemble the full message list for cactus_complete.

    Layout matches FunctionGemma's training template:
      <start_of_turn>developer
      {SYSTEM_PROMPT}<function_declarations><end_of_turn>
      <start_of_turn>user\n{few-shot query}<end_of_turn>
      <start_of_turn>model\n<function_call>...<end_of_turn>
      <start_of_turn>user\n{actual query}<end_of_turn>
      <start_of_turn>model\n   ← generation starts here
    """
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    prompt = [{"role": "system", "content": SYSTEM_PROMPT}]
    prompt += _pick_fewshot(tools, user_msg)
    prompt += messages
    return prompt


# ─── Validation ─────────────────────────────────────────────────

def _validate_local_result(function_calls, tools):
    """Accept local result if every call references a valid tool and has arguments."""
    if not function_calls:
        return False
    valid_names = {t["name"] for t in tools}
    for call in function_calls:
        if not isinstance(call, dict):
            return False
        if call.get("name") not in valid_names:
            return False
        if not isinstance(call.get("arguments"), dict):
            return False
    return True


# ─── Inference backends ─────────────────────────────────────────

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    global _model
    cactus_reset(_model)

    opt_tools = _optimize_tools(tools)
    cactus_tools = [{"type": "function", "function": t} for t in opt_tools]
    prompt = _build_messages(messages, opt_tools)

    raw_str = cactus_complete(
        _model,
        prompt,
        tools=cactus_tools,
        force_tools=True,
        temperature=0.0,
        top_k=1,
        max_tokens=512,
        confidence_threshold=0.01,
        tool_rag_top_k=0,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        print(f"    [DEBUG] JSON decode failed, raw_str={raw_str[:500]}")
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    print(
        f"    [DEBUG] confidence={raw.get('confidence', '?'):.3f}"
        f" cloud_handoff={raw.get('cloud_handoff', '?')}"
        f" calls={raw.get('function_calls', [])}"
        f" response={str(raw.get('response', ''))[:200]}"
    )

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ─── Hybrid routing (submission interface) ──────────────────────

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Local-first with cloud fallback.

    Push 1: prompt format fix + few-shot + tool optimization.
    No difficulty router yet — every query tries local first.
    """
    local = generate_cactus(messages, tools)

    if _validate_local_result(local["function_calls"], tools):
        local["source"] = "on-device"
        return local

    # Fallback to cloud only when local output is empty or malformed
    try:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local.get("confidence", 0)
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud
    except Exception:
        local["source"] = "on-device"
        return local


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
