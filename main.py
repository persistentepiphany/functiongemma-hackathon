
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

# ── Persistent model singleton: init once, reuse across calls ──
_model = cactus_init(functiongemma_path)

SYSTEM_PROMPT = (
    "You are a function calling assistant. When the user asks you to do multiple things, "
    "you MUST call ALL relevant functions. Make one function call per request. "
    "Extract exact parameter values from the user message."
)


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


def _estimate_expected_calls(user_message):
    """Heuristic: count conjunctions/commas to guess how many calls are needed."""
    text = user_message.lower()
    count = 1
    # " and " between clauses signals an extra action
    count += text.count(" and ")
    # ", " can also separate actions (but discount trailing commas / commas in values)
    commas = text.count(", ")
    # Avoid double-counting ", and"
    commas -= text.count(", and")
    count += max(0, commas)
    return count


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    global _model
    cactus_reset(_model)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        _model,
        [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
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
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cactus_multipass(messages, tools):
    """Run on-device inference with optional second pass for multi-call cases."""
    result = generate_cactus(messages, tools)

    user_msg = messages[-1]["content"] if messages else ""
    expected = _estimate_expected_calls(user_msg)

    if len(result["function_calls"]) < expected:
        # Second pass to pick up missed calls
        result2 = generate_cactus(messages, tools)
        seen = {c["name"] for c in result["function_calls"]}
        for call in result2["function_calls"]:
            if call["name"] not in seen:
                result["function_calls"].append(call)
                seen.add(call["name"])
        result["total_time_ms"] += result2["total_time_ms"]

    return result


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


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Optimized hybrid: accept local result unless structurally invalid."""
    local = generate_cactus_multipass(messages, tools)

    if _validate_local_result(local["function_calls"], tools):
        local["source"] = "on-device"
        return local

    # Fallback to cloud only when local output is empty or malformed
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


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
