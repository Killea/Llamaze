"""OpenAI Responses API adapter for llama-server.

Translates OpenAI Responses API (/v1/responses) requests to llama-server's
native /v1/chat/completions endpoint and converts chat completions responses
back to Responses format (streaming SSE events and non-streaming JSON).

Ported from CLIProxyAPI's Go implementation in
internal/translator/openai/openai/responses/ and
sdk/api/handlers/openai/openai_responses_*.

Run as a module: ``python -m responses_adapter`` for a standalone smoke test.
Import ``ResponsesProxyHandler`` and ``ResponsesWebsocketHandler`` to integrate
into an existing http.server based GUI.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional


# ---------------------------------------------------------------------------
# JSON helpers (mirror gjson/sjson semantics used by the Go port)
# ---------------------------------------------------------------------------

_MISSING = object()


def _gget(obj: Any, path: str, default: Any = None) -> Any:
    """gjson-style dotted path getter. Returns default when missing."""
    if obj is None:
        return default
    cur: Any = obj
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return default
        elif isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            return default
    return cur


def _gget_str(obj: Any, path: str) -> str:
    v = _gget(obj, path)
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def _gget_int(obj: Any, path: str) -> int:
    v = _gget(obj, path)
    if v is None or v == "":
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0


def _gget_float(obj: Any, path: str) -> float:
    v = _gget(obj, path)
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _gget_bool(obj: Any, path: str) -> bool:
    v = _gget(obj, path)
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return bool(v)


def _gexists(obj: Any, path: str) -> bool:
    """Check if a dotted path exists in the object (not None and present)."""
    sentinel = object()
    return _gget(obj, path, sentinel) is not sentinel


def _is_array(obj: Any, path: str) -> bool:
    v = _gget(obj, path)
    return isinstance(v, list)


def _is_object(obj: Any, path: str) -> bool:
    v = _gget(obj, path)
    return isinstance(v, dict)


def _sset(obj: dict, path: str, value: Any) -> None:
    """sjson-style dotted path setter that creates intermediate dicts/lists.

    Numeric path segments index into lists; the list is grown with None
    placeholders when needed. ``-1`` appends to a list.
    """
    parts = path.split(".")
    cur: Any = obj
    for i, part in enumerate(parts[:-1]):
        nxt_part = parts[i + 1]
        want_list = nxt_part.lstrip("-").isdigit()
        if isinstance(cur, list):
            idx = int(part)
            _list_grow(cur, idx)
            if cur[idx] is None:
                cur[idx] = [] if want_list else {}
            cur = cur[idx]
        else:
            if part not in cur or cur[part] is None:
                cur[part] = [] if want_list else {}
            cur = cur[part]
    last = parts[-1]
    if isinstance(cur, list):
        if last == "-1":
            cur.append(value)
        else:
            idx = int(last)
            _list_grow(cur, idx)
            cur[idx] = value
    else:
        cur[last] = value


def _list_grow(lst: list, idx: int) -> None:
    while len(lst) <= idx:
        lst.append(None)


def _sdelete(obj: dict, path: str) -> None:
    parts = path.split(".")
    cur: Any = obj
    for part in parts[:-1]:
        if isinstance(cur, list):
            idx = int(part)
            if idx >= len(cur) or cur[idx] is None:
                return
            cur = cur[idx]
        else:
            if part not in cur or cur[part] is None:
                return
            cur = cur[part]
    last = parts[-1]
    if isinstance(cur, list):
        idx = int(last)
        if idx < len(cur):
            cur[idx] = None
    else:
        cur.pop(last, None)


def _sset_raw(obj: dict, path: str, raw: Any) -> None:
    """Like _sset but treats dict/list values as already-serialized JSON
    structures (no re-encoding). Equivalent to sjson.SetRawBytes."""
    _sset(obj, path, raw)


def _join_raw_array(items: list) -> list:
    """Mirror of common.JoinRawArray: just returns the list."""
    return list(items)


def _sse_event(event: str, payload_obj: dict) -> bytes:
    body = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Tool conversion (openai_openai-responses_tools.go)
# ---------------------------------------------------------------------------

def _tool_name(tool: dict) -> str:
    name = (_gget_str(tool, "name") or _gget_str(tool, "function.name")).strip()
    return name


def _tool_description(tool: dict) -> str:
    return _gget_str(tool, "description") or _gget_str(tool, "function.description")


def _tool_parameters(tool: dict) -> Any:
    for p in (
        "parameters",
        "parametersJsonSchema",
        "input_schema",
        "function.parameters",
        "function.parametersJsonSchema",
    ):
        if _gexists(tool, p):
            return _gget(tool, p)
    return _MISSING


def _qualify_namespace_tool_name(namespace_name: str, child_name: str) -> str:
    child_name = (child_name or "").strip()
    if child_name == "" or namespace_name == "" or child_name.startswith("mcp__"):
        return child_name
    if child_name.startswith(namespace_name):
        return child_name
    if namespace_name.endswith("__"):
        return namespace_name + child_name
    return namespace_name + "__" + child_name


def _convert_function_tool(tool: dict, override_name: str = "") -> Optional[dict]:
    name = (override_name or "").strip() or _tool_name(tool)
    if name == "":
        return None
    chat_tool: dict = {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
    desc = _tool_description(tool)
    if desc:
        chat_tool["function"]["description"] = desc
    params = _tool_parameters(tool)
    if params is not _MISSING:
        chat_tool["function"]["parameters"] = params
    return chat_tool


def _convert_custom_tool(tool: dict, override_name: str = "") -> Optional[dict]:
    name = (override_name or "").strip() or _tool_name(tool)
    if name == "":
        return None
    chat_tool: dict = {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        },
    }
    desc = _tool_description(tool)
    if desc:
        chat_tool["function"]["description"] = desc
    return chat_tool


def _convert_namespace_tool(tool: dict) -> list:
    namespace_name = _gget_str(tool, "name").strip()
    children = _gget(tool, "tools")
    if not isinstance(children, list):
        return []
    out: list = []
    for child in children:
        if not isinstance(child, dict):
            continue
        child_name = _tool_name(child)
        qualified = _qualify_namespace_tool_name(namespace_name, child_name)
        ctype = _gget_str(child, "type").strip()
        if ctype in ("", "function"):
            t = _convert_function_tool(child, qualified)
            if t:
                out.append(t)
        elif ctype == "custom":
            t = _convert_custom_tool(child, qualified)
            if t:
                out.append(t)
    return out


def _convert_responses_tool_to_chat_tools(tool: dict) -> list:
    tool_type = _gget_str(tool, "type").strip()
    if tool_type in ("", "function"):
        t = _convert_function_tool(tool, "")
        return [t] if t else []
    if tool_type == "namespace":
        return _convert_namespace_tool(tool)
    if tool_type == "custom":
        t = _convert_custom_tool(tool, "")
        return [t] if t else []
    return []


def _responses_tool_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for part in output:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                t = _gget(part, "text")
                if t is not None:
                    parts.append(str(t))
        return "".join(parts)
    if output is not None:
        return json.dumps(output, separators=(",", ":"), ensure_ascii=False)
    return ""


def _unwrap_custom_tool_input(arguments: str) -> str:
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    if isinstance(parsed, dict) and "input" in parsed:
        v = parsed["input"]
        if isinstance(v, str):
            return v
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    return arguments


def _collect_custom_tool_names(request: dict) -> set[str]:
    names: set[str] = set()

    def collect(tools: Any, namespace_name: str) -> None:
        if not isinstance(tools, list):
            return
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            ttype = _gget_str(tool, "type").strip()
            if ttype == "custom":
                name = _tool_name(tool)
                if namespace_name:
                    name = _qualify_namespace_tool_name(namespace_name, name)
                if name:
                    names.add(name)
            elif ttype == "namespace":
                collect(_gget(tool, "tools"), _gget_str(tool, "name").strip())

    collect(_gget(request, "tools"), "")
    inp = _gget(request, "input")
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and _gget_str(item, "type") == "additional_tools":
                collect(_gget(item, "tools"), "")
    return names


def _single_custom_tool_name(request: dict) -> Optional[str]:
    names = _collect_custom_tool_names(request)
    if len(names) != 1:
        return None
    tool_count = 0

    def count(tools: Any) -> None:
        nonlocal tool_count
        if not isinstance(tools, list):
            return
        for tool in tools:
            if isinstance(tool, dict):
                tool_count += len(_convert_responses_tool_to_chat_tools(tool))

    count(_gget(request, "tools"))
    inp = _gget(request, "input")
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and _gget_str(item, "type") == "additional_tools":
                count(_gget(item, "tools"))
    if tool_count != 1:
        return None
    return next(iter(names))


def _split_qualified_function_call(request: dict, qualified_name: str) -> tuple[str, str]:
    """Returns (name, namespace). Mirrors splitResponsesQualifiedFunctionCallFromRequest."""
    qualified_name = (qualified_name or "").strip()
    if qualified_name == "":
        return "", ""
    best_namespace = ""
    best_child = ""

    def collect(tools: Any) -> None:
        nonlocal best_namespace, best_child
        if not isinstance(tools, list):
            return
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if _gget_str(tool, "type").strip() != "namespace":
                continue
            namespace_name = _gget_str(tool, "name").strip()
            if namespace_name == "":
                continue
            children = _gget(tool, "tools")
            if not isinstance(children, list):
                continue
            for child in children:
                if not isinstance(child, dict):
                    continue
                child_name = _tool_name(child)
                if child_name == "":
                    continue
                if _qualify_namespace_tool_name(namespace_name, child_name) == qualified_name:
                    best_namespace = namespace_name
                    best_child = child_name

    collect(_gget(request, "tools"))
    inp = _gget(request, "input")
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and _gget_str(item, "type") == "additional_tools":
                collect(_gget(item, "tools"))
    if best_namespace == "" or best_child == "":
        return qualified_name, ""
    return best_child, best_namespace


def _apply_namespace_fields(item: dict, request: dict, qualified_name: str, item_path: str = "") -> dict:
    name, namespace = _split_qualified_function_call(request, qualified_name)
    name_path = "name" if item_path == "" else f"{item_path}.name"
    ns_path = "namespace" if item_path == "" else f"{item_path}.namespace"
    _sset(item, name_path, name)
    if namespace:
        _sset(item, ns_path, namespace)
    else:
        _sdelete(item, ns_path)
    return item


def _pick_request(original: Optional[dict], current: Optional[dict]) -> Optional[dict]:
    if original is not None and _is_valid_json(original):
        return original
    if current is not None and _is_valid_json(current):
        return current
    return None


def _is_valid_json(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Request conversion: Responses -> Chat Completions
# (openai_openai-responses_request.go)
# ---------------------------------------------------------------------------


def _collect_reasoning_content(item: dict) -> str:
    summary = _gget(item, "summary")
    if isinstance(summary, list):
        parts: list[str] = []
        for s in summary:
            if isinstance(s, dict) and _gget_str(s, "type") == "summary_text":
                parts.append(_gget_str(s, "text"))
        text = "".join(parts)
        if text:
            return text
    return "[reasoning unavailable]"


def convert_responses_request_to_chat_completions(
    model_name: str, request: dict, stream: bool
) -> dict:
    """Convert an OpenAI Responses API request body to a Chat Completions
    request body. Mirrors ConvertOpenAIResponsesRequestToOpenAIChatCompletions."""
    out: dict = {"model": model_name, "messages": [], "stream": stream}

    # Request usage info in streaming mode so response.completed can include
    # usage data. Without this, llama-server omits the usage chunk and Codex
    # may treat the response as incomplete.
    if stream:
        out["stream_options"] = {"include_usage": True}

    max_tokens = _gget(request, "max_output_tokens")
    if max_tokens is not None:
        out["max_tokens"] = _gget_int(request, "max_output_tokens")

    messages: list = []

    def append_message(msg: dict) -> None:
        messages.append(msg)

    instructions = _gget(request, "instructions")
    if instructions is not None and instructions != "":
        append_message({"role": "system", "content": instructions})

    inp = _gget(request, "input")
    if isinstance(inp, list):
        input_items = inp
        # Collect output call IDs for adjacency tracking
        output_call_ids: set[str] = set()
        for item in input_items:
            if not isinstance(item, dict):
                continue
            itype = _gget_str(item, "type")
            if itype not in ("function_call_output", "custom_tool_call_output"):
                continue
            call_id = _gget_str(item, "call_id").strip()
            if call_id:
                output_call_ids.add(call_id)

        pending_tool_calls: list = []
        pending_tool_call_ids: list[str] = []
        pending_reasoning_content = ""
        awaiting_tool_outputs: set[str] = set()
        deferred_messages: list = []

        def take_pending_reasoning() -> str:
            nonlocal pending_reasoning_content
            v = pending_reasoning_content
            pending_reasoning_content = ""
            return v

        def flush_pending_tool_calls() -> None:
            nonlocal pending_tool_calls, pending_tool_call_ids
            if not pending_tool_calls:
                return
            assistant_msg: dict = {"role": "assistant", "tool_calls": pending_tool_calls}
            rc = take_pending_reasoning()
            if rc:
                assistant_msg["reasoning_content"] = rc
            append_message(assistant_msg)
            for cid in pending_tool_call_ids:
                cid = cid.strip()
                if cid:
                    awaiting_tool_outputs.add(cid)
            pending_tool_calls = []
            pending_tool_call_ids = []

        def flush_deferred() -> None:
            nonlocal deferred_messages
            for m in deferred_messages:
                append_message(m)
            deferred_messages = []

        def has_awaiting() -> bool:
            return any(cid in output_call_ids for cid in awaiting_tool_outputs)

        def append_regular(msg: dict) -> None:
            if has_awaiting():
                deferred_messages.append(msg)
                return
            append_message(msg)

        def append_pending_reasoning_message() -> None:
            rc = take_pending_reasoning()
            if rc == "":
                return
            append_regular({"role": "assistant", "content": "", "reasoning_content": rc})

        for item in input_items:
            if not isinstance(item, dict):
                continue
            itype = _gget_str(item, "type")
            if itype == "" and _gget_str(item, "role") != "":
                itype = "message"
            if itype not in ("function_call", "custom_tool_call"):
                flush_pending_tool_calls()

            if itype in ("message", ""):
                role = _gget_str(item, "role")
                if role == "developer":
                    role = "user"
                if role != "assistant":
                    append_pending_reasoning_message()
                msg: dict = {"role": role, "content": []}
                content = _gget(item, "content")
                if isinstance(content, list):
                    content_items: list = []
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        ctype = _gget_str(c, "type") or "input_text"
                        if ctype in ("input_text", "output_text"):
                            content_items.append({"type": "text", "text": _gget_str(c, "text")})
                        elif ctype == "input_image":
                            part: dict = {"type": "image_url", "image_url": {"url": _gget_str(c, "image_url")}}
                            if _gexists(c, "detail"):
                                part["image_url"]["detail"] = _gget_str(c, "detail")
                            content_items.append(part)
                    msg["content"] = content_items
                elif isinstance(content, str):
                    msg["content"] = content
                if role == "assistant":
                    rc = _gget_str(item, "reasoning_content")
                    if rc == "":
                        rc = take_pending_reasoning()
                    else:
                        pending_reasoning_content = ""
                    if rc:
                        msg["reasoning_content"] = rc
                append_regular(msg)

            elif itype == "reasoning":
                rc = _collect_reasoning_content(item)
                if pending_reasoning_content == "":
                    pending_reasoning_content = rc
                else:
                    pending_reasoning_content += rc

            elif itype == "function_call":
                tool_call: dict = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                if _gexists(item, "call_id"):
                    tool_call["id"] = _gget_str(item, "call_id")
                if _gexists(item, "name"):
                    fn_name = _gget_str(item, "name")
                    ns = _gget_str(item, "namespace").strip()
                    if ns:
                        fn_name = _qualify_namespace_tool_name(ns, fn_name)
                    tool_call["function"]["name"] = fn_name
                if _gexists(item, "arguments"):
                    tool_call["function"]["arguments"] = _gget_str(item, "arguments")
                pending_tool_calls.append(tool_call)
                cid = _gget_str(item, "call_id").strip()
                if cid:
                    pending_tool_call_ids.append(cid)

            elif itype == "function_call_output":
                tool_msg: dict = {"role": "tool", "tool_call_id": "", "content": ""}
                call_id = ""
                if _gexists(item, "call_id"):
                    call_id = _gget_str(item, "call_id").strip()
                    tool_msg["tool_call_id"] = call_id
                if _gexists(item, "output"):
                    tool_msg["content"] = _gget_str(item, "output")
                append_message(tool_msg)
                if call_id:
                    awaiting_tool_outputs.discard(call_id)
                if not awaiting_tool_outputs and deferred_messages:
                    flush_deferred()

            elif itype == "custom_tool_call":
                tool_call = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                tool_call["id"] = _gget_str(item, "call_id")
                tool_call["function"]["name"] = _gget_str(item, "name")
                wrapped = {"input": _gget_str(item, "input")}
                tool_call["function"]["arguments"] = json.dumps(wrapped, separators=(",", ":"), ensure_ascii=False)
                pending_tool_calls.append(tool_call)
                cid = _gget_str(item, "call_id").strip()
                if cid:
                    pending_tool_call_ids.append(cid)

            elif itype == "custom_tool_call_output":
                tool_msg = {"role": "tool", "tool_call_id": "", "content": ""}
                call_id = _gget_str(item, "call_id").strip()
                tool_msg["tool_call_id"] = call_id
                tool_msg["content"] = _responses_tool_output_text(_gget(item, "output"))
                append_message(tool_msg)
                if call_id:
                    awaiting_tool_outputs.discard(call_id)
                if not awaiting_tool_outputs and deferred_messages:
                    flush_deferred()

        flush_pending_tool_calls()
        append_pending_reasoning_message()
        flush_deferred()

    elif isinstance(inp, str):
        append_message({"role": "user", "content": inp})

    # Tools
    chat_tools: list = []

    def append_chat_tools(tools: Any) -> None:
        if not isinstance(tools, list):
            return
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            for ct in _convert_responses_tool_to_chat_tools(tool):
                chat_tools.append(ct)

    append_chat_tools(_gget(request, "tools"))
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and _gget_str(item, "type") == "additional_tools":
                append_chat_tools(_gget(item, "tools"))
    if chat_tools:
        out["tools"] = chat_tools
        if _gexists(request, "parallel_tool_calls"):
            out["parallel_tool_calls"] = _gget_bool(request, "parallel_tool_calls")
        if _gexists(request, "tool_choice"):
            out["tool_choice"] = _gget(request, "tool_choice")

    # When the conversation ends with a tool result, inject a system nudge
    # to encourage tool use.  Some models (e.g. Qwen 3.6 abliterated) tend
    # to describe actions in text instead of emitting tool calls after
    # receiving tool results.  The nudge reminds the model to use tools.
    # Note: we do NOT force tool_choice=required here because that causes
    # infinite loops (the model can never stop to give a final answer).
    if chat_tools and messages and messages[-1].get("role") == "tool":
        messages.append({
            "role": "system",
            "content": (
                "You have received tool results. If the task is not yet complete, "
                "call the appropriate tool to perform the next action. "
                "Do NOT describe actions in text - use tool calls to actually execute them. "
                "If the task is fully done, respond with a summary of what was accomplished."
            ),
        })

    if messages:
        out["messages"] = messages

    reasoning_effort = _gget_str(request, "reasoning.effort").strip().lower()
    if reasoning_effort:
        out["reasoning_effort"] = reasoning_effort

    return out


# ---------------------------------------------------------------------------
# Non-streaming response conversion: Chat Completions -> Responses JSON
# (openai_openai-responses_response.go::...NonStream)
# ---------------------------------------------------------------------------

_response_id_counter = 0
_response_id_counter_lock = threading.Lock()


def _next_response_id_counter() -> int:
    global _response_id_counter
    with _response_id_counter_lock:
        _response_id_counter += 1
        return _response_id_counter



def _echo_request_fields(resp: dict, request: Optional[dict]) -> None:
    """Echo request fields into the response object (both stream and non-stream)."""
    if not request:
        return
    field_map_int = ["max_output_tokens", "max_tool_calls", "top_logprobs"]
    field_map_str = [
        "instructions",
        "model",
        "previous_response_id",
        "prompt_cache_key",
        "safety_identifier",
        "service_tier",
        "truncation",
    ]
    field_map_bool = ["parallel_tool_calls", "store"]
    field_map_float = ["temperature", "top_p"]
    field_map_raw = [
        "reasoning",
        "text",
        "tool_choice",
        "tools",
        "user",
        "metadata",
    ]
    for f in field_map_str:
        if _gexists(request, f):
            _sset(resp, f, _gget(request, f))
    for f in field_map_int:
        if _gexists(request, f):
            _sset(resp, f, _gget_int(request, f))
    # max_tokens fallback: if max_output_tokens not in request, check max_tokens
    if not _gexists(request, "max_output_tokens") and _gexists(request, "max_tokens"):
        _sset(resp, "max_output_tokens", _gget_int(request, "max_tokens"))
    for f in field_map_bool:
        if _gexists(request, f):
            _sset(resp, f, _gget_bool(request, f))
    for f in field_map_float:
        if _gexists(request, f):
            _sset(resp, f, _gget_float(request, f))
    for f in field_map_raw:
        if _gexists(request, f):
            _sset(resp, f, _gget(request, f))


def convert_chat_response_to_responses_json(
    chat_resp: dict, request: Optional[dict] = None
) -> dict:
    """Convert a non-streaming Chat Completions response to a Responses JSON object.

    Mirrors ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream.
    """
    resp: dict = {
        "id": "",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
    }

    rid = _gget_str(chat_resp, "id")
    if rid == "":
        rid = f"resp_{int(time.time_ns()):x}_{_next_response_id_counter()}"
    resp["id"] = rid

    created = _gget_int(chat_resp, "created")
    if created == 0:
        created = int(time.time())
    resp["created_at"] = created

    # Echo request fields
    if request:
        _echo_request_fields(resp, request)
        if not _gexists(resp, "model"):
            if _gexists(request, "model"):
                _sset(resp, "model", _gget(request, "model"))
            elif _gexists(chat_resp, "model"):
                _sset(resp, "model", _gget(chat_resp, "model"))
    elif _gexists(chat_resp, "model"):
        _sset(resp, "model", _gget(chat_resp, "model"))

    outputs: list = []

    # Reasoning content
    rc_text = _gget_str(chat_resp, "choices.0.message.reasoning_content")
    include_reasoning = rc_text != ""
    if not include_reasoning and request:
        include_reasoning = _gexists(request, "reasoning")
    if include_reasoning:
        rid_stripped = rid
        if rid_stripped.startswith("resp_"):
            rid_stripped = rid_stripped[len("resp_"):]
        reasoning_item: dict = {"id": f"rs_{rid_stripped}", "type": "reasoning", "encrypted_content": "", "summary": []}
        if rc_text:
            reasoning_item["summary"] = [{"type": "summary_text", "text": rc_text}]
        outputs.append(reasoning_item)

    request_for_namespace = _pick_request(request, request)

    choices = _gget(chat_resp, "choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            msg = _gget(choice, "message")
            if not isinstance(msg, dict):
                continue
            # Text message
            content = _gget_str(msg, "content")
            if content:
                item: dict = {
                    "id": f"msg_{rid}_{_gget_int(choice, 'index')}",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": content}],
                    "role": "assistant",
                }
                outputs.append(item)
            # Tool calls
            tcs = _gget(msg, "tool_calls")
            if isinstance(tcs, list):
                custom_names = _collect_custom_tool_names(request) if request else set()
                for tc_idx, tc in enumerate(tcs):
                    if not isinstance(tc, dict):
                        continue
                    call_id = _gget_str(tc, "id")
                    if call_id == "":
                        call_id = f"call_{rid}_{_gget_int(choice, 'index')}_{tc_idx}"
                    name = _gget_str(tc, "function.name")
                    args = _gget_str(tc, "function.arguments")
                    if name in custom_names:
                        item = {
                            "id": f"ctc_{call_id}",
                            "type": "custom_tool_call",
                            "status": "completed",
                            "input": _unwrap_custom_tool_input(args),
                            "call_id": call_id,
                            "name": name,
                        }
                        outputs.append(item)
                        continue
                    item = {
                        "id": f"fc_{call_id}",
                        "type": "function_call",
                        "status": "completed",
                        "arguments": args,
                        "call_id": call_id,
                        "name": name,
                    }
                    if request_for_namespace:
                        item = _apply_namespace_fields(item, request_for_namespace, name, "")
                    outputs.append(item)

    if outputs:
        resp["output"] = outputs

    # Usage
    usage = _gget(chat_resp, "usage")
    if isinstance(usage, dict):
        if _gexists(usage, "prompt_tokens") or _gexists(usage, "completion_tokens") or _gexists(usage, "total_tokens"):
            resp["usage"] = {"input_tokens": _gget_int(usage, "prompt_tokens")}
            if _gexists(usage, "prompt_tokens_details.cached_tokens"):
                _sset(resp, "usage.input_tokens_details.cached_tokens", _gget_int(usage, "prompt_tokens_details.cached_tokens"))
            _sset(resp, "usage.output_tokens", _gget_int(usage, "completion_tokens"))
            if _gexists(usage, "output_tokens_details.reasoning_tokens"):
                _sset(resp, "usage.output_tokens_details.reasoning_tokens", _gget_int(usage, "output_tokens_details.reasoning_tokens"))
            _sset(resp, "usage.total_tokens", _gget_int(usage, "total_tokens"))
        else:
            resp["usage"] = usage

    return resp


# ---------------------------------------------------------------------------
# Streaming response conversion: Chat Completions SSE -> Responses SSE events
# (openai_openai-responses_response.go::Convert...Stream)
# ---------------------------------------------------------------------------


@dataclass
class _ReasoningRec:
    reasoning_id: str
    reasoning_data: str
    output_index: int


@dataclass
class OaiToResponsesState:
    """Mutable state for the streaming converter. One instance per stream."""
    seq: int = 0
    response_id: str = ""
    created: int = 0
    started: bool = False
    completion_pending: bool = False
    completed_emitted: bool = False
    reasoning_id: str = ""
    reasoning_index: int = 0
    msg_text_buf: dict[int, list[str]] = field(default_factory=dict)
    reasoning_buf: list[str] = field(default_factory=list)
    reasonings: list[_ReasoningRec] = field(default_factory=list)
    func_args_buf: dict[str, list[str]] = field(default_factory=dict)
    func_names: dict[str, str] = field(default_factory=dict)
    func_call_ids: dict[str, str] = field(default_factory=dict)
    func_output_ix: dict[str, int] = field(default_factory=dict)
    func_args_sent: dict[str, int] = field(default_factory=dict)
    msg_output_ix: dict[int, int] = field(default_factory=dict)
    next_output_ix: int = 0
    msg_item_added: dict[int, bool] = field(default_factory=dict)
    msg_content_added: dict[int, bool] = field(default_factory=dict)
    msg_item_done: dict[int, bool] = field(default_factory=dict)
    func_item_added: dict[str, bool] = field(default_factory=dict)
    func_item_custom: dict[str, bool] = field(default_factory=dict)
    func_args_done: dict[str, bool] = field(default_factory=dict)
    func_item_done: dict[str, bool] = field(default_factory=dict)
    custom_tool_names: set[str] = field(default_factory=set)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    usage_seen: bool = False
    finish_reason: str = ""
    has_tool_calls: bool = False

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def alloc_output_index(self) -> int:
        ix = self.next_output_ix
        self.next_output_ix += 1
        return ix

    def reset_aggregation(self) -> None:
        self.msg_text_buf = {}
        self.reasoning_buf = []
        self.reasoning_id = ""
        self.reasoning_index = 0
        self.func_args_buf = {}
        self.func_names = {}
        self.func_call_ids = {}
        self.func_output_ix = {}
        self.func_args_sent = {}
        self.msg_output_ix = {}
        self.next_output_ix = 0
        self.msg_item_added = {}
        self.msg_content_added = {}
        self.msg_item_done = {}
        self.func_item_added = {}
        self.func_item_custom = {}
        self.func_args_done = {}
        self.func_item_done = {}
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.reasoning_tokens = 0
        self.usage_seen = False
        self.completion_pending = False
        self.completed_emitted = False

    def _reasoning_buf_str(self) -> str:
        return "".join(self.reasoning_buf)

    def _msg_text(self, idx: int) -> str:
        return "".join(self.msg_text_buf.get(idx, []))

    def _func_args(self, key: str) -> str:
        return "".join(self.func_args_buf.get(key, []))


def _build_completed_event(st: OaiToResponsesState, request: Optional[dict]) -> bytes:
    completed: dict = {
        "type": "response.completed",
        "sequence_number": 0,
        "response": {
            "id": "",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "background": False,
            "error": None,
        },
    }
    completed["sequence_number"] = st.next_seq()
    completed["response"]["id"] = st.response_id
    completed["response"]["created_at"] = st.created

    if request:
        _echo_request_fields(completed["response"], request)

    output_items: list[tuple[int, dict]] = []
    for r in st.reasonings:
        item = {"id": r.reasoning_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": r.reasoning_data}]}
        output_items.append((r.output_index, item))
    for idx, added in st.msg_item_added.items():
        if not added:
            continue
        txt = st._msg_text(idx)
        item = {
            "id": f"msg_{st.response_id}_{idx}",
            "type": "message",
            "status": "completed",
            "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": txt}],
            "role": "assistant",
        }
        output_items.append((st.msg_output_ix.get(idx, 0), item))
    for key, buf in st.func_args_buf.items():
        args = st._func_args(key)
        call_id = st.func_call_ids.get(key, "")
        name = st.func_names.get(key, "")
        if st.func_item_custom.get(key):
            item = {
                "id": f"ctc_{call_id}",
                "type": "custom_tool_call",
                "status": "completed",
                "input": _unwrap_custom_tool_input(args),
                "call_id": call_id,
                "name": name,
            }
            output_items.append((st.func_output_ix.get(key, 0), item))
            continue
        item = {
            "id": f"fc_{call_id}",
            "type": "function_call",
            "status": "completed",
            "arguments": args,
            "call_id": call_id,
            "name": name,
        }
        if request:
            item = _apply_namespace_fields(item, request, name, "")
        output_items.append((st.func_output_ix.get(key, 0), item))

    output_items.sort(key=lambda x: x[0])
    if output_items:
        completed["response"]["output"] = [it for _, it in output_items]

    if st.usage_seen:
        usage: dict = {
            "input_tokens": st.prompt_tokens,
            "output_tokens": st.completion_tokens,
        }
        if st.cached_tokens:
            usage["input_tokens_details"] = {"cached_tokens": st.cached_tokens}
        if st.reasoning_tokens > 0:
            usage["output_tokens_details"] = {"reasoning_tokens": st.reasoning_tokens}
        total = st.total_tokens
        if total == 0:
            total = st.prompt_tokens + st.completion_tokens
        usage["total_tokens"] = total
        completed["response"]["usage"] = usage

    # end_turn: false when the model made tool calls (wants to continue),
    # true when the model finished with stop.  Codex checks this field to
    # decide whether to continue the turn.
    if st.has_tool_calls or st.finish_reason == "tool_calls":
        completed["response"]["end_turn"] = False
    else:
        completed["response"]["end_turn"] = True
        full_text = "".join("".join(v) for v in st.msg_text_buf.values())
        # Detect potential unparseable tool calls in the model text.
        tool_markers = ["<function=", "</function>", "<tool_call", "</tool_call>"]
        found_markers = [m for m in tool_markers if m in full_text]
        if found_markers:
            import logging
            logging.getLogger("responses_adapter").warning(
                "finish_reason=stop but text contains tool call markers: %s (text: %.200s)",
                found_markers, full_text
            )

    return _sse_event("response.completed", completed)


def _emit_message_done_events(st: OaiToResponsesState, idx: int) -> list[bytes]:
    """Emit output_text.done, content_part.done, output_item.done for a message."""
    out: list[bytes] = []
    if not st.msg_item_added.get(idx) or st.msg_item_done.get(idx):
        return out
    msg_output_index = st.msg_output_ix.get(idx, 0)
    full_text = st._msg_text(idx)
    item_id = f"msg_{st.response_id}_{idx}"

    done = {"type": "response.output_text.done", "sequence_number": st.next_seq(), "item_id": item_id, "output_index": msg_output_index, "content_index": 0, "text": full_text, "logprobs": []}
    out.append(_sse_event("response.output_text.done", done))

    part_done = {"type": "response.content_part.done", "sequence_number": st.next_seq(), "item_id": item_id, "output_index": msg_output_index, "content_index": 0, "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": full_text}}
    out.append(_sse_event("response.content_part.done", part_done))

    item_done = {"type": "response.output_item.done", "sequence_number": st.next_seq(), "output_index": msg_output_index, "item": {"id": item_id, "type": "message", "status": "completed", "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": full_text}], "role": "assistant"}}
    out.append(_sse_event("response.output_item.done", item_done))
    st.msg_item_done[idx] = True
    return out


def _stop_reasoning(st: OaiToResponsesState, text: str) -> list[bytes]:
    out: list[bytes] = []
    text_done = {"type": "response.reasoning_summary_text.done", "sequence_number": st.next_seq(), "item_id": st.reasoning_id, "output_index": st.reasoning_index, "summary_index": 0, "text": text}
    out.append(_sse_event("response.reasoning_summary_text.done", text_done))
    part_done = {"type": "response.reasoning_summary_part.done", "sequence_number": st.next_seq(), "item_id": st.reasoning_id, "output_index": st.reasoning_index, "summary_index": 0, "part": {"type": "summary_text", "text": text}}
    out.append(_sse_event("response.reasoning_summary_part.done", part_done))
    item_done = {"type": "response.output_item.done", "item": {"id": st.reasoning_id, "type": "reasoning", "encrypted_content": "", "summary": [{"type": "summary_text", "text": text}]}, "output_index": st.reasoning_index, "sequence_number": st.next_seq()}
    out.append(_sse_event("response.output_item.done", item_done))
    st.reasonings.append(_ReasoningRec(st.reasoning_id, text, st.reasoning_index))
    st.reasoning_id = ""
    return out


def _emit_tool_item(st: OaiToResponsesState, key: str, force: bool, request: Optional[dict]) -> list[bytes]:
    out: list[bytes] = []
    if st.func_item_added.get(key):
        return out
    call_id = st.func_call_ids.get(key, "")
    name = st.func_names.get(key, "")
    if not force and (call_id == "" or name == ""):
        return out
    if name == "":
        single = _single_custom_tool_name(request) if request else None
        if single:
            name = single
            st.func_names[key] = single
    if call_id == "":
        call_id = f"call_{st.response_id}_{key.replace(':', '_')}"
        st.func_call_ids[key] = call_id
    output_index = st.func_output_ix.get(key, 0)
    is_custom = name in st.custom_tool_names
    st.func_item_custom[key] = is_custom
    if is_custom:
        o = {"type": "response.output_item.added", "sequence_number": st.next_seq(), "output_index": output_index, "item": {"id": f"ctc_{call_id}", "type": "custom_tool_call", "status": "in_progress", "input": "", "call_id": call_id, "name": name}}
        out.append(_sse_event("response.output_item.added", o))
    else:
        o = {"type": "response.output_item.added", "sequence_number": st.next_seq(), "output_index": output_index, "item": {"id": f"fc_{call_id}", "type": "function_call", "status": "in_progress", "arguments": "", "call_id": call_id, "name": name}}
        if request:
            o["item"] = _apply_namespace_fields(o["item"], request, name, "")
        out.append(_sse_event("response.output_item.added", o))
    st.func_item_added[key] = True
    return out


def _emit_pending_function_args(st: OaiToResponsesState, key: str) -> list[bytes]:
    out: list[bytes] = []
    if not st.func_item_added.get(key) or st.func_item_custom.get(key):
        return out
    args = st._func_args(key)
    sent = st.func_args_sent.get(key, 0)
    if len(args) <= sent:
        return out
    delta = args[sent:]
    call_id = st.func_call_ids.get(key, "")
    ad = {"type": "response.function_call_arguments.delta", "sequence_number": st.next_seq(), "item_id": f"fc_{call_id}", "output_index": st.func_output_ix.get(key, 0), "delta": delta}
    out.append(_sse_event("response.function_call_arguments.delta", ad))
    st.func_args_sent[key] = len(args)
    return out


def convert_chat_stream_chunk_to_responses_events(
    st: OaiToResponsesState,
    chunk: dict,
    request: Optional[dict] = None,
) -> list[bytes]:
    """Convert one Chat Completions streaming chunk to a list of Responses SSE
    event bytes. Mirrors ConvertOpenAIChatCompletionsResponseToOpenAIResponses."""
    out: list[bytes] = []
    raw_text = ""
    # chunk may already be parsed dict; if it's the raw SSE line, caller passes dict.

    # [DONE] handling is done by caller via is_done_marker; but guard here too.
    if chunk is None:
        return out

    obj = _gget_str(chunk, "object")
    if obj and obj != "chat.completion.chunk":
        return out
    if not _gexists(chunk, "choices") or not _is_array(chunk, "choices"):
        # Could be a usage-only chunk
        usage = _gget(chunk, "usage")
        if isinstance(usage, dict):
            _record_usage(st, usage)
        return out

    usage = _gget(chunk, "usage")
    if isinstance(usage, dict):
        _record_usage(st, usage)

    request_for_namespace = _pick_request(request, request)

    if not st.started:
        st.response_id = _gget_str(chunk, "id")
        st.created = _gget_int(chunk, "created")
        st.reset_aggregation()
        st.custom_tool_names = _collect_custom_tool_names(request) if request else set()
        # response.created
        created = {"type": "response.created", "sequence_number": st.next_seq(), "response": {"id": st.response_id, "object": "response", "created_at": st.created, "status": "in_progress", "background": False, "error": None, "output": []}}
        out.append(_sse_event("response.created", created))
        inprog = {"type": "response.in_progress", "sequence_number": st.next_seq(), "response": {"id": st.response_id, "object": "response", "created_at": st.created, "status": "in_progress"}}
        out.append(_sse_event("response.in_progress", inprog))
        st.started = True

    choices = _gget(chunk, "choices")
    if not isinstance(choices, list):
        return out

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        idx = _gget_int(choice, "index")
        delta = _gget(choice, "delta")

        if isinstance(delta, dict):
            # content
            c = _gget_str(delta, "content")
            if c:
                if st.reasoning_id:
                    out.extend(_stop_reasoning(st, st._reasoning_buf_str()))
                    st.reasoning_buf = []
                if idx not in st.msg_output_ix:
                    st.msg_output_ix[idx] = st.alloc_output_index()
                msg_output_index = st.msg_output_ix[idx]
                if not st.msg_item_added.get(idx):
                    item = {"type": "response.output_item.added", "sequence_number": st.next_seq(), "output_index": msg_output_index, "item": {"id": f"msg_{st.response_id}_{idx}", "type": "message", "status": "in_progress", "content": [], "role": "assistant"}}
                    out.append(_sse_event("response.output_item.added", item))
                    st.msg_item_added[idx] = True
                if not st.msg_content_added.get(idx):
                    part = {"type": "response.content_part.added", "sequence_number": st.next_seq(), "item_id": f"msg_{st.response_id}_{idx}", "output_index": msg_output_index, "content_index": 0, "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}}
                    out.append(_sse_event("response.content_part.added", part))
                    st.msg_content_added[idx] = True
                msg = {"type": "response.output_text.delta", "sequence_number": st.next_seq(), "item_id": f"msg_{st.response_id}_{idx}", "output_index": msg_output_index, "content_index": 0, "delta": c, "logprobs": []}
                out.append(_sse_event("response.output_text.delta", msg))
                st.msg_text_buf.setdefault(idx, []).append(c)

            # reasoning_content
            rc = _gget_str(delta, "reasoning_content")
            if not rc:
                rc = _gget_str(delta, "reasoning")
            if rc:
                if st.reasoning_id == "":
                    st.reasoning_id = f"rs_{st.response_id}_{idx}"
                    st.reasoning_index = st.alloc_output_index()
                    item = {"type": "response.output_item.added", "sequence_number": st.next_seq(), "output_index": st.reasoning_index, "item": {"id": st.reasoning_id, "type": "reasoning", "status": "in_progress", "summary": []}}
                    out.append(_sse_event("response.output_item.added", item))
                    part = {"type": "response.reasoning_summary_part.added", "sequence_number": st.next_seq(), "item_id": st.reasoning_id, "output_index": st.reasoning_index, "summary_index": 0, "part": {"type": "summary_text", "text": ""}}
                    out.append(_sse_event("response.reasoning_summary_part.added", part))
                st.reasoning_buf.append(rc)
                msg = {"type": "response.reasoning_summary_text.delta", "sequence_number": st.next_seq(), "item_id": st.reasoning_id, "output_index": st.reasoning_index, "summary_index": 0, "delta": rc}
                out.append(_sse_event("response.reasoning_summary_text.delta", msg))

            # tool calls
            tcs = _gget(delta, "tool_calls")
            if isinstance(tcs, list):
                st.has_tool_calls = True
                if st.reasoning_id:
                    out.extend(_stop_reasoning(st, st._reasoning_buf_str()))
                    st.reasoning_buf = []
                # Close open message for this idx
                out.extend(_emit_message_done_events(st, idx))
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tool_index = _gget_int(tc, "index")
                    key = f"{idx}:{tool_index}"
                    if key not in st.func_args_buf:
                        st.func_args_buf[key] = []
                        st.func_output_ix[key] = st.alloc_output_index()
                    new_call_id = _gget_str(tc, "id")
                    if new_call_id and not st.func_call_ids.get(key):
                        st.func_call_ids[key] = new_call_id
                    name_chunk = _gget_str(tc, "function.name")
                    if name_chunk and not st.func_item_added.get(key):
                        st.func_names[key] = name_chunk
                    args = _gget_str(tc, "function.arguments")
                    if args:
                        st.func_args_buf.setdefault(key, []).append(args)
                    out.extend(_emit_tool_item(st, key, False, request_for_namespace))
                    out.extend(_emit_pending_function_args(st, key))

        # finish_reason
        fr = _gget_str(choice, "finish_reason")
        if fr:
            st.finish_reason = fr
            # Emit message done events for all started messages
            if st.msg_item_added:
                idxs = sorted(st.msg_item_added.keys(), key=lambda i: st.msg_output_ix.get(i, 0))
                for i in idxs:
                    out.extend(_emit_message_done_events(st, i))
            if st.reasoning_id:
                out.extend(_stop_reasoning(st, st._reasoning_buf_str()))
                st.reasoning_buf = []
            # Function call done events
            if st.func_args_buf:
                keys = sorted(st.func_args_buf.keys(), key=lambda k: (st.func_output_ix.get(k, 0), k))
                for key in keys:
                    out.extend(_emit_tool_item(st, key, True, request_for_namespace))
                    out.extend(_emit_pending_function_args(st, key))
                    call_id = st.func_call_ids.get(key, "")
                    if not call_id or st.func_item_done.get(key):
                        continue
                    output_index = st.func_output_ix.get(key, 0)
                    args = st._func_args(key) or "{}"
                    if st.func_item_custom.get(key):
                        input_text = _unwrap_custom_tool_input(args)
                        input_done = {"type": "response.custom_tool_call_input.done", "sequence_number": st.next_seq(), "item_id": f"ctc_{call_id}", "output_index": output_index, "input": input_text}
                        out.append(_sse_event("response.custom_tool_call_input.done", input_done))
                        item_done = {"type": "response.output_item.done", "sequence_number": st.next_seq(), "output_index": output_index, "item": {"id": f"ctc_{call_id}", "type": "custom_tool_call", "status": "completed", "input": input_text, "call_id": call_id, "name": st.func_names.get(key, "")}}
                        out.append(_sse_event("response.output_item.done", item_done))
                        st.func_item_done[key] = True
                        st.func_args_done[key] = True
                        continue
                    fc_done = {"type": "response.function_call_arguments.done", "sequence_number": st.next_seq(), "item_id": f"fc_{call_id}", "output_index": output_index, "arguments": args}
                    out.append(_sse_event("response.function_call_arguments.done", fc_done))
                    item_done = {"type": "response.output_item.done", "sequence_number": st.next_seq(), "output_index": output_index, "item": {"id": f"fc_{call_id}", "type": "function_call", "status": "completed", "arguments": args, "call_id": call_id, "name": st.func_names.get(key, "")}}
                    if request_for_namespace:
                        item_done["item"] = _apply_namespace_fields(item_done["item"], request_for_namespace, st.func_names.get(key, ""), "")
                    out.append(_sse_event("response.output_item.done", item_done))
                    st.func_item_done[key] = True
                    st.func_args_done[key] = True
            st.completion_pending = True

    return out


def _record_usage(st: OaiToResponsesState, usage: dict) -> None:
    if _gexists(usage, "prompt_tokens"):
        st.prompt_tokens = _gget_int(usage, "prompt_tokens")
        st.usage_seen = True
    if _gexists(usage, "prompt_tokens_details.cached_tokens"):
        st.cached_tokens = _gget_int(usage, "prompt_tokens_details.cached_tokens")
        st.usage_seen = True
    if _gexists(usage, "completion_tokens"):
        st.completion_tokens = _gget_int(usage, "completion_tokens")
        st.usage_seen = True
    elif _gexists(usage, "output_tokens"):
        st.completion_tokens = _gget_int(usage, "output_tokens")
        st.usage_seen = True
    if _gexists(usage, "output_tokens_details.reasoning_tokens"):
        st.reasoning_tokens = _gget_int(usage, "output_tokens_details.reasoning_tokens")
        st.usage_seen = True
    elif _gexists(usage, "completion_tokens_details.reasoning_tokens"):
        st.reasoning_tokens = _gget_int(usage, "completion_tokens_details.reasoning_tokens")
        st.usage_seen = True
    if _gexists(usage, "total_tokens"):
        st.total_tokens = _gget_int(usage, "total_tokens")
        st.usage_seen = True


def is_done_marker(line: str) -> bool:
    return line.strip() == "[DONE]"


# ---------------------------------------------------------------------------
# SSE framer (responsesSSEFramer in openai_responses_handlers.go)
# ---------------------------------------------------------------------------


class ResponsesSSEFramer:
    """Reassembles upstream SSE chunks into complete event frames and repairs
    response.completed payloads that are missing response.output."""

    def __init__(self) -> None:
        self.pending: bytearray = bytearray()
        self.output_items: dict[int, bytes] = {}
        self.output_order: list[int] = []
        self.unindexed_output_items: list[bytes] = []

    def write_chunk(self, wfile, chunk: bytes) -> None:
        if not chunk:
            return
        if _sse_needs_line_break(self.pending, chunk):
            self.pending.append(ord("\n"))
        self.pending.extend(chunk)
        while True:
            frame_len = _sse_frame_len(self.pending)
            if frame_len == 0:
                break
            frame = bytes(self.pending[:frame_len])
            del self.pending[:frame_len]
            self._write_frame(wfile, frame)
        if len(bytes(self.pending).strip()) == 0:
            self.pending = bytearray()
            return
        if self.pending and _sse_can_emit_without_delimiter(self.pending):
            self._write_frame(wfile, bytes(self.pending))
            self.pending = bytearray()

    def flush(self, wfile) -> None:
        if not self.pending:
            return
        if len(bytes(self.pending).strip()) == 0:
            self.pending = bytearray()
            return
        if not _sse_can_emit_without_delimiter(self.pending):
            self.pending = bytearray()
            return
        self._write_frame(wfile, bytes(self.pending))
        self.pending = bytearray()

    def _write_frame(self, wfile, frame: bytes) -> None:
        repaired = self._repair_frame(frame)
        _write_sse_chunk(wfile, repaired)

    def _repair_frame(self, frame: bytes) -> bytes:
        payload, ok = _sse_data_payload(frame)
        if not ok or not payload or payload == b"[DONE]":
            return frame
        try:
            obj = json.loads(payload)
        except (TypeError, ValueError):
            return frame
        if not isinstance(obj, dict):
            return frame
        etype = obj.get("type", "")
        if etype == "response.output_item.done":
            self._record_output_item(obj)
        elif etype == "response.completed":
            repaired = self._repair_completed(obj)
            if repaired is not obj:
                return _sse_frame_with_data(frame, json.dumps(repaired, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        return frame

    def _record_output_item(self, payload: dict) -> None:
        item = payload.get("item")
        if not isinstance(item, dict) or not item.get("type"):
            return
        raw = json.dumps(item, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if "output_index" in payload:
            index = int(payload["output_index"])
            if index not in self.output_items:
                self.output_order.append(index)
            self.output_items[index] = raw
            return
        self.unindexed_output_items.append(raw)

    def _repair_completed(self, payload: dict) -> dict:
        if not self.output_order and not self.unindexed_output_items:
            return payload
        resp = payload.get("response")
        if not isinstance(resp, dict):
            return payload
        output = resp.get("output")
        if isinstance(output, list) and len(output) > 0:
            return payload
        items: list = []
        for index in sorted(self.output_order):
            raw = self.output_items.get(index)
            if raw is not None:
                items.append(json.loads(raw))
        for raw in self.unindexed_output_items:
            items.append(json.loads(raw))
        resp["output"] = items
        return payload


def _write_sse_chunk(wfile, chunk: bytes) -> None:
    if not chunk:
        return
    try:
        wfile.write(chunk)
    except (BrokenPipeError, ConnectionResetError, OSError):
        return
    if chunk.endswith(b"\n\n") or chunk.endswith(b"\r\n\r\n"):
        return
    suffix = b"\n\n"
    if chunk.endswith(b"\r\n"):
        suffix = b"\r\n"
    elif chunk.endswith(b"\n"):
        suffix = b"\n"
    try:
        wfile.write(suffix)
    except (BrokenPipeError, ConnectionResetError, OSError):
        return


def _sse_frame_len(chunk: bytes) -> int:
    if not chunk:
        return 0
    lf = chunk.find(b"\n\n")
    crlf = chunk.find(b"\r\n\r\n")
    if lf < 0:
        if crlf < 0:
            return 0
        return crlf + 4
    if crlf < 0:
        return lf + 2
    if lf < crlf:
        return lf + 2
    return crlf + 4


def _sse_needs_more_data(chunk: bytes) -> bool:
    trimmed = chunk.strip()
    if not trimmed:
        return False
    return _sse_has_field(trimmed, b"event:") and not _sse_has_field(trimmed, b"data:")


def _sse_has_field(chunk: bytes, prefix: bytes) -> bool:
    s = chunk
    while s:
        nl = s.find(b"\n")
        if nl >= 0:
            line = s[:nl]
            s = s[nl + 1:]
        else:
            line = s
            s = b""
        line = line.strip()
        if line.startswith(prefix):
            return True
    return False


def _sse_can_emit_without_delimiter(chunk: bytes) -> bool:
    trimmed = chunk.strip()
    if not trimmed or _sse_needs_more_data(trimmed) or not _sse_has_field(trimmed, b"data:"):
        return False
    return _sse_data_lines_valid(trimmed)


def _sse_data_lines_valid(chunk: bytes) -> bool:
    s = chunk
    while s:
        nl = s.find(b"\n")
        if nl >= 0:
            line = s[:nl]
            s = s[nl + 1:]
        else:
            line = s
            s = b""
        line = line.strip()
        if not line or not line.startswith(b"data:"):
            continue
        data = line[len(b"data:"):].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            json.loads(data)
        except (TypeError, ValueError):
            return False
    return True


def _sse_needs_line_break(pending: bytes, chunk: bytes) -> bool:
    if not pending or not chunk:
        return False
    if pending.endswith(b"\n") or pending.endswith(b"\r"):
        return False
    if chunk[0:1] in (b"\n", b"\r"):
        return False
    trimmed = chunk.lstrip(b" \t")
    if not trimmed:
        return False
    for prefix in (b"data:", b"event:", b"id:", b"retry:", b":"):
        if trimmed.startswith(prefix):
            return True
    return False


def _sse_data_payload(frame: bytes) -> tuple[bytes, bool]:
    payload = bytearray()
    found = False
    for line in frame.split(b"\n"):
        line = line.rstrip(b"\r")
        trimmed = line.strip()
        if not trimmed.startswith(b"data:"):
            continue
        data = trimmed[len(b"data:"):].strip()
        if found:
            payload.append(ord("\n"))
        payload.extend(data)
        found = True
    return bytes(payload), found


def _sse_frame_with_data(frame: bytes, payload: bytes) -> bytes:
    out = bytearray()
    for line in frame.split(b"\n"):
        line = line.rstrip(b"\r")
        trimmed = line.strip()
        if not trimmed or trimmed.startswith(b"data:"):
            continue
        out.extend(line)
        out.append(ord("\n"))
    for line in payload.split(b"\n"):
        out.extend(b"data: ")
        out.extend(line)
        out.append(ord("\n"))
    out.append(ord("\n"))
    return bytes(out)


# ---------------------------------------------------------------------------
# Upstream client: calls llama-server /v1/chat/completions
# ---------------------------------------------------------------------------


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, body: bytes = b"") -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.body = body


def _upstream_host_port(host: str, port: int) -> tuple[str, int]:
    return host, int(port)


def upstream_chat_completions_nonstream(
    host: str, port: int, body: dict, timeout: float = 300.0
) -> dict:
    """Send a non-streaming chat completions request to llama-server."""
    data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection(host, int(port), timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=data, headers={"Content-Type": "application/json", "Accept": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status >= 400:
            raise UpstreamError(resp.status, raw.decode("utf-8", errors="replace"), raw)
        return json.loads(raw)
    finally:
        conn.close()


def upstream_chat_completions_stream(
    host: str, port: int, body: dict, timeout: float = 0
) -> Iterator[bytes]:
    """Yield raw SSE line bytes from a streaming chat completions request.

    timeout=0 means no socket timeout (block indefinitely). This is critical
    for long-running generations where the model may pause between tokens
    (e.g., during extended reasoning). A finite timeout would cause the
    connection to abort mid-stream, terminating the Codex session."""
    body["stream"] = True
    data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection(host, int(port), timeout=timeout if timeout > 0 else None)
    try:
        conn.request("POST", "/v1/chat/completions", body=data, headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        resp = conn.getresponse()
        if resp.status >= 400:
            raw = resp.read()
            raise UpstreamError(resp.status, raw.decode("utf-8", errors="replace"), raw)
        # Read line-by-line; yield raw bytes including newlines so the framer can
        # reassemble frames.
        buf = bytearray()
        while True:
            chunk = resp.read(4096)
            if not chunk:
                if buf:
                    yield bytes(buf)
                    buf = bytearray()
                break
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buf[: nl + 1])
                del buf[: nl + 1]
                yield line
        if buf:
            yield bytes(buf)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error response helpers
# ---------------------------------------------------------------------------


def _stream_error_code(status: int) -> str:
    if status == 401:
        return "invalid_api_key"
    if status == 403:
        return "insufficient_quota"
    if status == 429:
        return "rate_limit_exceeded"
    if status == 404:
        return "model_not_found"
    if status == 408:
        return "request_timeout"
    if status >= 500:
        return "internal_server_error"
    if status >= 400:
        return "invalid_request_error"
    return "unknown_error"


def build_stream_error_chunk(status: int, err_text: str, seq: int = 0) -> bytes:
    if status <= 0:
        status = 500
    if seq < 0:
        seq = 0
    message = (err_text or "").strip()
    if not message:
        message = http.client.responses.get(status, "Internal Server Error")
    code = _stream_error_code(status)
    trimmed = (err_text or "").strip()
    if trimmed:
        try:
            payload = json.loads(trimmed)
            if isinstance(payload, dict):
                t = payload.get("type")
                if t == "error":
                    m = payload.get("message")
                    if isinstance(m, str) and m.strip():
                        message = m.strip()
                    v = payload.get("code")
                    if v is not None:
                        code = str(v).strip() or code
                    v = payload.get("sequence_number")
                    if isinstance(v, (int, float)) and seq == 0:
                        seq = int(v)
                e = payload.get("error")
                if isinstance(e, dict):
                    m = e.get("message")
                    if isinstance(m, str) and m.strip():
                        message = m.strip()
                    v = e.get("code")
                    if v is not None:
                        code = str(v).strip() or code
        except (TypeError, ValueError):
            pass
    if not code.strip():
        code = "unknown_error"
    obj = {"type": "error", "code": code, "message": message, "sequence_number": seq}
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_error_response_body(status: int, err_text: str) -> dict:
    if status <= 0:
        status = 500
    if not (err_text or "").strip():
        err_text = http.client.responses.get(status, "Internal Server Error")
    trimmed = (err_text or "").strip()
    if trimmed:
        try:
            payload = json.loads(trimmed)
            if isinstance(payload, dict):
                return payload
        except (TypeError, ValueError):
            pass
    err_type = "invalid_request_error"
    code = ""
    if status == 401:
        err_type = "authentication_error"
        code = "invalid_api_key"
    elif status == 403:
        err_type = "permission_error"
        code = "insufficient_quota"
    elif status == 429:
        err_type = "rate_limit_error"
        code = "rate_limit_exceeded"
    elif status == 404:
        err_type = "invalid_request_error"
        code = "model_not_found"
    elif status >= 500:
        err_type = "server_error"
        code = "internal_server_error"
    return {"error": {"message": err_text, "type": err_type, "code": code}}


def _write_json_response(handler, status: int, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass


def _write_error_response(handler, status: int, err_text: str) -> None:
    _write_json_response(handler, status, build_error_response_body(status, err_text))


def _read_request_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# HTTP handler: POST /v1/responses, POST /v1/responses/compact
# ---------------------------------------------------------------------------


class ResponsesProxyHandler:
    """Handles Responses API HTTP requests by proxying to llama-server's
    /v1/chat/completions and translating formats.

    Usage from a BaseHTTPRequestHandler subclass::

        proxy = ResponsesProxyHandler(host="127.0.0.1", port=8080)
        proxy.handle_responses(handler)        # POST /v1/responses
        proxy.handle_compact(handler)          # POST /v1/responses/compact
        proxy.handle_websocket_upgrade(handler)  # GET /v1/responses
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, log_fn: Optional[Callable[[str], None]] = None) -> None:
        self.host = host
        self.port = int(port)
        self.log_fn = log_fn or (lambda msg: None)

    def _log(self, msg: str) -> None:
        try:
            self.log_fn(msg)
        except Exception:
            pass

    def handle_responses(self, handler) -> None:
        """POST /v1/responses entry point."""
        try:
            request = _read_request_body(handler)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            _write_error_response(handler, 400, f"Invalid request: {exc}")
            return

        stream = bool(_gget(request, "stream"))
        if stream:
            self._handle_streaming(handler, request)
        else:
            self._handle_non_streaming(handler, request)

    def handle_compact(self, handler) -> None:
        """POST /v1/responses/compact — non-streaming only.

        Go's CLIProxyAPI forwards to the upstream's native /responses/compact
        endpoint (for providers that support it). Since llama-server has no
        native Responses endpoint, we convert to chat completions and return
        the raw chat completions JSON without Responses wrapping. This matches
        the compact semantic: a lightweight response without full Responses
        event conversion."""
        try:
            request = _read_request_body(handler)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            _write_error_response(handler, 400, f"Invalid request: {exc}")
            return
        if _gget(request, "stream"):
            _write_error_response(handler, 400, "Streaming not supported for compact responses")
            return
        model_name = _gget_str(request, "model")
        chat_req = convert_responses_request_to_chat_completions(model_name, request, stream=False)
        try:
            chat_resp = upstream_chat_completions_nonstream(self.host, self.port, chat_req)
        except UpstreamError as exc:
            _write_error_response(handler, exc.status, exc.message)
            return
        except (socket.error, OSError) as exc:
            _write_error_response(handler, 502, f"upstream connection failed: {exc}")
            return
        body = json.dumps(chat_resp, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        try:
            handler.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _handle_non_streaming(self, handler, request: dict) -> None:
        model_name = _gget_str(request, "model")
        chat_req = convert_responses_request_to_chat_completions(model_name, request, stream=False)
        try:
            chat_resp = upstream_chat_completions_nonstream(self.host, self.port, chat_req)
        except UpstreamError as exc:
            _write_error_response(handler, exc.status, exc.message)
            return
        except (socket.error, OSError) as exc:
            _write_error_response(handler, 502, f"upstream connection failed: {exc}")
            return
        responses_json = convert_chat_response_to_responses_json(chat_resp, request)
        _write_json_response(handler, 200, responses_json)

    def _handle_streaming(self, handler, request: dict) -> None:
        _stream_responses_direct(handler, self, request)

    def handle_websocket_upgrade(self, handler) -> bool:
        return _handle_websocket_upgrade_impl(handler, self)


# We parse SSE data lines directly and feed each chunk dict to the
# converter. This is cleaner for our use case since llama-server produces
# well-formed SSE.

def _stream_responses_direct(handler, proxy: ResponsesProxyHandler, request: dict) -> None:
    """Stream Responses SSE events by parsing upstream chat completions SSE
    line-by-line and converting each chunk through the state machine.

    HTTP headers are deferred until the first upstream chunk arrives, so if
    the upstream fails immediately we can still return a proper HTTP error
    response (matching Go's handleStreamingResponse peek-first pattern)."""
    model_name = _gget_str(request, "model")
    chat_req = convert_responses_request_to_chat_completions(model_name, request, stream=True)

    st = OaiToResponsesState()
    wfile = handler.wfile
    headers_sent = False
    input_count = len(_gget(request, "input") or [])
    tools_count = len(_gget(chat_req, "tools") or [])
    tool_choice = _gget(chat_req, "tool_choice", "auto")
    chat_tools = _gget(chat_req, "tools") or []
    chat_tool_names = [_gget_str(t, "function.name") for t in chat_tools if isinstance(t, dict)]
    chat_msgs = _gget(chat_req, "messages") or []
    msg_summary = []
    for m in chat_msgs[-6:]:
        if not isinstance(m, dict):
            continue
        role = _gget_str(m, "role")
        if role == "tool":
            msg_summary.append(f"tool(tid={_gget_str(m, 'tool_call_id')[:12]}..)")
        elif role == "assistant" and _gget(m, "tool_calls"):
            tcs = _gget(m, "tool_calls")
            names = [_gget_str(tc, "function.name") for tc in tcs if isinstance(tc, dict)]
            msg_summary.append(f"asst(tc={names})")
        elif role == "assistant":
            c = _gget_str(m, "content")[:40]
            msg_summary.append(f"asst({c})")
        else:
            c = _gget_str(m, "content")[:40]
            msg_summary.append(f"{role}({c})")
    max_tok = _gget(chat_req, "max_tokens")
    proxy._log(f"sse: upstream request model={model_name} input_items={input_count} tools={tools_count} tool_choice={tool_choice} max_tokens={max_tok} tool_names={chat_tool_names}")
    proxy._log(f"sse: last msgs: {msg_summary}")
    # Check if system nudge was injected / tool_choice overridden
    if chat_msgs and isinstance(chat_msgs[-1], dict) and chat_msgs[-1].get("role") == "system" and "MUST call" in chat_msgs[-1].get("content", ""):
        actual_tc = _gget(chat_req, "tool_choice")
        proxy._log(f"sse: injected system nudge + tool_choice={actual_tc} (tool result -> force tool call)")

    def send_sse_headers() -> None:
        nonlocal headers_sent
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        # Connection: close (not keep-alive) because http.server does not
        # auto-chunk HTTP/1.1 streaming responses. Go uses keep-alive via
        # Gin's chunked transfer encoding, which http.server lacks.
        handler.send_header("Connection", "close")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        headers_sent = True

    def emit(data: bytes) -> None:
        try:
            wfile.write(data)
            wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    try:
        stream = upstream_chat_completions_stream(proxy.host, proxy.port, chat_req)
        for raw_line in stream:
            line = raw_line.rstrip(b"\r\n")
            if not line:
                continue
            if line.startswith(b"data:"):
                payload = line[len(b"data:"):].strip()
            elif line.startswith(b"event:") or line.startswith(b":"):
                continue
            else:
                payload = line.strip()
            if not payload or payload == b"[DONE]":
                if st.completion_pending and not st.completed_emitted:
                    if not headers_sent:
                        send_sse_headers()
                    st.completed_emitted = True
                    final_ev = _build_completed_event(st, request)
                    final_obj = _sse_event_to_json(final_ev)
                    if final_obj is not None:
                        resp = _gget(final_obj, "response")
                        end_turn = _gget(resp, "end_turn") if isinstance(resp, dict) else None
                        output = _gget(resp, "output") if isinstance(resp, dict) else []
                        output_types = [_gget_str(it, "type") for it in output if isinstance(it, dict)]
                        proxy._log(f"sse: response.completed([DONE]) end_turn={end_turn} output_types={output_types} finish_reason={st.finish_reason} has_tool_calls={st.has_tool_calls}")
                    emit(final_ev)
                continue
            try:
                chunk = json.loads(payload)
            except (TypeError, ValueError):
                continue
            choices = _gget(chunk, "choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0]
                delta = _gget(ch0, "delta")
                has_tc = isinstance(_gget(delta, "tool_calls"), list)
                text_content = _gget_str(delta, "content")
                has_text = bool(text_content)
                fr = _gget_str(ch0, "finish_reason")
                if fr or has_tc:
                    proxy._log(f"sse: upstream chunk finish_reason={fr or 'none'} has_tool_calls={has_tc} has_text={has_text}")
                    if fr == "stop" and not has_tc:
                        full_text = "".join("".join(v) for v in st.msg_text_buf.values())
                        proxy._log(f"sse: full assistant text ({len(full_text)} chars): {full_text[:500]}")
            events = convert_chat_stream_chunk_to_responses_events(st, chunk, request)
            if events and not headers_sent:
                send_sse_headers()
            for ev in events:
                ev_obj = _sse_event_to_json(ev)
                if ev_obj is not None:
                    etype = _gget_str(ev_obj, "type")
                    if etype == "response.output_item.done":
                        item = _gget(ev_obj, "item")
                        if isinstance(item, dict):
                            item_type = _gget_str(item, "type")
                            item_text = ""
                            if item_type == "message":
                                content = _gget(item, "content")
                                if isinstance(content, list):
                                    for part in content:
                                        if isinstance(part, dict) and _gget_str(part, "type") == "output_text":
                                            item_text = _gget_str(part, "text")[:200]
                            elif item_type in ("function_call", "custom_tool_call"):
                                item_text = f"name={_gget_str(item, 'name')} args={_gget_str(item, 'arguments')[:100] if item_type == 'function_call' else _gget_str(item, 'input')[:100]}"
                            proxy._log(f"sse: output_item.done type={item_type} call_id={_gget_str(item, 'call_id')} {item_text}")
                    if etype == "response.completed":
                        resp = _gget(ev_obj, "response")
                        end_turn = _gget(resp, "end_turn") if isinstance(resp, dict) else None
                        output = _gget(resp, "output") if isinstance(resp, dict) else []
                        output_types = [_gget_str(it, "type") for it in output if isinstance(it, dict)]
                        proxy._log(f"sse: response.completed end_turn={end_turn} output_types={output_types} finish_reason={st.finish_reason} has_tool_calls={st.has_tool_calls}")
                emit(ev)
        if not headers_sent:
            send_sse_headers()
        if st.completion_pending and not st.completed_emitted:
            st.completed_emitted = True
            final_ev = _build_completed_event(st, request)
            final_obj = _sse_event_to_json(final_ev)
            if final_obj is not None:
                resp = _gget(final_obj, "response")
                end_turn = _gget(resp, "end_turn") if isinstance(resp, dict) else None
                output = _gget(resp, "output") if isinstance(resp, dict) else []
                output_types = [_gget_str(it, "type") for it in output if isinstance(it, dict)]
                proxy._log(f"sse: response.completed(built) end_turn={end_turn} output_types={output_types} finish_reason={st.finish_reason} has_tool_calls={st.has_tool_calls}")
            emit(final_ev)
        emit(b"\n")
    except UpstreamError as exc:
        proxy._log(f"sse: upstream error status={exc.status} msg={exc.message[:200]}")
        if not headers_sent:
            _write_error_response(handler, exc.status, exc.message)
        else:
            err_chunk = build_stream_error_chunk(exc.status, exc.message, 0)
            emit(f"event: error\ndata: {err_chunk.decode('utf-8')}\n\n".encode("utf-8"))
    except (socket.error, OSError) as exc:
        proxy._log(f"sse: socket error: {exc}")
        if not headers_sent:
            _write_error_response(handler, 502, f"upstream connection failed: {exc}")
        else:
            err_chunk = build_stream_error_chunk(502, str(exc), 0)
            emit(f"event: error\ndata: {err_chunk.decode('utf-8')}\n\n".encode("utf-8"))
    except Exception as exc:
        proxy._log(f"sse: unexpected error: {exc}")
        raise
    finally:
        handler.close_connection = True


# ---------------------------------------------------------------------------
# WebSocket request normalization
# (openai_responses_websocket_requests.go)
# ---------------------------------------------------------------------------

_WS_REQUEST_TYPE_CREATE = "response.create"
_WS_REQUEST_TYPE_APPEND = "response.append"
_WS_EVENT_TYPE_ERROR = "error"
_WS_EVENT_TYPE_COMPLETED = "response.completed"
_WS_EVENT_TYPE_DONE = "response.done"
_WS_DONE_MARKER = "[DONE]"
_CODEX_LOCAL_COMPACTION_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary "
    "of its thinking process. You also have access to the state of the tools that "
    "were used by that language model. Use this to build on the work that has "
    "already been done and avoid duplicating work. Here is the summary produced by "
    "the other language model, use the information in this summary to assist with "
    "your own analysis:"
)


def _is_responses_tool_call_type(item_type: str) -> bool:
    return item_type.strip() in ("function_call", "custom_tool_call")


def _is_responses_tool_call_output_type(item_type: str) -> bool:
    return item_type.strip() in ("function_call_output", "custom_tool_call_output")


def _merge_json_array_raw(existing_raw: Any, append_raw: Any) -> list:
    """Merge two JSON arrays (as Python lists). None/empty -> []."""
    existing = existing_raw if isinstance(existing_raw, list) else []
    append_items = append_raw if isinstance(append_raw, list) else []
    return list(existing) + list(append_items)


def _normalize_json_array_raw(raw: Any) -> list:
    if isinstance(raw, list):
        return raw
    return []


def _input_contains_full_transcript(inp: Any) -> bool:
    if not isinstance(inp, list):
        return False
    for item in inp:
        if isinstance(item, dict):
            t = _gget_str(item, "type")
            if t in ("compaction", "compaction_summary"):
                return True
    return False


def _input_without_compaction_items(inp: Any) -> list:
    if not isinstance(inp, list):
        return _normalize_json_array_raw(inp)
    return [item for item in inp if not (isinstance(item, dict) and _gget_str(item, "type") in ("compaction", "compaction_summary"))]


def _codex_local_compaction_message_text(message: dict) -> str:
    content = _gget(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and _gget_str(part, "type") == "input_text":
                parts.append(_gget_str(part, "text"))
        return "".join(parts)
    return ""


def _input_has_codex_local_compaction_summary(inp: Any) -> bool:
    if not isinstance(inp, list):
        return False
    has_summary = False
    for index, item in enumerate(inp):
        if not isinstance(item, dict):
            return False
        item_type = _gget_str(item, "type").strip()
        if item_type == "additional_tools":
            tools = _gget(item, "tools")
            if index != 0 or _gget_str(item, "role") != "developer" or not isinstance(tools, list):
                return False
            for tool in tools:
                if not isinstance(tool, dict) or _gget_str(tool, "type") == "":
                    return False
            continue
        if item_type != "" and item_type != "message":
            return False
        role = _gget_str(item, "role").strip()
        if role not in ("user", "developer"):
            return False
        if role == "user" and _codex_local_compaction_message_text(item).startswith(_CODEX_LOCAL_COMPACTION_SUMMARY_PREFIX + "\n"):
            has_summary = True
    return has_summary


def _input_satisfies_pending_tool_calls(inp: Any, pending_call_ids: list[str]) -> bool:
    if not pending_call_ids:
        return True
    if not isinstance(inp, list):
        return False
    outputs: set[str] = set()
    for item in inp:
        if isinstance(item, dict):
            itype = _gget_str(item, "type").strip()
            if itype in ("function_call_output", "custom_tool_call_output"):
                call_id = _gget_str(item, "call_id").strip()
                if call_id:
                    outputs.add(call_id)
    for call_id in pending_call_ids:
        call_id = call_id.strip()
        if call_id and call_id not in outputs:
            return False
    return True


def _should_replace_websocket_transcript(raw: dict, next_input: Any) -> bool:
    request_type = _gget_str(raw, "type").strip()
    if request_type not in (_WS_REQUEST_TYPE_CREATE, _WS_REQUEST_TYPE_APPEND):
        return False
    if _gget_str(raw, "previous_response_id").strip():
        return False
    if not isinstance(next_input, list):
        return False
    if request_type == _WS_REQUEST_TYPE_CREATE and not _gexists(raw, "previous_response_id") and _input_has_codex_local_compaction_summary(next_input):
        return True
    for item in next_input:
        if not isinstance(item, dict):
            continue
        itype = _gget_str(item, "type").strip()
        if itype in ("function_call", "custom_tool_call"):
            return True
        if itype == "message" and _gget_str(item, "role").strip() == "assistant":
            return True
    return False


def _normalize_response_transcript_replacement(raw: dict, last_request: Optional[dict]) -> dict:
    normalized = dict(raw)
    normalized.pop("type", None)
    normalized.pop("previous_response_id", None)
    if not _gexists(normalized, "model") and last_request:
        model_name = _gget_str(last_request, "model").strip()
        if model_name:
            normalized["model"] = model_name
    if not _gexists(normalized, "instructions") and last_request:
        if _gexists(last_request, "instructions"):
            normalized["instructions"] = _gget(last_request, "instructions")
    normalized["stream"] = True
    return normalized


def _dedupe_function_calls_by_call_id(items: list) -> list:
    seen: set[str] = set()
    filtered: list = []
    for item in items:
        if not isinstance(item, dict):
            if item is not None:
                filtered.append(item)
            continue
        itype = _gget_str(item, "type").strip()
        if _is_responses_tool_call_type(itype):
            call_id = _gget_str(item, "call_id").strip()
            if call_id:
                if call_id in seen:
                    continue
                seen.add(call_id)
        filtered.append(item)
    return filtered


def _dedupe_input_items_by_id(items: list) -> list:
    if not items:
        return []
    meta: list[tuple[str, str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            meta.append(("", "", ""))
            continue
        meta.append((_gget_str(item, "type").strip(), _gget_str(item, "id").strip(), _gget_str(item, "call_id").strip()))

    referenced_call_ids: set[str] = set()
    for i, m in enumerate(meta):
        if m[0] in ("function_call_output", "custom_tool_call_output") and m[2]:
            referenced_call_ids.add(m[2])

    keep_index_by_id: dict[str, int] = {}
    keep_referenced_by_id: dict[str, bool] = {}
    for i, m in enumerate(meta):
        item_id = m[1]
        if not item_id:
            continue
        referenced = m[2] in referenced_call_ids and bool(m[2])
        if item_id not in keep_index_by_id:
            keep_index_by_id[item_id] = i
            keep_referenced_by_id[item_id] = referenced
            continue
        if referenced or not keep_referenced_by_id[item_id]:
            keep_index_by_id[item_id] = i
            keep_referenced_by_id[item_id] = referenced

    filtered: list = []
    for i, item in enumerate(items):
        if item is None:
            continue
        item_id = meta[i][1]
        if item_id and keep_index_by_id.get(item_id) != i:
            continue
        filtered.append(item)
    return filtered


def normalize_response_create_request(raw: dict) -> tuple[Optional[dict], Optional[dict], Optional[tuple[int, str]]]:
    """Returns (normalized, updated_last_request, error). error = (status, message) or None."""
    normalized = dict(raw)
    normalized.pop("type", None)
    normalized["stream"] = True
    if not _gexists(normalized, "input"):
        normalized["input"] = []
    model_name = _gget_str(normalized, "model").strip()
    if not model_name:
        return None, None, (400, "missing model in response.create request")
    return normalized, dict(normalized), None


def normalize_response_subsequent_request(
    raw: dict,
    last_request: Optional[dict],
    last_response_output: Any,
    last_response_id: str,
    last_response_pending_tool_call_ids: list[str],
    allow_incremental_input_with_previous_response_id: bool = True,
    allow_compaction_replay_bypass: bool = True,
) -> tuple[Optional[dict], Optional[dict], Optional[tuple[int, str]]]:
    if not last_request:
        return None, last_request, (400, "websocket request received before response.create")

    next_input = _gget(raw, "input")
    if not isinstance(next_input, list):
        return None, last_request, (400, "websocket request requires array field: input")

    if _should_replace_websocket_transcript(raw, next_input):
        normalized = _normalize_response_transcript_replacement(raw, last_request)
        return normalized, dict(normalized), None

    if allow_incremental_input_with_previous_response_id:
        prev = _gget_str(raw, "previous_response_id").strip()
        if prev == "":
            if not _input_satisfies_pending_tool_calls(next_input, last_response_pending_tool_call_ids):
                normalized = _normalize_response_transcript_replacement(raw, last_request)
                return normalized, dict(normalized), None
            prev = (last_response_id or "").strip()
        if prev:
            normalized = dict(raw)
            normalized.pop("type", None)
            normalized["previous_response_id"] = prev
            if not _gexists(normalized, "model"):
                model_name = _gget_str(last_request, "model").strip()
                if model_name:
                    normalized["model"] = model_name
            if not _gexists(normalized, "instructions") and _gexists(last_request, "instructions"):
                normalized["instructions"] = _gget(last_request, "instructions")
            normalized["stream"] = True
            return normalized, dict(normalized), None

    # Merge input with last request + last response output
    if allow_compaction_replay_bypass and _input_contains_full_transcript(next_input):
        merged_input = next_input
    else:
        append_input = next_input
        if _input_contains_full_transcript(next_input):
            append_input = _input_without_compaction_items(next_input)
        existing_input = _gget(last_request, "input")
        merged_input = _merge_json_array_raw(_merge_json_array_raw(_normalize_json_array_raw(existing_input), _normalize_json_array_raw(last_response_output)), append_input)

    merged_input = _dedupe_function_calls_by_call_id(merged_input)
    merged_input = _dedupe_input_items_by_id(merged_input)

    normalized = dict(raw)
    normalized.pop("type", None)
    normalized.pop("previous_response_id", None)
    normalized["input"] = merged_input
    if not _gexists(normalized, "model"):
        model_name = _gget_str(last_request, "model").strip()
        if model_name:
            normalized["model"] = model_name
    if not _gexists(normalized, "instructions") and _gexists(last_request, "instructions"):
        normalized["instructions"] = _gget(last_request, "instructions")
    normalized["stream"] = True
    return normalized, dict(normalized), None


def normalize_websocket_request(
    raw: dict,
    last_request: Optional[dict],
    last_response_output: Any,
    last_response_id: str = "",
    last_response_pending_tool_call_ids: Optional[list[str]] = None,
) -> tuple[Optional[dict], Optional[dict], Optional[tuple[int, str]]]:
    """Normalize a websocket request (response.create / response.append).
    Returns (request_json, updated_last_request, error).

    For stateless backends like llama-server, previous_response_id is not
    supported, so allow_incremental_input_with_previous_response_id is False.
    This forces the normalizer to merge the new input with the last request
    and last response output, rebuilding the full conversation context."""
    request_type = _gget_str(raw, "type").strip()
    pending = last_response_pending_tool_call_ids or []
    if request_type == _WS_REQUEST_TYPE_CREATE:
        if not last_request:
            return normalize_response_create_request(raw)
        return normalize_response_subsequent_request(
            raw, last_request, last_response_output, last_response_id, pending,
            allow_incremental_input_with_previous_response_id=False,
        )
    if request_type == _WS_REQUEST_TYPE_APPEND:
        return normalize_response_subsequent_request(
            raw, last_request, last_response_output, last_response_id, pending,
            allow_incremental_input_with_previous_response_id=False,
        )
    return None, last_request, (400, f"unsupported websocket request type: {request_type}")


# ---------------------------------------------------------------------------
# WebSocket completion event helpers
# (openai_responses_websocket_forward.go)
# ---------------------------------------------------------------------------


def _is_websocket_completion_event(event_type: str) -> bool:
    return event_type in (_WS_EVENT_TYPE_COMPLETED, _WS_EVENT_TYPE_DONE)


def _sse_event_to_json(ev) -> Optional[dict]:
    """Extract the JSON dict from an SSE-formatted event (bytes or str).

    The converter returns events as b"event: type\\ndata: {json}\\n\\n".
    WebSocket clients need the raw JSON object, not the SSE wrapper."""
    if isinstance(ev, bytes):
        ev = ev.decode("utf-8", errors="replace")
    for line in ev.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
            if line:
                try:
                    return json.loads(line)
                except (TypeError, ValueError):
                    return None
    return None


def _websocket_json_payloads_from_chunk(chunk: bytes) -> list[bytes]:
    """Extract JSON payloads from an SSE-style chunk or bare JSON."""
    payloads: list[bytes] = []
    for line in chunk.split(b"\n"):
        line = line.strip()
        if not line or line.startswith(b"event:"):
            continue
        if line.startswith(b"data:"):
            line = line[len(b"data:"):].strip()
        if not line or line == b"[DONE]":
            continue
        try:
            json.loads(line)
            payloads.append(line)
        except (TypeError, ValueError):
            continue
    if payloads:
        return payloads
    trimmed = chunk.strip()
    if trimmed.startswith(b"data:"):
        trimmed = trimmed[len(b"data:"):].strip()
    if trimmed and trimmed != b"[DONE]":
        try:
            json.loads(trimmed)
            payloads.append(trimmed)
        except (TypeError, ValueError):
            pass
    return payloads


def _is_complete_websocket_tool_call(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    call_id = _gget(item, "call_id")
    name = _gget(item, "name")
    if not isinstance(call_id, str) or not call_id.strip() or not isinstance(name, str) or not name.strip():
        return False
    itype = _gget_str(item, "type").strip()
    if itype == "function_call":
        return _gexists(item, "arguments") and isinstance(_gget(item, "arguments"), str)
    if itype == "custom_tool_call":
        return _gexists(item, "input") and isinstance(_gget(item, "input"), str)
    return False


def _response_completed_output_from_payload(payload: dict, output_items_by_index: dict, output_items_fallback: list) -> list:
    output = _gget(payload, "response.output")
    if isinstance(output, list) and output:
        return output
    if not output_items_by_index and not output_items_fallback:
        return []

    items: list = []

    def append_collected(raw: dict) -> None:
        if not isinstance(raw, dict):
            return
        if _is_responses_tool_call_type(_gget_str(raw, "type")) and not _is_complete_websocket_tool_call(raw):
            return
        items.append(raw)

    for index in sorted(output_items_by_index.keys()):
        append_collected(output_items_by_index[index])
    for item in output_items_fallback:
        append_collected(item)
    return items


def _response_completed_id_from_payload(payload: dict) -> str:
    return _gget_str(payload, "response.id").strip()


def _record_pending_tool_call_ids(pending: set[str], payload: dict) -> None:
    item = _gget(payload, "item")
    if isinstance(item, dict):
        _update_pending_tool_call_ids(pending, item)
    output = _gget(payload, "response.output")
    if isinstance(output, list):
        for it in output:
            if isinstance(it, dict):
                _update_pending_tool_call_ids(pending, it)


def _update_pending_tool_call_ids(pending: set[str], item: dict) -> None:
    if not isinstance(item, dict):
        return
    itype = _gget_str(item, "type").strip()
    if itype in ("function_call", "custom_tool_call"):
        if _is_complete_websocket_tool_call(item):
            call_id = _gget_str(item, "call_id").strip()
            if call_id:
                pending.add(call_id)
    elif itype in ("function_call_output", "custom_tool_call_output"):
        call_id = _gget_str(item, "call_id").strip()
        if call_id:
            pending.discard(call_id)


def _build_websocket_error_payload(status: int, err_text: str) -> dict:
    payload: dict = {"type": _WS_EVENT_TYPE_ERROR, "status": status}
    body = build_error_response_body(status, err_text)
    if isinstance(body, dict) and "error" in body:
        payload["error"] = body["error"]
    else:
        payload["error"] = {"type": "server_error", "message": err_text}
    return payload


# ---------------------------------------------------------------------------
# WebSocket handler: GET /v1/responses upgrade
# (openai_responses_websocket.go)
# ---------------------------------------------------------------------------


# Minimal WebSocket frame protocol (RFC 6455) using stdlib only.
# This avoids dependency on the websockets library's evolving API for
# adopting an existing socket.

_WS_OPCODE_CONT = 0x0
_WS_OPCODE_TEXT = 0x1
_WS_OPCODE_BINARY = 0x2
_WS_OPCODE_CLOSE = 0x8
_WS_OPCODE_PING = 0x9
_WS_OPCODE_PONG = 0xA


class _WebSocketConnection:
    """Minimal synchronous WebSocket server connection over a raw socket."""

    _PING_INTERVAL = 30.0  # seconds between unsolicited pings

    def __init__(self, sock: socket.socket, initial_data: bytes = b"") -> None:
        self.sock = sock
        self._recv_buf = bytearray(initial_data)
        self._closed = False
        self._write_lock = threading.Lock()
        self._ping_stop = threading.Event()
        self._ping_thread: Optional[threading.Thread] = None

    def start_keepalive(self) -> None:
        """Start a background thread that sends periodic WebSocket pings."""
        if self._ping_thread is not None:
            return

        def _ping_loop() -> None:
            while not self._ping_stop.wait(self._PING_INTERVAL):
                if self._closed:
                    return
                try:
                    self._send_frame(_WS_OPCODE_PING, b"keepalive")
                except OSError:
                    return

        self._ping_thread = threading.Thread(target=_ping_loop, daemon=True)
        self._ping_thread.start()

    def stop_keepalive(self) -> None:
        self._ping_stop.set()
        if self._ping_thread is not None:
            self._ping_thread.join(timeout=2.0)
            self._ping_thread = None

    def recv_message(self) -> Optional[bytes | str]:
        """Read one complete WebSocket message. Returns str for text, bytes for
        binary, or None if the connection was closed cleanly."""
        while True:
            msg = self._try_parse_frame()
            if msg is not None:
                return msg
            # Need more data
            try:
                chunk = self.sock.recv(4096)
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            self._recv_buf.extend(chunk)

    def _try_parse_frame(self) -> Optional[bytes | str]:
        buf = self._recv_buf
        if len(buf) < 2:
            return None
        b0 = buf[0]
        b1 = buf[1]
        fin = (b0 & 0x80) != 0
        opcode = b0 & 0x0F
        masked = (b1 & 0x80) != 0
        length = b1 & 0x7F
        pos = 2
        if length == 126:
            if len(buf) < pos + 2:
                return None
            length = struct.unpack("!H", buf[pos:pos + 2])[0]
            pos += 2
        elif length == 127:
            if len(buf) < pos + 8:
                return None
            length = struct.unpack("!Q", buf[pos:pos + 8])[0]
            pos += 8
        if masked:
            if len(buf) < pos + 4:
                return None
            mask = buf[pos:pos + 4]
            pos += 4
        if len(buf) < pos + length:
            return None
        payload = bytes(buf[pos:pos + length])
        del buf[:pos + length]
        if masked:
            payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))

        if opcode == _WS_OPCODE_CLOSE:
            self._closed = True
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return None
        if opcode == _WS_OPCODE_PING:
            self._send_frame(_WS_OPCODE_PONG, payload)
            return None  # control frame, keep reading
        if opcode == _WS_OPCODE_PONG:
            return None
        # For text/binary, we handle single-frame messages (fin=1).
        # Fragmented messages (fin=0) are rare for our use case; we accumulate.
        if fin:
            if opcode == _WS_OPCODE_TEXT:
                return payload.decode("utf-8", errors="replace")
            return payload
        # Fragmented: accumulate (simplified - append to buffer and retry)
        # For our use case, clients send single-frame messages.
        return payload

    def send_text(self, data: str) -> None:
        self._send_frame(_WS_OPCODE_TEXT, data.encode("utf-8"))

    def send_binary(self, data: bytes) -> None:
        self._send_frame(_WS_OPCODE_BINARY, data)

    def close(self, code: int = 1000, reason: str = "") -> None:
        if self._closed:
            return
        payload = struct.pack("!H", code) + reason.encode("utf-8")[:123]
        try:
            self._send_frame(_WS_OPCODE_CLOSE, payload)
        except OSError:
            pass
        self._closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytearray()
        header.append(0x80 | opcode)  # FIN + opcode
        length = len(payload)
        if length < 126:
            header.append(length)
        elif length < 65536:
            header.append(126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", length))
        with self._write_lock:
            if self._closed:
                return
            try:
                self.sock.sendall(bytes(header) + payload)
            except OSError:
                self._closed = True


def _handle_websocket_upgrade_impl(handler, proxy: "ResponsesProxyHandler") -> bool:
    """Attempt a WebSocket upgrade on the given BaseHTTPRequestHandler.

    Returns True if the upgrade was handled (caller should not process further),
    False if this was not a WebSocket upgrade request.
    """
    upgrade = (handler.headers.get("Upgrade", "") or "").lower()
    connection = (handler.headers.get("Connection", "") or "").lower()
    if "websocket" not in upgrade or "upgrade" not in connection:
        return False

    key = handler.headers.get("Sec-WebSocket-Key", "")
    if not key:
        handler.send_response(400)
        handler.end_headers()
        return True

    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
    ).decode("ascii")

    handler.send_response(101)
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    turn_state = (handler.headers.get("x-codex-turn-state", "") or "").strip()
    if turn_state:
        handler.send_header("x-codex-turn-state", turn_state)
    handler.end_headers()

    sock = handler.request
    handler.close_connection = True

    # Drain bytes that BufferedReader (rfile) read ahead beyond the HTTP
    # headers.  These belong to the WebSocket stream and must be fed to the
    # WebSocket connection before reading from the raw socket.
    initial_data = b""
    try:
        sock.setblocking(False)
        peeked = handler.rfile.peek(4096)
        if peeked:
            initial_data = handler.rfile.read(len(peeked))
    except (BlockingIOError, OSError, ValueError):
        pass
    finally:
        try:
            sock.setblocking(True)
        except OSError:
            pass

    # Prevent HTTPServer.shutdown_request() from closing the real socket after
    # do_GET returns.  BaseServer.process_request calls finish_request then
    # shutdown_request, which calls socket.shutdown(SHUT_WR) + socket.close().
    # We patch the server's shutdown_request to skip sockets that have been
    # handed off to WebSocket threads.
    _mark_socket_detached(handler.server, sock)

    def _run_ws() -> None:
        try:
            sock.settimeout(None)
            ws_conn = _WebSocketConnection(sock, initial_data)
            _handle_websocket_session_sync(ws_conn, proxy)
        except Exception as exc:
            proxy._log(f"responses websocket session error: {exc}")
        finally:
            try:
                sock.close()
            except OSError:
                pass

    thread = threading.Thread(target=_run_ws, daemon=True)
    thread.start()
    return True


def _mark_socket_detached(server, sock: socket.socket) -> None:
    """Mark a socket as detached so the server's shutdown_request skips it.

    Patches server.shutdown_request on first call to check a set of detached
    socket ids.  This prevents HTTPServer from closing sockets that have been
    handed off to WebSocket threads."""
    if not hasattr(server, "_ws_detached_ids"):
        server._ws_detached_ids = set()
        _orig_shutdown_request = server.shutdown_request

        def _patched_shutdown_request(request, *args, **kwargs):
            if id(request) in server._ws_detached_ids:
                server._ws_detached_ids.discard(id(request))
                return
            _orig_shutdown_request(request, *args, **kwargs)

        server.shutdown_request = _patched_shutdown_request
    server._ws_detached_ids.add(id(sock))


def _handle_websocket_session_sync(ws: _WebSocketConnection, proxy: "ResponsesProxyHandler") -> None:
    """Synchronous WebSocket session loop."""
    last_request: Optional[dict] = None
    last_response_output: list = []
    last_response_id = ""
    last_response_pending_tool_call_ids: list[str] = []
    turn_number = 0

    ws.start_keepalive()
    proxy._log("ws: session started")
    try:
        while True:
            message = ws.recv_message()
            if message is None:
                proxy._log("ws: client disconnected (recv_message returned None)")
                break
            if not isinstance(message, str):
                continue

            turn_number += 1
            try:
                raw = json.loads(message)
            except (TypeError, ValueError):
                proxy._log(f"ws: turn {turn_number} invalid JSON ({len(message)} bytes)")
                err = _build_websocket_error_payload(400, "invalid JSON")
                ws.send_text(json.dumps(err, separators=(",", ":"), ensure_ascii=False))
                continue

            req_type = _gget_str(raw, "type")
            prev_id = _gget_str(raw, "previous_response_id")
            input_items = _gget(raw, "input")
            input_count = len(input_items) if isinstance(input_items, list) else 0
            proxy._log(f"ws: turn {turn_number} request type={req_type} previous_response_id={prev_id or 'none'} input_items={input_count}")

            request_json, updated_last_request, error = normalize_websocket_request(
                raw, last_request, last_response_output, last_response_id, last_response_pending_tool_call_ids
            )
            if error:
                status, msg = error
                err = _build_websocket_error_payload(status, msg)
                ws.send_text(json.dumps(err, separators=(",", ":"), ensure_ascii=False))
                continue
            if request_json is None or updated_last_request is None:
                err = _build_websocket_error_payload(500, "internal normalization error")
                ws.send_text(json.dumps(err, separators=(",", ":"), ensure_ascii=False))
                continue

            last_request = updated_last_request

            completed = False
            completed_output: list = []
            completed_response_id = ""
            pending_tool_call_ids: set[str] = set()
            ws_state = OaiToResponsesState()
            turn_event_count = 0

            model_name = _gget_str(request_json, "model")
            chat_req = convert_responses_request_to_chat_completions(model_name, request_json, stream=True)
            proxy._log(f"ws: upstream request model={model_name} input_items={len(_gget(request_json, 'input') or [])}")

            try:
                for raw_line in upstream_chat_completions_stream(proxy.host, proxy.port, chat_req):
                    line = raw_line.rstrip(b"\r\n")
                    if not line:
                        continue
                    if line.startswith(b"data:"):
                        payload = line[len(b"data:"):].strip()
                    elif line.startswith(b"event:") or line.startswith(b":"):
                        continue
                    else:
                        payload = line.strip()
                    if not payload or payload == b"[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except (TypeError, ValueError):
                        continue
                    # Log upstream chunk summary for debugging
                    choices = _gget(chunk, "choices")
                    if isinstance(choices, list) and choices:
                        ch0 = choices[0]
                        delta = _gget(ch0, "delta")
                        has_tc = isinstance(_gget(delta, "tool_calls"), list)
                        has_text = bool(_gget_str(delta, "content"))
                        fr = _gget_str(ch0, "finish_reason")
                        if fr or has_tc:
                            proxy._log(f"ws: upstream chunk finish_reason={fr or 'none'} has_tool_calls={has_tc} has_text={has_text}")
                    events = convert_chat_stream_chunk_to_responses_events(ws_state, chunk, request_json)
                    for ev_bytes in events:
                        ev_obj = _sse_event_to_json(ev_bytes)
                        if ev_obj is None:
                            continue
                        event_type = _gget_str(ev_obj, "type")
                        turn_event_count += 1
                        if event_type == "response.output_item.done":
                            item = _gget(ev_obj, "item")
                            if isinstance(item, dict):
                                _record_pending_tool_call_ids(pending_tool_call_ids, item)
                                proxy._log(f"ws: output_item.done type={_gget_str(item, 'type')} call_id={_gget_str(item, 'call_id')}")
                        if _is_websocket_completion_event(event_type):
                            resp = _gget(ev_obj, "response")
                            if isinstance(resp, dict):
                                completed = True
                                completed_output = _gget(resp, "output") if isinstance(_gget(resp, "output"), list) else []
                                completed_response_id = _gget_str(ev_obj, "response.id").strip()
                                end_turn = _gget(resp, "end_turn")
                                output_types = [_gget_str(it, "type") for it in completed_output if isinstance(it, dict)]
                                proxy._log(f"ws: response.completed id={completed_response_id} end_turn={end_turn} output_types={output_types} pending_tools={sorted(pending_tool_call_ids)}")
                        _record_pending_tool_call_ids(pending_tool_call_ids, ev_obj)
                        ws.send_text(json.dumps(ev_obj, separators=(",", ":"), ensure_ascii=False))
                        if event_type == _WS_EVENT_TYPE_ERROR:
                            break
                if not completed and ws_state.completion_pending and not ws_state.completed_emitted:
                    ws_state.completed_emitted = True
                    final_ev = _build_completed_event(ws_state, request_json)
                    final_obj = _sse_event_to_json(final_ev)
                    if final_obj is not None:
                        ws.send_text(json.dumps(final_obj, separators=(",", ":"), ensure_ascii=False))
                        completed = True
                        resp = _gget(final_obj, "response")
                        completed_output = _gget(resp, "output") if isinstance(_gget(resp, "output"), list) else []
                        completed_response_id = _gget_str(final_obj, "response.id").strip()
                        end_turn = _gget(resp, "end_turn")
                        output_types = [_gget_str(it, "type") for it in completed_output if isinstance(it, dict)]
                        proxy._log(f"ws: response.completed(built) id={completed_response_id} end_turn={end_turn} output_types={output_types} finish_reason={ws_state.finish_reason} has_tool_calls={ws_state.has_tool_calls}")
                if not completed:
                    proxy._log(f"ws: stream closed before response.completed (events={turn_event_count} finish_reason={ws_state.finish_reason} has_tool_calls={ws_state.has_tool_calls})")
                    err = _build_websocket_error_payload(408, "stream closed before response.completed")
                    ws.send_text(json.dumps(err, separators=(",", ":"), ensure_ascii=False))
                else:
                    last_response_output = completed_output
                    last_response_id = completed_response_id
                    last_response_pending_tool_call_ids = sorted(pending_tool_call_ids)
            except UpstreamError as exc:
                proxy._log(f"ws: upstream error status={exc.status} msg={exc.message[:200]}")
                err = _build_websocket_error_payload(exc.status, exc.message)
                ws.send_text(json.dumps(err, separators=(",", ":"), ensure_ascii=False))
            except (socket.error, OSError) as exc:
                proxy._log(f"ws: socket error: {exc}")
                err = _build_websocket_error_payload(502, f"upstream connection failed: {exc}")
                ws.send_text(json.dumps(err, separators=(",", ":"), ensure_ascii=False))
            except Exception as exc:
                proxy._log(f"ws: unexpected error: {exc}")
                err = _build_websocket_error_payload(500, str(exc))
                ws.send_text(json.dumps(err, separators=(",", ":"), ensure_ascii=False))
    finally:
        ws.stop_keepalive()
