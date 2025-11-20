from typing import Any, List, Dict
import asyncio
import copy


# apply chat_template to a message, and then convert back to message
def convert_tool_to_user_message(tool_message, tokenizer, format="qwen"):
    assert format == "qwen"

    if tool_message["role"] == "user":
        return tool_message
    elif tool_message["role"] == "tool" and len(tool_message["tool_calls"]) > 0:
        assert len(tool_message["tool_calls"]) == 1
        return {
            "role": "user",
            "content": str(tool_message["tool_calls"][0]["result"]),
        }


def run_async_coro__no_matter_what(coro):
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    if not in_loop:
        final_res = asyncio.run(coro)
    else:
        import threading

        _res_holder = {}
        _exc_holder = {}

        def _run():
            try:
                _res_holder["res"] = asyncio.run(coro)
            except Exception as _e:
                _exc_holder["exc"] = _e

        _t = threading.Thread(target=_run, daemon=True)
        _t.start()
        _t.join()
        if "exc" in _exc_holder:
            raise _exc_holder["exc"]
        final_res = _res_holder["res"]
    return final_res


def remove_fields(d: Dict, fields: List[str]) -> Dict:
    d = copy.deepcopy(d)
    for field in fields:
        d.pop(field.strip(), None)
    return d
