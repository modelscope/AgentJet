import asyncio
import concurrent.futures
import copy
from typing import Any, Dict, List


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


def run_async_coroutine_with_timeout(coro, timeout: int = 3600) -> Any:
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    if not in_loop:
        final_res = asyncio.run(coro)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            try:
                final_res = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise
            except Exception:
                raise
    return final_res


def remove_fields(d: Dict, fields: List[str]) -> Dict:
    d = copy.deepcopy(d)
    for field in fields:
        d.pop(field.strip(), None)
    return d
