
### case 1

```python
pprint(messages)
[{'role': 'assistant',
  'tool_calls': [{'function': {'arguments': {'location': 'Paris, France',
                                             'unit': 'celsius'},
                               'name': 'get_current_temperature'},
                  'type': 'function'}]}]
```

---------------------------------------------------------------------------------------------------
print(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
```python
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
</tool_call><|im_end|>
```


### case 2

```python
pprint(messages)
[{'content': 'THIS IS CONTENT',
  'role': 'assistant',
  'tool_calls': [{'function': {'arguments': {'location': 'Paris, France',
                                             'unit': 'celsius'},
                               'name': 'get_current_temperature'},
                  'type': 'function'}]}]
```

---------------------------------------------------------------------------------------------------
print(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
```python
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>assistant
THIS IS CONTENT
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
</tool_call><|im_end|>
```

### case 3

```python
pprint(messages)
[{'content': 'THIS IS CONTENT',
  'role': 'assistant',
  'tool_calls': [{'function': {'arguments': {'location': 'Paris, France',
                                             'unit': 'celsius'},
                               'name': 'get_current_temperature'},
                  'type': 'function'}]}]
```

```python
print(tools)
[
    {'type': 'function', 'function': {'name': 'execute_python_code', 'parameters': {'properties': {'code': {'description': 'The Python code to be executed.', 'type': 'string'}, 'timeout': {'default': 300, 'description': 'The maximum time (in seconds) allowed for the code to run.', 'type': 'number'}}, 'required': ['code'], 'type': 'object'}, 'description': 'Execute the given python code in a temp file and capture the return\n\ncode, standard output and error. Note you must `print` the output to get\nthe result, and the tmp file will be removed right after the execution.'}},
    {'type': 'function', 'function': {'name': 'generate_response', 'parameters': {'properties': {'response': {'description': 'Your response to the user.', 'type': 'string'}, 'result': {'description': 'Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}', 'type': 'string'}}, 'required': ['response', 'result'], 'type': 'object'}, 'description': 'Generate a response. Note only the input argument `response` is\n\nvisible to the others, you should include all the necessary\ninformation in the `response` argument.'}}
]
```
---------------------------------------------------------------------------------------------------

```python
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "execute_python_code", "parameters": {"properties": {"code": {"description": "The Python code to be executed.", "type": "string"}, "timeout": {"default": 300, "description": "The maximum time (in seconds) allowed for the code to run.", "type": "number"}}, "required": ["code"], "type": "object"}, "description": "Execute the given python code in a temp file and capture the return\n\ncode, standard output and error. Note you must `print` the output to get\nthe result, and the tmp file will be removed right after the execution."}}
{"type": "function", "function": {"name": "generate_response", "parameters": {"properties": {"response": {"description": "Your response to the user.", "type": "string"}, "result": {"description": "Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}", "type": "string"}}, "required": ["response", "result"], "type": "object"}, "description": "Generate a response. Note only the input argument `response` is\n\nvisible to the others, you should include all the necessary\ninformation in the `response` argument."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>assistant
THIS IS CONTENT
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
</tool_call><|im_end|>
```

###

/mnt/data_cpfs/fuqingxu/astune/.venv/bin/python -m vllm.entrypoints.cli.main serve /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct --tensor-parallel-size 4 --dtype auto --enforce-eager --gpu-memory-utilization 0.9 --disable-custom-all-reduce --max-num-seqs 10 --max-model-len 18000 --load-format auto --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching --seed 12345 --port 18000

```
<|im_start|>system

You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}.


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "execute_python_code", "description": "Execute the given python code in a temp file and capture the return\
\
code, standard output and error. Note you must `print` the output to get\
the result, and the tmp file will be removed right after the execution.", "parameters": {"properties": {"code": {"description": "The Python code to be executed.", "type": "string"}, "timeout": {"default": 300, "description": "The maximum time (in seconds) allowed for the code to run.", "type": "number"}}, "required": ["code"], "type": "object"}}}
{"type": "function", "function": {"name": "generate_response", "description": "Generate a response. Note only the input argument `response` is\
\
visible to the others, you should include all the necessary\
information in the `response` argument.", "parameters": {"properties": {"response": {"description": "Your response to the user.", "type": "string"}, "result": {"description": "Your solution of the given math problem. Put your final answer in boxed format, e.g., \\\\boxed{42}", "type": "string"}}, "required": ["response", "result"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user

Natalia sold clips to 48 of her friends in  April,  and  then  she
sold half as many clips in May. How many clips  did  Natalia  sell
altogether in April and May?
<|im_end|>
<|im_start|>assistant
```