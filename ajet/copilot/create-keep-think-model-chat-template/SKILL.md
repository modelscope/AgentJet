

Your task is to investigate the chat template of given model, go to its tokenizer config and check whether the following behavior exists:

>
> Remove history <think> block from the input when apply chat template when converting messages.
>

This behavior will make RL training slower, if this behavior exists, please change the chat template to forbid such behavior.

You must not do this in-place, instead, please create another model.
E.g., "/mnt/data_cpfs/xielipeng.xlp/models/Qwen3-8B" -> "/mnt/data_cpfs/xielipeng.xlp/models/Qwen3-8B-Keep-History"
For all files within the original model path, please create symbolic links instead of copying files.
With only one exception, the tokenizer config file, which should be copied and modified to change the chat template.
