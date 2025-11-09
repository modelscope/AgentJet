import copy
from textwrap import dedent
from typing import List, Tuple
from astune.schema.trajectory import Sample, Reward
from astune.context_manager.cmt_linear import ExtendedMessage, CMTLinear
from beast_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem

class MultiSampleCMT(CMTLinear):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []
        self.current_context_status = ""
        max_response_length = self.config.astune.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.astune.rollout.max_model_len

        assert self.config.astune.data.max_response_length < self.config.astune.data.max_prompt_length, "think linear template requires a big max_prompt_length"

        self.max_seq_length: int = max_model_len - max_response_length
        assert self.max_seq_length <= self.config.astune.data.max_prompt_length, "max_seq_length should be less than or equal to max_prompt_length"


        self.max_env_output_length: int = self.config.astune.rollout.max_env_len
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")

        self.terminal_rewards_dict = {}
        self.latest_llm_interaction_socket: List[ExtendedMessage] = None
        self.grouped_steps: List[List[ExtendedMessage]] = []

        self.discarded = False
        self.is_terminated = False
        self.context_time_cost = 0
        self.already_mad_flag = False

        self.force_think = config.astune.rollout.force_think
        self.env_action_preference = config.astune.task_reader.env_service.env_action_preference
        if not self.force_think:
            # think_hint_for_qwen3 =
            self.think_hint: str = "\n\nThink about the next step before answering. Your thought (<think>...</think>) should be as short and concise as possible."
        else:
            if self.env_action_preference == "box":
                force_think_prompt = dedent("""
                    Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce your answer with \\box{}.
                    For example:
                    <think>...your thinking process...</think>
                    \\box{...your final answer...}
                """)
            elif self.env_action_preference == "code":
                force_think_prompt = dedent("""
                    Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce the next-step action.
                    For example:
                    <think>...your thinking process...</think>
                    ```python
                    # your action here
                    ```
                """)
            else:
                raise ValueError(f"Unsupported env_action_preference: {self.env_action_preference}")
            # think_hint_for_qwen2 =
            self.think_hint: str = force_think_prompt

    def _get_seq_length(self, messages: List[dict]) -> int:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])

    def check_context_token_num_safe(self, messages: List[dict]) -> Tuple[bool, str]:
        if self.already_mad_flag and self.config.astune.rollout.agent_madness_termination:
            return False, "already_mad"
        if self._get_seq_length(messages) < self.max_seq_length:   # self.config.env_engine.max_seq_length = 20480
            return True, "safe"
        else:
            return False, "token_overflow"

    @property
    def steps(self):
        # TODO: need revise
        return self.prepare_previous_context(mod='future')

    def generate_log(self, task_id = None, global_step="NA"):
        task_id = self.task_id
        nested_items_print_buffer = {}
        for index, ext_steps in enumerate(self.grouped_steps):
            cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps, index=index, total_steps=len(self.grouped_steps))
            text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
            input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
            loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
            buffer = {
                "text_arr": text_arr,
                "input_id_arr": input_id_arr,
                "loss_mask_color_arr": loss_mask_color_arr,
            }
            raw_reward = self.reward_structure.raw_reward
            step_reward:float = self.reward_structure.step_reward[index]
            try:
                step_advantage = self.reward_structure.step_advantage[index]
                step_advantage_simple = self.reward_structure.step_advantage_simple[index]
            except:
                step_advantage = 0.0
                step_advantage_simple = 0.0
            task_outcome = str(self.reward_structure.success_rate)
            selectors = [task_id, task_outcome, str(index)]
            len_prompt_ids = len(cmt_tokenized["prompt_ids"])
            len_response_ids = len(cmt_tokenized["response_ids"])
            len_input_ids = len(cmt_tokenized["input_ids"])
            assert len_prompt_ids + len_response_ids == len_input_ids, "len_prompt_ids + len_response_ids should equal to len_input_ids"
            # print(f"Task {task_id}, outcome {task_outcome}, group {index}, len_prompt_ids {len_prompt_ids}, len_response_ids {len_response_ids}, len_input_ids {len_input_ids}")
            nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
                item_id=f"item",
                outcome=task_outcome,
                len_prompt_ids=len_prompt_ids,
                len_response_ids=len_response_ids,
                len_input_ids=len_input_ids,
                raw_reward=f"{float(raw_reward):.3f}",
                step_reward=f"{float(step_reward):.3f}",
                step_advantage=f"{float(step_advantage):.3f}",
                step_advantage_simple=f"{float(step_advantage_simple):.3f}",
                content=SeqItem(
                    text = buffer['text_arr'],  # 文本
                    title = buffer['text_arr'], # 鼠标悬浮文本
                    count = buffer['input_id_arr'], # 高亮文本
                    color = buffer['loss_mask_color_arr']   # 颜色
                )
            )
        print_nested(nested_items_print_buffer,
            main_content="This is the main content of the nested JSON",
            header=f"[{global_step}] Task {task_id} (Reward {float(step_reward):.3f})",
            mod="rollout",
            narrow=False,
            attach="copy this"
        )


    def group_tokenize(self):
        return self.group_tokenize_multi_group()


    def process_reward(self, reward_structure: Reward):
        # lienar 模式有多条轨迹
        use_step_reward_from_env = self.config.astune.rollout.get("use_step_reward_from_env", False)
        if not use_step_reward_from_env:
            self.reward_structure = reward_structure
            self.reward_structure.step_reward = [0.0 for _ in range(len(self.grouped_steps))]
            for index, ext_steps in enumerate(self.grouped_steps):
                self.reward_structure.step_reward[index] = self.compute_step_level_reward(
                    ext_steps=ext_steps,
                    index=index,
                    total_steps=len(self.grouped_steps)
                )
        else:
            step_reward = reward_structure.raw_step_reward
            assert reward_structure.raw_step_reward
            assert len(reward_structure.raw_step_reward) == len(self.grouped_steps), f"len(reward_structure.raw_step_reward) {len(reward_structure.raw_step_reward)} should equal to len(self.grouped_steps) {len(self.grouped_steps)}"
            self.reward_structure = reward_structure
            self.reward_structure.step_reward = reward_structure.raw_step_reward

    def compute_step_level_reward(self, ext_steps: List[ExtendedMessage], index: int, total_steps:int)->float:
        assert self.reward_structure is not None

        # --------------- global level reward ---------------
        global_reward = self.reward_structure.raw_reward
        gamma = self.config.astune.rollout.gamma
        step_reward_base = global_reward * (gamma ** (total_steps - index - 1))

        # --------------- compute step level reward ---------------
        step_reward = step_reward_base
        if self.already_mad_flag:
            step_reward = self.config.astune.rollout.agent_madness_reward
            self.reward_structure.madness = -1.0

        return step_reward



class LinearThinkCMT(MultiSampleCMT):
    """
    A linear context manager template that handles the conversation flow between LLM and environment.
    This class manages the context window, tokenization, and message history in a linear fashion.

    Attributes:
        config: Configuration object containing environment and model settings
        tokenizer: Tokenizer instance for processing text
        full_context (List[ExtendedMessage]): List of all messages in the conversation
        current_context_status (str): Current status of the context
        max_seq_length (int): Maximum sequence length for the context window
        max_env_output_length (int): Maximum length for environment outputs
        terminal_rewards_dict (dict): Dictionary storing terminal rewards

    """





    def prepare_next_llm_context(self):
        self.latest_llm_interaction_socket = []
        # 筛选出 `初始message-user-llm-user-llm`` 或者 `初始message-llm-user-llm-user``
        self.latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "llm", "env"])

        for index, ext_msg in enumerate(list(self.latest_llm_interaction_socket)):
            # is_last 是最后一条信息
            # remove history llm author's think (and add /no_think tag to every but last message)
            is_last = (index == len(self.latest_llm_interaction_socket) - 1)
            # 根据消息类型进行处理
            if ext_msg.author == "llm":
                # 如果是以往的llm消息，去掉think标签
                import re
                new_ext_msg_content = re.sub(r'<think>.*?</think>', '', ext_msg.content, flags=re.DOTALL).strip()
                new_ext_msg_content = new_ext_msg_content.replace("<think>", "")
                new_ext_msg_content = new_ext_msg_content.replace("</think>", "")
                # new_ext_msg_content = re.sub(r'<think>.*?</think>', '<think>\n\n</think>', ext_msg.content, flags=re.DOTALL)

                if self.config.astune.context_manager.linear_think_cm.train_history_infer_token:
                    assert ext_msg.author == "llm"
                    self.latest_llm_interaction_socket[index] = ExtendedMessage(
                        author=ext_msg.author,
                        role=ext_msg.role,
                        content=new_ext_msg_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
                else:
                    assert ext_msg.author == "llm"
                    author_override = "llm(do_not_train)"
                    self.latest_llm_interaction_socket[index] = ExtendedMessage(
                        author=author_override,
                        role=ext_msg.role,
                        content=new_ext_msg_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
            elif ext_msg.author in ["env", "initialization"]:
                if self.config.astune.context_manager.linear_think_cm.train_history_infer_token:
                    # 如果是初始化或者环境反馈，都加上 /no_think 标签
                    if not is_last:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + "\n/no_think",
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                    else:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + self.think_hint,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                else:
                    # 如果是初始化或者环境反馈
                    if not is_last:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                    else:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + self.think_hint,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
            else:
                raise RuntimeError(f"Unknown author {ext_msg.author} in latest_llm_interaction_socket")

        dict_context = self.to_role_content(self.latest_llm_interaction_socket)
        return dict_context



    def save_llm_output(self, llm_output, input_msg_ref):
        ext_msg = super().save_llm_output(llm_output, input_msg_ref)
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])
        self.grouped_steps += [this_interaction]
        self.latest_llm_interaction_socket = []
        return ext_msg


    def save_env_output(self, env_output:dict, input_msg_ref:List[dict]=None, add_nothink=False):
        super().save_env_output(env_output, input_msg_ref, add_nothink)
        return


    def prepare_world_interaction(self) -> str:
        latest_content = self.full_context[-1].content
        if self.config.astune.context_manager.linear_think_cm.remove_think_before_submit_as_action:
            import re
            new_ext_msg_content = re.sub(r'<think>.*?</think>', '', latest_content, flags=re.DOTALL).strip()
            new_ext_msg_content = new_ext_msg_content.replace("<think>", "")
            new_ext_msg_content = new_ext_msg_content.replace("</think>", "")
            latest_content = new_ext_msg_content.strip()
        if self.config.astune.context_manager.linear_think_cm.extract_box_before_submit_as_action:
            # take content within \box
            # 提取 \box 中的内容
            import re
            box_pattern = r'\\box\{(.*?)\}'
            match = re.search(box_pattern, latest_content, re.DOTALL)
            if match:
                latest_content = match.group(1).strip()
            else:
                # 如果没有找到 \box，选择保留原内容
                pass
        return latest_content


