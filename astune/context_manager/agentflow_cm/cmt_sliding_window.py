from typing import List, Callable, Tuple
from beast_logger import print_listofdict
from astune.context_manager.agentflow_cm.cmt_linear_think import ExtendedMessage, MultiSampleCMT
from loguru import logger

"""
滑窗context管理器
- 当context超出最大长度时，开始新的滑窗
- 新的滑窗保留初始化信息、最新的env和llm信息
- 其他信息忽略，生成一条“[Previous {x} conversation has been omitted for brevity.]”信息
"""


class SlidingWindowCMT(MultiSampleCMT):
    """
    A non-linear context manager template that handles the conversation flow between LLM and environment.
    """

    def __init__(self, config, tokenizer, llm_chat_fn):
        self.llm_chat_fn = llm_chat_fn
        self.latest_env_response_id = ""
        self.latest_env_response_content = ""
        self.console_debug_mode = False
        self.force_think = config.astune.rollout.force_think
        self.env_cnt = 0
        self.llm_cnt = 0
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []
        self.current_context_status = ""
        max_response_length = self.config.astune.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.astune.rollout.max_model_len
        self.max_seq_length: int = max_model_len - max_response_length
        self.max_env_output_length: int = self.config.astune.rollout.max_env_len
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")
        self.terminal_rewards_dict = {}
        self.latest_llm_interaction_socket: List[ExtendedMessage] = None
        self.grouped_steps: List[List[ExtendedMessage]] = []
        self.discarded = False
        self.is_terminated = False
        self.context_time_cost = 0
        self.generated_token_cnt = 0
        self.omitted_msg_so_far = 0
        self.prompt_part_token_overflow = False
        self.already_mad_flag = False
        self.round_cnt = 0

    def prepare_next_llm_context(self):
        """Prepare the next LLM context with sliding window logic. When the token length exceeds the maximum limit, start a new sliding window.
        """
        self.latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "llm", "llm(do_not_train)", "env", "memory"])
        dict_context = self.to_role_content(self.latest_llm_interaction_socket)

        # if token overflow, begin new sliding window
        cur_seq_len = self._get_seq_length(dict_context)
        # print(f"cur_seq_len {cur_seq_len}, self.max_seq_length {self.max_seq_length}")

        is_safe: bool = cur_seq_len < self.max_seq_length
        if not is_safe:
            _, previous_interaction = self._prepare_next_llm_context_static()
            self.begin_new_sliding_window(previous_interaction=previous_interaction)

            dict_context, self.latest_llm_interaction_socket = self._prepare_next_llm_context_static()
            cur_seq_len = self._get_seq_length(dict_context)
            if cur_seq_len > self.config.astune.data.max_prompt_length:
                print(f"Warning! cur_seq_len={cur_seq_len} immediately after new sliding window is created")
                print_listofdict(
                    dict_context, mod='env_clip'
                )
                self.prompt_part_token_overflow = True

        return dict_context


    def _prepare_next_llm_context_static(self):
        """Fetch from context and convert to dict format.
        """
        latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "llm", "llm(do_not_train)", "env", "memory"])
        dict_context = self.to_role_content(latest_llm_interaction_socket)
        return dict_context, latest_llm_interaction_socket


    def check_context_token_num_safe(self, messages: List[dict], tools=[]) -> Tuple[bool, str]:
        """Always be safe because we already check in `prepare_next_llm_context`
        """
        if self.already_mad_flag and self.config.astune.rollout.agent_madness_termination:
            return False, "already_mad"

        assert self._get_seq_length(messages, tools) < self.max_seq_length

        if self.prompt_part_token_overflow:
            return False, "prompt_part_token_overflow"
        else:
            return True, "safe"


    def begin_new_sliding_window(self, previous_interaction):
        """Begin a new sliding window by preserving initialization, latest env and llm messages, and summarizing the rest into a memory message.
        """
        self.grouped_steps += [previous_interaction]
        recall_x_action = 2
        # delete most `llm` and `env` messages, keep only the last 2 of each
        preserve_messages = self.filter_context_via_authors_with_limit(
            authors = ["initialization", "llm", "env", "memory"],
            limit={
                "llm": f"keep_last@{recall_x_action}",
                "env": f"keep_last@{recall_x_action+1}",
                "memory": "keep_last@1",
            }
        )
        other_messages = [ext_msg for ext_msg in self.filter_context_via_authors(authors = ["initialization", "llm", "env", "memory"]) if ext_msg not in preserve_messages]
        # TODO: find a way to summarize previous messages
        self.omitted_msg_so_far += len(other_messages)
        # init message in `preserve_messages`
        init_message_in_preserve_messages = [msg for msg in preserve_messages if msg.author == "initialization"]
        # create memory message
        other_messages = init_message_in_preserve_messages + other_messages # include init when create memory
        memory_msg = self.create_memory_message(other_messages)
        # inseart `preserve_messages` after initialization
        new_context_beginning = init_message_in_preserve_messages + [memory_msg] + [msg for msg in preserve_messages if msg.author != "initialization"]
        # disable llm training for all message in `new_context_beginning`
        for i in range(len(new_context_beginning)):
            ext_msg = new_context_beginning[i]
            if ext_msg.author == 'llm':
                author_override = "llm(do_not_train)"
                new_context_beginning[i] = ExtendedMessage(
                    author=author_override,
                    role=ext_msg.role,
                    content=ext_msg.content_for_future,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                )
        self.full_context = new_context_beginning
        # delete old memory message
        self.full_context = self.filter_context_via_authors_with_limit(
            authors = ["initialization", "llm", "llm(do_not_train)", "env", "memory"],
            limit={
                "memory": "keep_last@1",
            }
        )

    def create_memory_message(self, msg_list: List[ExtendedMessage]) -> ExtendedMessage:
        """TODO: create a better summary message
        """
        x = self.omitted_msg_so_far // 2
        enable_llm_memory_extraction = self.config.astune.context_manager.sliding_window_cm
        if not enable_llm_memory_extraction:
            return ExtendedMessage(
                author="memory",
                role="user",
                content=f"[Previous {x} round of conversations have been omitted for brevity.]",
                token_generator='auto',
                tokenizer=self.tokenizer,
            )
        else:
            return ExtendedMessage(
                author="memory",
                role="user",
                content=self.llm_memory_extraction(msg_list),
                token_generator='auto',
                tokenizer=self.tokenizer,
            )

    def ensure_terminate_rollout_stage(self):
        previous_interaction_dict_context, previous_interaction = self._prepare_next_llm_context_static()
        if any([ext_msg.need_training for ext_msg in previous_interaction]):
            self.grouped_steps += [previous_interaction]


    def save_env_output(self, env_output, input_msg_ref = None, add_nothink=False):
        self.env_cnt += 1
        env_output['content'] = f"[Current Env Step {self.env_cnt}]\n\n" + env_output['content']
        return super().save_env_output(env_output, input_msg_ref, add_nothink)


    def save_llm_output(self, llm_output, input_msg_ref, auto_register_full_context=True):
        self.llm_cnt += 1
        return super().save_llm_output(llm_output, input_msg_ref)


    def llm_memory_extraction(self, msg_list: List[ExtendedMessage]) -> str:
        """Use LLM to extract memory from previous messages.
        """
        from astune.context_manager.agentflow_cm.cmt_foreign_llm import construct_alien_llm_chat_fn
        from textwrap import dedent
        self.alien_llm_chat_fn: Callable = construct_alien_llm_chat_fn(self.config, self.config.actor_rollout_ref.rollout)
        messages = self.to_role_content(msg_list)
        messages.append({
            "role": "user",
            "content": dedent("""
                New task: Summarize the previous attempts into a concise memory statement that captures the key points and context.
                - Start with: Previously, X attempts have been made, in these attempts, ...
                - Focus on the main events, actions, and outcomes.
                - If there are big or repeated failures, try to find reason and provide some future advice.
            """)
        })

        try:
            llm_output = self.alien_llm_chat_fn(messages, request_id="")
        except Exception as e:
            logger.bind(exception=True).exception(f"call alien_llm_chat_fn error with {e}")
            x = self.omitted_msg_so_far // 2
            llm_output_content = f"[Previous {x} round of conversations have been omitted for brevity.]"
            return llm_output_content
        return llm_output['content']
