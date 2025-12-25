from typing import List

from beast_logger import print_listofdict

from astuner.context_tracker.basic_tracker import ExtendedMessage


def is_timeline_mergeable(
    source_timeline: List[ExtendedMessage],
    target_timeline: List[ExtendedMessage],
    debug=False,
    compare_level="text",  # "text" or "token"
) -> bool:
    # compare_level = "text"  # relaxed compare with text, more easier to match, at very little cost
    # compare_level = "token" # strict compare with token, cause less aggressive merging

    can_merge = False
    if len(source_timeline) >= len(target_timeline):
        all_msg_match = True
        for i in range(len(target_timeline)):
            if compare_level == "text":
                same = (
                    # TODO: currently does not consider tools, may aggressively merge responses with different tool calls
                    # TODO: maybe we should give option for user to choose merging behavior
                    source_timeline[i].content_for_future
                    == target_timeline[i].content_for_future
                )
            elif compare_level == "token":
                same = source_timeline[i].token_arr == target_timeline[i].token_arr
            else:
                raise NotImplementedError
            if not same:
                all_msg_match = False
                break
        if all_msg_match:
            can_merge = True

    # developer only: code below is only for debugging (print a nice comparison table)
    if debug:
        debug_listofdict = []
        if len(source_timeline) >= len(target_timeline):
            all_msg_match = False
            for i in range(len(target_timeline)):
                d = {}
                d["source"] = source_timeline[i].content_for_future
                d["target"] = target_timeline[i].content_for_future
                if compare_level == "text":
                    same = (
                        source_timeline[i].content_for_future
                        == target_timeline[i].content_for_future
                    )
                elif compare_level == "token":
                    same = source_timeline[i].token_arr == target_timeline[i].token_arr
                else:
                    raise NotImplementedError
                if not same:
                    d["match"] = "NO"
                else:
                    d["match"] = "YES"
                debug_listofdict.append(d)
        print_listofdict(debug_listofdict, header=f"is_timeline_mergeable debug: {can_merge}")

    return can_merge


def toggle_author_and_mask(
    source_timeline: List[ExtendedMessage],
    target_timeline: List[ExtendedMessage],
) -> List[ExtendedMessage]:
    # if any message in `target_timeline` is author == 'llm',
    # but same-index message in `source_timeline` is author != 'llm'
    # change source_timeline's message author to 'llm'
    # overwrite `token_arr` and `token_logprob_arr` accordingly
    for i in range(len(target_timeline)):
        if target_timeline[i].author == "llm" and source_timeline[i].author != "llm":
            source_timeline[i].author = target_timeline[i].author
            source_timeline[i].token_arr = target_timeline[i].token_arr
            source_timeline[i].token_logprob_arr = target_timeline[i].token_logprob_arr
            assert source_timeline[i].need_training
    return source_timeline


def merge_tracker_timelines(
    timelines: List[List[ExtendedMessage]],
) -> List[List[ExtendedMessage]]:
    """Merge multiple timelines by absorbing those that can be merged.
    > Input:  a list of timelines. (a timeline means: List[ExtendedMessage])
    > Output: a shorter list of timelines after merging
    """
    absorbed_step_indices = []
    reversed_timelines = list(reversed(timelines))
    for i in range(len(reversed_timelines)):
        if i in absorbed_step_indices:
            continue
        # check whether [i, i+1, ..., ..., len(reversed_timelines)-1] can be merged
        for j in range(i + 1, len(reversed_timelines)):
            if j in absorbed_step_indices:
                continue
            source_timeline = reversed_timelines[i]
            target_timeline = reversed_timelines[j]
            if is_timeline_mergeable(source_timeline, target_timeline):
                source_timeline = toggle_author_and_mask(source_timeline, target_timeline)
                reversed_timelines[i] = source_timeline
                absorbed_step_indices += [j]

    # reverse back and exclude absorbed steps
    reversed_timelines_clean = []
    for i in range(len(reversed_timelines)):
        if i not in absorbed_step_indices:
            reversed_timelines_clean.append(reversed_timelines[i])
    timelines = list(reversed(reversed_timelines_clean))

    return timelines
