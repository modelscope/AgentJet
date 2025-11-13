from typing import List, Type
from astune.context_manager.cmt_linear import CMTLinear, ExtendedMessage, replace_token_ids
from beast_logger import print_nested, print_listofdict, NestedJsonItem, SeqItem



def can_merge_steps(source_step: List[ExtendedMessage], target_step: List[ExtendedMessage], debug=False) -> bool:
    # if `source_step` has more messages than `target_step`
    # and if `source_step` and `target_step` share same token_arr in [0:len(target_step)]
    # even if the authors are different, we can still merge them
    can_merge = False
    # compare_level = 'token' # strict compare with token ids
    compare_level = 'text' # relaxed compare with text, more easier to match, at very little cost
    if len(source_step) >= len(target_step):
        all_msg_match = True
        for i in range(len(target_step)):
            if compare_level == 'text':
                same = source_step[i].content_for_future == target_step[i].content_for_future
            elif compare_level == 'token':
                same = source_step[i].token_arr == target_step[i].token_arr
            else:
                raise NotImplementedError
            if not same:
                all_msg_match = False
                break
        if all_msg_match:
            can_merge = True

    if debug:
        debug_listofdict = []
        if len(source_step) >= len(target_step):
            all_msg_match = False
            for i in range(len(target_step)):
                d = {}
                d['source'] = source_step[i].content_for_future
                d['target'] = target_step[i].content_for_future
                if compare_level == 'text':
                    same = source_step[i].content_for_future == target_step[i].content_for_future
                elif compare_level == 'token':
                    same = source_step[i].token_arr == target_step[i].token_arr
                else:
                    raise NotImplementedError
                if not same:
                    d['match'] = 'NO'
                else:
                    d['match'] = 'YES'
                debug_listofdict.append(d)
        print_listofdict(debug_listofdict, header=f"can_merge_steps debug: {can_merge}")

    return can_merge