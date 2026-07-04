import re

def strip_tag_blocks(s: str, tag: str = "think") -> str:
    # remove all <think>...</think>, even if nested; case-insensitive
    pat = re.compile(rf"(?is)(<\s*{tag}\b[^>]*>)|(</\s*{tag}\s*>)")
    out, depth, last = [], 0, 0

    for m in pat.finditer(s):
        open_tag, close_tag = m.group(1), m.group(2)
        if open_tag:
            if depth == 0:
                out.append(s[last:m.start()])  # keep text before this block
            depth += 1
        else:  # close_tag
            if depth > 0:
                depth -= 1
                if depth == 0:
                    last = m.end()  # drop the whole block
    out.append(s[last:])
    return "".join(out)

def after_last_think_close(s: str) -> str:
    # matches </think>, allowing spaces/newlines: </  think   >
    pat = re.compile(r"</\s*think\s*>", re.IGNORECASE)
    last = None
    for m in pat.finditer(s):
        last = m
    return s[last.end():].strip().strip('\n') if last else s


def parse_reasoning_model_answer(model_answer: str) -> str:
    model_answer = after_last_think_close(model_answer)

    answer_match = re.findall(r"<answer>\s*(.*?)\s*</answer>", model_answer, re.DOTALL | re.IGNORECASE)
    boxed_answer_match = re.findall(r"\\+boxed\s*\{\s*([^{}]*?)\s*\}", model_answer, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match[-1].strip().strip("\n")
    elif boxed_answer_match:
        return boxed_answer_match[-1].strip().strip("\n")
    else:
        return model_answer
