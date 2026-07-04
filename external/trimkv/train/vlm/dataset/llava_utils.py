# following https://github.com/zjysteven/lmms-finetune/blob/main/collators/chat_template_monkey_patch.py
import torch
import re
from inspect import isfunction
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from pathlib import Path
import PIL

from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import TensorType, get_json_schema, logging
from transformers.utils.chat_template_utils import _compile_jinja_template, _render_with_assistant_indices
from transformers.models.llava.processing_llava import LlavaProcessorKwargs


logger = logging.get_logger(__name__)

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def llava_apply_chat_template(
    self,
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    tools: Optional[List[Dict]] = None,
    documents: Optional[List[Dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
    tokenize: bool = True,
    padding: bool = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_dict: bool = False,
    return_assistant_tokens_mask: bool = False,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[str, List[int], List[str], List[List[int]], BatchEncoding]:
    """
    Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
    ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
    determine the format and control tokens to use when converting.

    Args:
        conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
            with "role" and "content" keys, representing the chat history so far.
        tools (`List[Dict]`, *optional*):
            A list of tools (callable functions) that will be accessible to the model. If the template does not
            support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
            giving the name, description and argument types for the tool. See our
            [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
            for more information.
        documents (`List[Dict[str, str]]`, *optional*):
            A list of dicts representing documents that will be accessible to the model if it is performing RAG
            (retrieval-augmented generation). If the template does not support RAG, this argument will have no
            effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
            see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
            for examples of passing documents with chat templates.
        chat_template (`str`, *optional*):
            A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
            argument, as the model's template will be used by default.
        add_generation_prompt (bool, *optional*):
            If this is set, a prompt with the token(s) that indicate
            the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
            Note that this argument will be passed to the chat template, and so it must be supported in the
            template for this argument to have any effect.
        continue_final_message (bool, *optional*):
            If this is set, the chat will be formatted so that the final
            message in the chat is open-ended, without any EOS tokens. The model will continue this message
            rather than starting a new one. This allows you to "prefill" part of
            the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
        tokenize (`bool`, defaults to `True`):
            Whether to tokenize the output. If `False`, the output will be a string.
        padding (`bool`, defaults to `False`):
            Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
        truncation (`bool`, defaults to `False`):
            Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
        max_length (`int`, *optional*):
            Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
            not specified, the tokenizer's `max_length` attribute will be used as a default.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
            values are:
            - `'tf'`: Return TensorFlow `tf.Tensor` objects.
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
            - `'jax'`: Return JAX `jnp.ndarray` objects.
        return_dict (`bool`, defaults to `False`):
            Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
        tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
        return_assistant_tokens_mask (`bool`, defaults to `False`):
            Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
            the mask will contain 1. For user and system tokens, the mask will contain 0.
            This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
        **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

    Returns:
        `Union[List[int], Dict]`: A list of token ids representing the tokenized chat so far, including control tokens. This
        output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
        set, will return a dict of tokenizer outputs instead.
    """

    if return_dict and not tokenize:
        raise ValueError(
            "`return_dict=True` is incompatible with `tokenize=False`, because there is no dict "
            "of tokenizer outputs to return."
        )

    if return_assistant_tokens_mask and not return_dict:
        raise ValueError("`return_assistant_tokens_mask=True` is incompatible with `return_dict=False`")

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    chat_template = self.get_chat_template(chat_template, tools)

    if return_assistant_tokens_mask and not re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):
        logger.warning_once(
            "return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword."
        )

    # Compilation function uses a cache to avoid recompiling the same template
    compiled_template = _compile_jinja_template(chat_template)

    if isinstance(conversation, (list, tuple)) and (
        isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
    ):
        conversations = conversation
        is_batched = True
    else:
        conversations = [conversation]
        is_batched = False

    if continue_final_message:
        if add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
            )
        if return_assistant_tokens_mask:
            raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

    # We accept either JSON schemas or functions for tools. If we get functions, we convert them to schemas
    if tools is not None:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif isfunction(tool):
                tool_schemas.append(get_json_schema(tool))
            else:
                raise ValueError(
                    "Tools should either be a JSON schema, or a callable function with type hints "
                    "and a docstring suitable for auto-conversion to a schema."
                )
    else:
        tool_schemas = None

    if documents is not None:
        for document in documents:
            if not isinstance(document, dict):
                raise TypeError("Documents should be a list of dicts with 'title' and 'text' keys!")

    rendered = []
    all_generation_indices = []
    template_kwargs = {**self.special_tokens_map, **kwargs}  # kwargs overwrite special tokens if both are present
    for chat in conversations:
        if hasattr(chat, "messages"):
            # Indicates it's a Conversation object
            chat = chat.messages
        if return_assistant_tokens_mask:
            rendered_chat, generation_indices = _render_with_assistant_indices(
                compiled_template=compiled_template,
                messages=chat,
                tools=tool_schemas,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
            all_generation_indices.append(generation_indices)
        else:
            rendered_chat = compiled_template.render(
                messages=chat,
                tools=tool_schemas,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
        if continue_final_message:
            final_message = chat[-1]["content"].strip()
            rendered_chat = rendered_chat[: rendered_chat.rindex(final_message) + len(final_message)].rstrip()
        rendered.append(rendered_chat)

    if not is_batched:
        rendered = rendered[0]

    if tokenize:
        out = self(
            rendered,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        if return_dict:
            if return_assistant_tokens_mask:
                assistant_masks = []
                if is_batched or return_tensors:
                    input_ids = out["input_ids"]
                else:
                    input_ids = [out["input_ids"]]
                for i in range(len(input_ids)):
                    current_mask = [0] * len(input_ids[i])
                    for assistant_start_char, assistant_end_char in all_generation_indices[i]:
                        start_token = out.char_to_token(i, assistant_start_char)
                        end_token = out.char_to_token(i, assistant_end_char - 1)
                        if start_token is None:
                            # start_token is out of bounds maybe due to truncation.
                            break
                        for token_id in range(start_token, end_token + 1 if end_token else len(input_ids)):
                            current_mask[token_id] = 1
                    assistant_masks.append(current_mask)
                out["assistant_masks"] = assistant_masks if is_batched else assistant_masks[0]
            return out
        else:
            return out["input_ids"]
    else:
        return rendered


def get_llava_rope_index(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    return None, None



def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}]}
    ]

    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_llava_visual(
    sources: List[Dict],
    processor,
):
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )

    # Already handled by replacing chat template
    # if use_im_start_end_tokens:
    #     text = text.replace(DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}")

    input_ids = full_result["input_ids"]

    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()

    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        # If we find `ASSISTANT: ` tokens (319, 1799, 9047, 13566)
        if pos + 4 < L and input_ids_flat[pos: pos+4] == [319, 1799, 9047, 13566]:
            ans_start = pos + 2
            ans_end = ans_start
            # Find the </s> token (2) or USER: tokens (3148, 1001, 29901)
            while (ans_end < L and input_ids_flat[ans_end] != 2):
                ans_end += 1
            if ans_end < L and input_ids_flat[ans_end] == 2:
                labels[0, ans_start + 3 : ans_end + 2] = input_ids[
                    0, ans_start + 3 : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels

    full_result["input_ids"] = input_ids

    return full_result
