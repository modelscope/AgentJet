# -*- coding: utf-8 -*-
"""
Multimodal helpers for AgentJet.

Main purpose: convert dataset samples that carry image payloads (PIL,
bytes, dict with 'bytes', file paths, or data URLs) into OpenAI
chat-completions content blocks so they can be forwarded to a
vision-language backend (vLLM serving e.g. Qwen2.5-VL).

The AgentJet task reader leaves the raw dataset example in
``task.metadata``, which usually contains an 'image' or 'images' field
when working with multimodal datasets such as geo3k. Use
``build_multimodal_messages`` or ``extract_image`` below to normalize
that into standard OpenAI vision format.
"""

from __future__ import annotations

import base64
import io
import os
from typing import List, Optional, Union

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    _PIL_AVAILABLE = False


ImageLike = Union["Image.Image", bytes, dict, str]


def _pil_to_data_url(img: "Image.Image", fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def load_image_to_pil(image: ImageLike) -> "Image.Image":
    """Load any supported image representation into a PIL.Image."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for multimodal support but is not installed.")

    if isinstance(image, Image.Image):
        return image

    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image)))

    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            return Image.open(io.BytesIO(image["bytes"]))
        if "path" in image and image["path"]:
            return Image.open(image["path"])
        raise ValueError(f"Unsupported image dict keys: {list(image.keys())}")

    if isinstance(image, str):
        if image.startswith("data:"):
            # data:image/png;base64,XXXX
            header, _, payload = image.partition(",")
            raw = base64.b64decode(payload)
            return Image.open(io.BytesIO(raw))
        if image.startswith("http://") or image.startswith("https://"):
            import urllib.request
            with urllib.request.urlopen(image) as r:
                return Image.open(io.BytesIO(r.read()))
        return Image.open(image)

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def encode_image_as_data_url(image: ImageLike) -> str:
    """Normalize many image representations into a data: URL."""
    if isinstance(image, str):
        # file path or already a URL/data URL
        if image.startswith("data:") or image.startswith("http://") or image.startswith("https://"):
            return image
        if os.path.isfile(image):
            with open(image, "rb") as f:
                raw = f.read()
            ext = os.path.splitext(image)[1].lstrip(".").lower() or "png"
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        raise ValueError(f"String image is neither a URL nor an existing file: {image!r}")

    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            raw = image["bytes"]
            mime = image.get("mime") or "image/png"
            return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        if "path" in image and image["path"]:
            return encode_image_as_data_url(image["path"])
        raise ValueError(f"Unsupported image dict keys: {list(image.keys())}")

    if isinstance(image, (bytes, bytearray)):
        return f"data:image/png;base64,{base64.b64encode(bytes(image)).decode('ascii')}"

    if _PIL_AVAILABLE and isinstance(image, Image.Image):
        return _pil_to_data_url(image)

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def extract_image(metadata: dict) -> Optional[ImageLike]:
    """Pull the first image out of a dataset row's metadata dict.

    Looks at 'image' and 'images' (list) keys. Returns None if no image.
    """
    img = metadata.get("image", metadata.get("images", None))
    if img is None:
        return None
    if isinstance(img, list):
        if len(img) == 0:
            return None
        img = img[0]
    return img


def build_multimodal_messages(
    system_prompt: Optional[str],
    user_text: str,
    image: Optional[ImageLike] = None,
    images: Optional[List[ImageLike]] = None,
) -> List[dict]:
    """Assemble OpenAI-compatible chat messages with optional images.

    If no image is provided, returns a plain text message. If one or
    more images are provided, the user turn uses the vision content
    blocks format (list of {type: image_url | text}).
    """
    msgs: List[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    imgs: List[ImageLike] = list(images or [])
    if image is not None:
        imgs.insert(0, image)

    if not imgs:
        msgs.append({"role": "user", "content": user_text})
    else:
        content: List[dict] = []
        for im in imgs:
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_as_data_url(im)},
            })
        content.append({"type": "text", "text": user_text})
        msgs.append({"role": "user", "content": content})

    return msgs


def task_metadata_to_messages(
    metadata: dict,
    system_prompt: Optional[str],
    user_text_key: str = "question",
    user_text: Optional[str] = None,
) -> List[dict]:
    """Shortcut: build messages from a task metadata dict.

    If ``user_text`` is None we fall back to ``metadata[user_text_key]``.
    """
    text = user_text if user_text is not None else metadata.get(user_text_key, "")
    image = extract_image(metadata)
    return build_multimodal_messages(system_prompt, text, image=image)


def extract_images_from_openai_messages(messages: List[dict]) -> List:
    """Pull image refs out of OpenAI chat messages with vision content blocks.

    Looks for ``{"type": "image_url", "image_url": {"url": ...}}`` and
    ``{"type": "image", "image": ...}`` blocks across all messages and
    returns the list of image references in order. Non-image content
    (text) is left alone.
    """
    out: List = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "image_url":
                iu = item.get("image_url")
                if isinstance(iu, dict):
                    url = iu.get("url")
                elif isinstance(iu, str):
                    url = iu
                else:
                    url = None
                if url is not None:
                    out.append(url)
            elif t == "image":
                ref = item.get("image") or item.get("url")
                if ref is not None:
                    out.append(ref)
    return out


__all__ = [
    "encode_image_as_data_url",
    "load_image_to_pil",
    "extract_image",
    "extract_images_from_openai_messages",
    "build_multimodal_messages",
    "task_metadata_to_messages",
]
