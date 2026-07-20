from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_utf8_environment() -> None:
    """统一当前程序及其后续 Python 子进程的编码。"""


    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)

        if stream is None or not hasattr(stream, "reconfigure"):
            continue

        try:
            stream.reconfigure(
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        except (AttributeError, ValueError, OSError):
            pass


def decode_output(data: str | bytes | bytearray | None) -> str:
    """
    将子进程、系统命令等输出安全转换为字符串。

    优先按照 UTF-8 解码；
    Windows 原生命令可能输出 GBK/GB18030，因此进行兼容回退。
    """

    if data is None:
        return ""

    if isinstance(data, str):
        return data

    raw = bytes(data)

    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")

    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def read_text_compatible(path: str | Path) -> str:
    """
    读取 UTF-8、UTF-8 BOM 或旧版 GBK/GB18030 文本文件。
    建议旧文件读取后重新保存为 UTF-8。
    """

    raw = Path(path).read_bytes()

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"无法识别文件编码：{path}")


def write_utf8_text(path: str | Path, text: str) -> None:
    """始终以 UTF-8 保存文本。"""

    Path(path).write_text(
        text,
        encoding="utf-8",
        errors="strict",
        newline="\n",
    )