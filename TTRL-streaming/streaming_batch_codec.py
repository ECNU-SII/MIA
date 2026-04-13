"""
Serialize / deserialize a collated batch dict (``collate_fn`` output) for HTTP.

Tensor fields use: ``{"dtype": "int64"|"float32"|..., "data": <nested list>}``.
Non-tensor fields: JSON list of length ``batch_size`` (becomes ``np.ndarray`` ``dtype=object``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


def _torch_dtype_name(t: torch.Tensor) -> str:
    m = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.bfloat16: "bfloat16",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.bool: "bool",
    }
    return m.get(t.dtype, "float32")


def strip_ground_truth_for_server_submit(batch_dict: dict) -> dict:
    """
    流式训练时标准答案只在 GT 服务（:6000）注册；发往 TTRL 的 ``reward_model`` 中清空 ``ground_truth``。
    """
    if "reward_model" not in batch_dict:
        return batch_dict
    v = batch_dict["reward_model"]
    if not isinstance(v, np.ndarray) or v.dtype != object:
        return batch_dict
    new_arr = np.empty(len(v), dtype=object)
    for i, rm in enumerate(v.tolist()):
        if isinstance(rm, dict):
            rm2 = dict(rm)
            rm2["ground_truth"] = ""
            new_arr[i] = rm2
        else:
            new_arr[i] = rm
    out = dict(batch_dict)
    out["reward_model"] = new_arr
    return out


def strip_media_for_server_submit(batch_dict: dict, *, drop_keys: frozenset[str] | None = None) -> dict:
    """
    流式 TTRL 服务端不接收原始图片/视频列；在 ``batch_dict_to_jsonable`` 之前删掉这些键。
    图片应由客户端另行 ``POST`` 到 Agent ``/streaming/register_images``。
    """
    drop = drop_keys if drop_keys is not None else frozenset({"images", "videos"})
    return {k: v for k, v in batch_dict.items() if k not in drop}


def batch_dict_to_jsonable(batch_dict: dict) -> dict:
    """Convert collated batch to JSON-serializable dict (for client POST)."""
    out: dict[str, Any] = {}
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = {"__tensor__": True, "dtype": _torch_dtype_name(v), "data": v.detach().cpu().tolist()}
        elif isinstance(v, np.ndarray):
            if v.dtype == object:
                out[k] = {"__ndarray_object__": True, "data": [x for x in v.tolist()]}
            else:
                out[k] = {
                    "__ndarray_numpy__": True,
                    "dtype": str(v.dtype),
                    "data": v.tolist(),
                }
        else:
            out[k] = v
    return out


def batch_dict_from_jsonable(payload: dict) -> dict:
    """Restore collated batch from JSON-decoded dict (server side)."""
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, dict) and v.get("__tensor__"):
            dt = _TORCH_DTYPES.get(v["dtype"], torch.float32)
            out[k] = torch.tensor(v["data"], dtype=dt)
        elif isinstance(v, dict) and v.get("__ndarray_object__"):
            arr = np.empty(len(v["data"]), dtype=object)
            for i, x in enumerate(v["data"]):
                arr[i] = x
            out[k] = arr
        elif isinstance(v, dict) and v.get("__ndarray_numpy__"):
            out[k] = np.array(v["data"], dtype=np.dtype(v["dtype"]))
        elif isinstance(v, list):
            arr = np.empty(len(v), dtype=object)
            for i, x in enumerate(v):
                arr[i] = x
            out[k] = arr
        else:
            out[k] = v
    return out
