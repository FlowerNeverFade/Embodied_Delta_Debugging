from __future__ import annotations

import argparse
import asyncio
import http
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


OPENPI_CLIENT_SRC_CANDIDATES = [
    Path(os.environ["OPENPI_CLIENT_SRC"]) if os.environ.get("OPENPI_CLIENT_SRC") else None,
    Path("/data2/yanghaoyun/research/openpi/packages/openpi-client/src"),
    Path("/root/autodl-tmp/research/openpi/packages/openpi-client/src"),
]
for _candidate in OPENPI_CLIENT_SRC_CANDIDATES:
    if _candidate is not None and _path_exists(_candidate):
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from openpi_client import msgpack_numpy  # noqa: E402


LOGGER = logging.getLogger("serve_lerobot_pi0fast_policy")


def _no_proxy_env() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        os.environ.pop(key, None)


def _image_to_tensor(value: object) -> torch.Tensor:
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")
    if arr.shape[0] == 3:
        chw = arr
    elif arr.shape[-1] == 3:
        chw = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}")
    chw = np.ascontiguousarray(chw)
    tensor = torch.from_numpy(chw)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    else:
        tensor = tensor.to(torch.float32)
        if float(tensor.max().item()) > 2.0:
            tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _state_to_tensor(value: object) -> torch.Tensor:
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    return torch.from_numpy(np.ascontiguousarray(state))


class LerobotPi0FastPolicy:
    def __init__(
        self,
        policy_dir: Path,
        device: str,
        compile_model: bool,
        compile_mode: str,
        action_tokenizer_path: Optional[Path],
        text_tokenizer_path: Optional[Path],
        local_files_only: bool,
    ) -> None:
        _no_proxy_env()
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

        config = PreTrainedConfig.from_pretrained(
            policy_dir,
            local_files_only=local_files_only,
        )
        config.device = device
        config.compile_model = compile_model
        config.compile_mode = compile_mode
        if action_tokenizer_path is not None:
            config.action_tokenizer_name = str(action_tokenizer_path)
        if text_tokenizer_path is not None:
            config.text_tokenizer_name = str(text_tokenizer_path)

        LOGGER.info("Loading PI0FastPolicy from %s", policy_dir)
        start = time.monotonic()
        self.policy = PI0FastPolicy.from_pretrained(
            policy_dir,
            config=config,
            local_files_only=local_files_only,
        ).to(device).eval()
        LOGGER.info("Loaded policy in %.2fs", time.monotonic() - start)

        preprocessor_overrides: dict[str, dict[str, object]] = {
            "device_processor": {"device": device},
        }
        if action_tokenizer_path is not None:
            preprocessor_overrides["action_tokenizer_processor"] = {
                "action_tokenizer_name": str(action_tokenizer_path)
            }
        if text_tokenizer_path is not None:
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(text_tokenizer_path)
            }
            preprocessor_overrides.setdefault("action_tokenizer_processor", {})[
                "paligemma_tokenizer_name"
            ] = str(text_tokenizer_path)

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            str(policy_dir),
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.metadata = {
            "policy_type": "lerobot_pi0fast",
            "policy_dir": str(policy_dir),
            "device": device,
            "compile_model": compile_model,
            "compile_mode": compile_mode,
            "action_tokenizer_path": None
            if action_tokenizer_path is None
            else str(action_tokenizer_path),
            "text_tokenizer_path": None if text_tokenizer_path is None else str(text_tokenizer_path),
            "local_files_only": local_files_only,
            "n_action_steps": int(self.policy.config.n_action_steps),
            "action_dim": int(self.policy.config.output_features["action"].shape[0]),
        }

    def infer(self, obs: dict) -> dict:
        frame = {
            "observation.images.image": _image_to_tensor(obs["observation/image"]),
            "observation.images.image2": _image_to_tensor(obs["observation/wrist_image"]),
            "observation.state": _state_to_tensor(obs["observation/state"]),
            "task": str(obs.get("prompt") or ""),
        }
        batch = self.preprocess(frame)
        with torch.inference_mode():
            action_chunk = self.policy.predict_action_chunk(batch)
            action_chunk = self.postprocess(action_chunk)
        actions = torch.as_tensor(action_chunk).detach().cpu()
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return {"actions": actions.numpy().astype(np.float32)}


class WebsocketServer:
    def __init__(self, policy: LerobotPi0FastPolicy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            LOGGER.info("Serving LeRobot pi0fast policy on %s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        LOGGER.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        prev_total_time = None
        while True:
            try:
                start = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                infer_start = time.monotonic()
                result = self.policy.infer(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                result["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start
            except websockets.ConnectionClosed:
                LOGGER.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: websocket_server.ServerConnection, request: websocket_server.Request
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a LeRobot PI0Fast policy over OpenPI websocket protocol.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--action-tokenizer-path", type=Path, default=None)
    parser.add_argument("--text-tokenizer-path", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-hub-download", dest="local_files_only", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    policy = LerobotPi0FastPolicy(
        policy_dir=args.policy_dir,
        device=args.device,
        compile_model=bool(args.compile_model),
        compile_mode=str(args.compile_mode),
        action_tokenizer_path=args.action_tokenizer_path,
        text_tokenizer_path=args.text_tokenizer_path,
        local_files_only=bool(args.local_files_only),
    )
    WebsocketServer(policy, args.host, int(args.port)).serve_forever()


if __name__ == "__main__":
    main()
