"""Local Hugging Face Transformers inference (Part B §19, §38, §39).

Imports of `torch` and `transformers` are deferred to construction, so a Colab runtime without a
GPU — or a CI machine without either package — can import the registry and every other provider
without paying for a multi-second import it will not use.

Device, dtype and VRAM are *detected* (§38). Nothing here assumes a GPU exists.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from civitas.config import Settings, get_settings
from civitas.runtime.providers.base import (
    Completion,
    CompletionRequest,
    ModelInfo,
    Provider,
    ProviderError,
    Role,
    Usage,
)
from civitas.runtime.providers.base import (
    CompletionRequest as _Req,  # noqa: F401  (kept for readability of the signature below)
)

log = logging.getLogger(__name__)


def detect_device() -> dict[str, Any]:
    """CUDA availability, device, dtype and VRAM (Part B §38).

    Returns a plain dict rather than raising when torch is absent: the Colab notebook shows this
    to the user before deciding what to launch, and "no torch" is information, not an error.
    """
    info: dict[str, Any] = {
        "torch": False, "cuda": False, "device": "cpu", "dtype": "float32",
        "device_name": None, "vram_total_gb": None, "vram_free_gb": None,
    }
    try:
        import torch
    except ImportError:
        return info

    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info.update(
            cuda=True,
            device="cuda",
            device_name=torch.cuda.get_device_name(0),
            vram_total_gb=round(total / 1024**3, 2),
            vram_free_gb=round(free / 1024**3, 2),
            # bfloat16 needs Ampere or newer; on older cards it silently falls back to a slow
            # emulation path, so the capability is checked rather than assumed.
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        )
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info.update(device="mps", dtype="float16")
    return info


class HuggingFaceProvider(Provider):
    """A local causal LM (Part B §19, §38).

    Weights are never stored in the repository (§39); the cache directory can be pointed at Drive
    so a Colab reconnect does not re-download them.
    """

    name = "huggingface"

    def __init__(
        self,
        model_id: str,
        settings: Settings | None = None,
        *,
        device: str | None = None,
        dtype: str | None = None,
        cache_dir: str | None = None,
        max_context: int | None = None,
    ):
        self._settings = settings or get_settings()
        self.model_id = model_id
        detected = detect_device()
        self.device = device or detected["device"]
        self.dtype_name = dtype or detected["dtype"]
        self.cache_dir = cache_dir or os.environ.get("HF_HOME")
        self._max_context = max_context
        self._model = None
        self._tokenizer = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._tokenizer
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderError(
                "huggingface: transformers and torch are not installed; "
                "install the `local` extra or use a remote provider",
                retryable=False,
            ) from exc

        dtype = getattr(torch, self.dtype_name, torch.float32)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()
        return self._model, self._tokenizer

    def model_info(self, model: str) -> ModelInfo:
        context = self._max_context or 8192
        if self._tokenizer is not None:
            context = getattr(self._tokenizer, "model_max_length", context)
            if context > 10**6:  # some tokenizers report a sentinel
                context = self._max_context or 8192
        return ModelInfo(
            name=model or self.model_id,
            provider=self.name,
            context_window=int(context),
            max_output_tokens=4096,
            # A base causal LM has no native tool protocol. Claiming otherwise would let the
            # runtime hand it tool specs it cannot honour.
            supports_tools=False,
            supports_streaming=True,
            supports_structured_output=False,
            is_deterministic=True,
            version=self.model_id,
        )

    def _render(self, request: CompletionRequest) -> str:
        _, tokenizer = self._load()
        chat = [
            {"role": m.role.value, "content": m.content}
            for m in request.messages
            if m.role in (Role.SYSTEM, Role.USER, Role.ASSISTANT)
        ]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in chat) + "\n\nassistant:"

    def complete(self, request: CompletionRequest) -> Completion:
        import torch

        model, tokenizer = self._load()
        started = time.perf_counter()
        prompt = self._render(request)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                # temperature 0 means greedy. Passing do_sample=True with temperature 0 is a
                # runtime error in transformers, and passing temperature with greedy decoding is
                # silently ignored — either way the frozen-model guarantee would be a fiction.
                do_sample=request.temperature > 0,
                temperature=request.temperature if request.temperature > 0 else None,
                top_p=request.top_p if request.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = out[0][prompt_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        return Completion(
            text=text,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=int(generated.shape[-1])),
            stop_reason="end_turn",
            model=self.model_id,
            model_version=self.model_id,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=0.0,
        )

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return super().count_tokens(text)
        return len(self._tokenizer.encode(text))

    def unload(self) -> None:
        """Free VRAM (Part B §58 model unload/reload)."""
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def close(self) -> None:
        self.unload()
