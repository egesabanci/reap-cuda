#!/usr/bin/env python3
"""
Interactive REPL for testing base & pruned LFM2.5 MoE models.

Usage:
    source .venv/bin/activate
    python scripts/repl.py [--context-len 8192]

Commands (type /help in the REPL):
    /switch <n>   Switch to model n
    /list         List available models
    /stats        Show current model stats
    /context <n>  Change max context length
    /clear        Clear conversation history
    /raw          Toggle raw mode (no chat template)
    /help         Show this help
    /quit         Exit
"""

from __future__ import annotations

import argparse
import cmd
import os
import sys
import time
import textwrap
from dataclasses import dataclass, field
from typing import Any

import torch
import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Available models ────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    name: str
    path: str
    description: str

MODELS: list[ModelEntry] = [
    ModelEntry(
        "base",
        "/data/models/LiquidAI/LFM2.5-8B-A1B",
        "LFM2.5-8B-A1B (32 experts, 16GB, bf16) — ORIGINAL",
    ),
    ModelEntry(
        "pruned-4096",
        "/data/reap-lfm2-cli-4096/LFM2.5-8B-A1B/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_42-0.50",
        "LFM2.5 pruned 32→16 experts (4096-sample calib, compression 0.5, seed=42)",
    ),
    ModelEntry(
        "pruned-200",
        "/data/reap-lfm2-run6/model_--data--models--LiquidAI--LFM2.5-8B-A1B-f1ade47e629a/dataset_theblackcat102--evol-codealpaca-v1-9d908ea05bb5/pruned_models/reap-renorm_true-seed_42-0.50",
        "LFM2.5 pruned 32→16 experts (200-sample calib, compression 0.5, seed=42)",
    ),
]

# Chat template markers for LFM2.5
CHAT_START = "<|startoftext|>"
USER_START = "<|im_start|>user\n"
USER_END = "<|im_end|>\n"
ASSISTANT_START = "<|im_start|>assistant\n"
ASSISTANT_END = "<|im_end|>\n"


# ── Model manager ───────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self, log_dir: str = "/data/reap-repl-logs"):
        self.model: Any = None
        self.tokenizer: Any = None
        self.current_idx: int = -1
        self.current_name: str = ""
        self.conversation: list[dict[str, str]] = []
        self.max_context: int = 4096
        self.raw_mode: bool = False
        self.log_file: Any = None
        self.log_dir: str = log_dir
        self._open_log()

    def load(self, idx: int) -> None:
        if idx == self.current_idx and self.model is not None:
            return
        entry = MODELS[idx]
        print(f"\n  Loading {entry.name}: {entry.description}")
        print(f"  Path: {entry.path}")
        t0 = time.time()

        # Free old model
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            entry.path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(entry.path)

        params = sum(p.numel() for p in self.model.parameters())
        vram = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        num_experts = self.model.config.num_experts

        self.current_idx = idx
        self.current_name = entry.name
        self.conversation = []

        elapsed = time.time() - t0
        print(f"  ✓ Loaded in {elapsed:.1f}s")
        print(f"    {params/1e9:.2f}B params | {num_experts} experts | VRAM: {vram:.1f}G alloc / {peak:.1f}G peak")
        print()

    def _open_log(self) -> None:
        import datetime
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"repl_{ts}.log")
        self.log_file = open(path, "a", buffering=1)
        self.log_file.write(f"# REAP REPL session started at {datetime.datetime.now().isoformat()}\n")
        self.log_file.write(f"# Log: {path}\n\n")
        print(f"  📝 Logging to {path}")

    def generate(self, user_msg: str, max_new_tokens: int = 512) -> str:
        if self.model is None:
            return "[no model loaded]"

        device = self.model.device
        t_start = time.time()

        if self.raw_mode:
            prompt = user_msg
        else:
            self.conversation.append({"role": "user", "content": user_msg})
            # Build chat string manually (avoids BatchEncoding issues)
            parts = [CHAT_START]
            for msg in self.conversation:
                if msg["role"] == "user":
                    parts.append(f"{USER_START}{msg['content']}{USER_END}")
                elif msg["role"] == "assistant":
                    parts.append(f"{ASSISTANT_START}{msg['content']}{ASSISTANT_END}")
            parts.append(ASSISTANT_START)
            prompt = "".join(parts)

        # Truncate to max_context tokens (approximate)
        enc = self.tokenizer(prompt, return_tensors="pt")
        if enc.input_ids.shape[1] > self.max_context:
            # Truncate from the beginning, keeping system context
            overflow = enc.input_ids.shape[1] - self.max_context
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context,
            )
            if not self.raw_mode:
                print(f"  ⚠️  Truncated {overflow} tokens to fit {self.max_context}-token window")

        inp = {k: v.to(device) for k, v in enc.items()}
        input_len = inp["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        new_tokens = out.shape[1] - input_len

        response = self.tokenizer.decode(
            out[0, input_len:], skip_special_tokens=True
        ).strip()

        if not self.raw_mode:
            self.conversation.append({"role": "assistant", "content": response})

        tok_s = new_tokens / elapsed if elapsed > 0 else 0
        print(f"\n  ── {new_tokens} tokens in {elapsed:.1f}s ({tok_s:.0f} tok/s) ──")

        # Log to file
        if self.log_file:
            self.log_file.write(f"## {time.strftime('%H:%M:%S')} | model={self.current_name} | {new_tokens} tok | {elapsed:.1f}s | {tok_s:.0f} tok/s\n")
            self.log_file.write(f"### User\n{user_msg}\n\n")
            self.log_file.write(f"### Assistant\n{response}\n\n\n")

        return response


# ── REPL ────────────────────────────────────────────────────────────────────

class ReplCmd(cmd.Cmd):
    intro = textwrap.dedent("""\
        ╔══════════════════════════════════════════════════╗
        ║   REAP LFM2.5 Model REPL                        ║
        ║   Base + pruned MoE models for interactive test  ║
        ╚══════════════════════════════════════════════════╝
        Type /help for commands, /list for models.
        First, load a model: /load <name or number>
    """)
    prompt = "\n  you> "

    def __init__(self, mgr: ModelManager):
        super().__init__()
        self.mgr = mgr

    def _parse_cmd(self, line: str) -> tuple[str, str]:
        """Split into command and args."""
        line = line.strip()
        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            return parts[0].lower(), parts[1] if len(parts) > 1 else ""
        return "", line

    def default(self, line: str) -> None:
        cmd_name, args = self._parse_cmd(line)
        if cmd_name:
            # Forward /command to do_command
            method = getattr(self, f"do_{cmd_name}", None)
            if method:
                method(args)
            else:
                print(f"  Unknown command: /{cmd_name}. Type /help.")
            return
        # User message
        if not self.mgr.model:
            print("  No model loaded. Use /load <name> first.")
            return
        print()
        response = self.mgr.generate(args or line)
        print(f"\n  {response}")

    # ── Commands ─────────────────────────────────────────────────────────

    def do_load(self, arg: str) -> None:
        """Load a model by name or index: /load base | /load 0"""
        arg = arg.strip()
        for i, m in enumerate(MODELS):
            if arg == m.name or arg == str(i):
                self.mgr.load(i)
                return
        print(f"  Unknown model: {arg}")
        print(f"  Available: {[m.name for m in MODELS]} (or 0-{len(MODELS)-1})")

    def do_list(self, _arg: str) -> None:
        """List available models."""
        print()
        for i, m in enumerate(MODELS):
            marker = " ← CURRENT" if i == self.mgr.current_idx else ""
            print(f"  [{i}] {m.name:<16} {m.description}{marker}")

    def do_switch(self, arg: str) -> None:
        """Alias for /load."""
        self.do_load(arg)

    def do_stats(self, _arg: str) -> None:
        """Show current model statistics."""
        if self.mgr.model is None:
            print("  No model loaded.")
            return
        m = self.mgr.model
        params = sum(p.numel() for p in m.parameters())
        vram = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        ctx_len = self.mgr.max_context
        conv_len = len(self.mgr.conversation)
        raw = "ON" if self.mgr.raw_mode else "OFF"

        print(f"""
  Model:      {self.mgr.current_name} ({MODELS[self.mgr.current_idx].description.split('(')[0].strip()})
  Experts:    {m.config.num_experts} (was 32, top_k={m.config.num_experts_per_tok})
  Params:     {params/1e9:.2f}B
  VRAM:       {vram:.1f}G alloc / {peak:.1f}G peak
  Context:    {ctx_len} tokens max  |  Raw mode: {raw}
  Conv hist:  {conv_len} messages  |  Device: {m.device}
""")

    def do_context(self, arg: str) -> None:
        """Set max context length: /context 8192"""
        try:
            n = int(arg.strip())
            if n < 128:
                print("  Min context: 128 tokens")
                return
            self.mgr.max_context = n
            print(f"  Max context set to {n} tokens")
        except ValueError:
            print(f"  Usage: /context <number>  (current: {self.mgr.max_context})")

    def do_clear(self, _arg: str) -> None:
        """Clear conversation history."""
        self.mgr.conversation = []
        torch.cuda.empty_cache()
        print("  Conversation cleared, VRAM cache freed.")

    def do_raw(self, _arg: str) -> None:
        """Toggle raw mode (no chat template)."""
        self.mgr.raw_mode = not self.mgr.raw_mode
        status = "ON (no chat template)" if self.mgr.raw_mode else "OFF (chat template)"
        print(f"  Raw mode: {status}")

    def do_help(self, _arg: str) -> None:
        """Show help."""
        print(textwrap.dedent("""\
        Commands:
          /load <name>    Load a model (base, pruned-4096, pruned-200) or index (0,1,2)
          /list           List available models
          /switch <name>  Alias for /load
          /stats          Show current model stats (VRAM, params, context)
          /context <n>    Set max context window (default 4096)
          /clear          Clear conversation history
          /raw            Toggle raw mode (no chat template)
          /help           Show this help
          /quit           Exit

        Chat format: <|startoftext|><|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n
        Model generates in <think>...</think> chain-of-thought style.
        Type your message directly (no / prefix) to chat.
        """))

    _quit_flag: bool = False

    def do_quit(self, _arg: str) -> bool:
        """Exit the REPL."""
        if self._quit_flag:
            return True
        self._quit_flag = True
        print("\n  Goodbye!")
        if hasattr(self.mgr, 'model') and self.mgr.model is not None:
            del self.mgr.model
            torch.cuda.empty_cache()
        if self.mgr.log_file:
            self.mgr.log_file.close()
        return True

    def do_EOF(self, _arg: str) -> bool:
        """Handle Ctrl+D."""
        print()
        return self.do_quit("")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="REAP LFM2.5 Model REPL")
    parser.add_argument(
        "--context-len", type=int, default=4096,
        help="Initial max context length in tokens (default: 4096)"
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Model to load on startup (name or index)"
    )
    args = parser.parse_args()

    mgr = ModelManager()
    mgr.max_context = args.context_len

    repl = ReplCmd(mgr)

    # Auto-load if requested
    if args.load:
        repl.do_load(args.load)
        repl.do_stats("")

    # Start REPL
    repl.cmdloop()


if __name__ == "__main__":
    main()
