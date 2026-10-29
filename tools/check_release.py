"""Check release text for private deployment settings without printing secrets."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "credential": re.compile(
        r"sk-[A-Za-z0-9_-]{16,}|(?i:wandb_api_key)\s*=\s*[\x22\x27][a-f0-9]{30,}"
    ),
    "private_path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/|/[a-z]+/home/"),
    "access_token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "literal_secret": re.compile(r"(?i)(?:password|passwd|api_key|access_token)\s*[:=]\s*[\x22\x27](?!EMPTY[\x22\x27])[^\x22\x27\n]{8,}[\x22\x27]"),
    "private_endpoint": re.compile(
        r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)"
    ),
    "communication_override": re.compile(
        r"(?:export\s+|environ\[)[\x22\x27]?(?:NCCL_(?:P2P_DISABLE|SHM_DISABLE|IB_DISABLE|SOCKET_IFNAME)|GLOO_SOCKET_IFNAME)"
    ),
}


def main():
    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(
            p in {".git", "__pycache__", "vendor", ".venv"} or p.endswith(".egg-info")
            for p in path.relative_to(ROOT).parts
        ):
            continue
        if path == Path(__file__).resolve():
            continue
        text = path.read_text()
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
        for line_number, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    failures.append(f"{path.relative_to(ROOT)}:{line_number}: {name}")
    print("\n".join(failures) if failures else "Release source checks passed")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
