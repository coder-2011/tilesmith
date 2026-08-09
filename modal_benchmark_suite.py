"""Run the Tilesmith workload suite on one Modal L4 GPU and save local charts."""

import json
import os
from pathlib import Path

import modal

from benchmark_suite import write_artifacts

ROOT = Path(__file__).resolve().parent
SKILL_ROOT = Path.home() / ".codex" / "skills"
CUDA_SKILL_ROOT = SKILL_ROOT / "high-performance-cuda"
CLIPROXY_VERSION = "7.2.120"
CLIPROXY_ARCHIVE = f"CLIProxyAPI_{CLIPROXY_VERSION}_linux_amd64.tar.gz"
CLIPROXY_SHA256 = "8933332737338be5d5cedae4b96254b9afc8dfe0f13c4322738c65fd7931ce0a"
CLIPROXY_API_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "VERTEX_API_KEY",
    "OPENROUTER_API_KEY",
    "TILESMITH_API_KEY",
)
CLIPROXY_API_ENVIRONMENT = (
    *CLIPROXY_API_KEYS,
    "TILESMITH_API_BASE_URL",
)
# Exported provider credentials enter the remote container only through a Modal secret.
_exported_api_environment = [name for name in CLIPROXY_API_ENVIRONMENT if os.environ.get(name)]
CLIPROXY_API_SECRETS = [modal.Secret.from_local_environ(_exported_api_environment)] if _exported_api_environment else []
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "curl")
    .run_commands(
        f"curl -fsSL https://github.com/router-for-me/CLIProxyAPI/releases/download/v{CLIPROXY_VERSION}/{CLIPROXY_ARCHIVE} -o /tmp/{CLIPROXY_ARCHIVE}",
        f"echo '{CLIPROXY_SHA256}  /tmp/{CLIPROXY_ARCHIVE}' | sha256sum -c -",
        "mkdir -p /tmp/tilesmith-cliproxyapi",
        f"tar -xzf /tmp/{CLIPROXY_ARCHIVE} -C /tmp/tilesmith-cliproxyapi",
        "install -m 0755 /tmp/tilesmith-cliproxyapi/cli-proxy-api /usr/local/bin/cliproxyapi",
        f"rm -f /tmp/{CLIPROXY_ARCHIVE}",
        "rm -rf /tmp/tilesmith-cliproxyapi",
    )
    .uv_pip_install("torch", "deepagents>=0.7.1,<0.8", "langchain-openai>=1.4.1,<2", "pyyaml>=6,<7")
    .add_local_dir(ROOT / "src", "/root/src")
    .add_local_file(ROOT / "benchmark_suite.py", "/root/benchmark_suite.py")
    .add_local_file(SKILL_ROOT / "check-normal" / "SKILL.md", "/root/.codex/skills/check-normal/SKILL.md")
    .add_local_file(SKILL_ROOT / "high-performance-triton" / "SKILL.md", "/root/.codex/skills/high-performance-triton/SKILL.md")
    .add_local_file(Path.home() / "Documents/kernel-design-agents/skills/KernelWiki/SKILL.md", "/root/Documents/kernel-design-agents/skills/KernelWiki/SKILL.md")
    .add_local_file(CUDA_SKILL_ROOT / "SKILL.md", "/root/.codex/skills/high-performance-cuda/SKILL.md")
    .add_local_file(CUDA_SKILL_ROOT / "references/core-principles.md", "/root/.codex/skills/high-performance-cuda/references/core-principles.md")
    .add_local_file(CUDA_SKILL_ROOT / "references/autotuning-and-benchmarking.md", "/root/.codex/skills/high-performance-cuda/references/autotuning-and-benchmarking.md")
    .add_local_file(CUDA_SKILL_ROOT / "references/numerical-correctness.md", "/root/.codex/skills/high-performance-cuda/references/numerical-correctness.md")
    .add_local_file(CUDA_SKILL_ROOT / "references/triton-and-codegen.md", "/root/.codex/skills/high-performance-cuda/references/triton-and-codegen.md")
)
APP = modal.App("tilesmith-workload-benchmarks")
EVIDENCE_VOLUME = modal.Volume.from_name("tilesmith-benchmark-evidence", create_if_missing=True)
CLIPROXY_AUTH_VOLUME = modal.Volume.from_name("tilesmith-cliproxyapi-auth", create_if_missing=True)


# run_suite places every workload and loop count on one shared L4 software stack.
@APP.function(
    image=IMAGE,
    gpu="L4",
    timeout=14_400,
    secrets=CLIPROXY_API_SECRETS,
    env={"PYTHONPATH": "/root/src", "TILESMITH_DEBUG": "1"},
    volumes={
        "/root/.cache/tilesmith": EVIDENCE_VOLUME,
        "/root/.cli-proxy-api": CLIPROXY_AUTH_VOLUME,
    },
)
def run_suite(agent_model, loops, max_model_calls, max_output_tokens):
    """Execute the CUDA suite and return its JSON-serializable evidence."""
    from benchmark_suite import run_suite as run_cuda_suite

    os.environ["TILESMITH_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    # Commit even after a fatal suite error so the last partial agent trace survives container teardown.
    try:
        result = run_cuda_suite(agent_model, loops=tuple(loops), max_model_calls=max_model_calls)
    finally:
        EVIDENCE_VOLUME.commit()
        CLIPROXY_AUTH_VOLUME.commit()
    return json.dumps(result, sort_keys=True)


# _parse_loops validates the local CLI's compact comma-separated loop list.
def _parse_loops(value):
    """Convert a comma-separated positive-integer list into an immutable tuple."""
    try:
        loops = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("loops must be comma-separated positive integers") from error
    if not loops or any(loop < 1 for loop in loops):
        raise ValueError("loops must be comma-separated positive integers")
    return loops


# _require_cliproxy_configuration prevents a GPU launch without OAuth, config, or exported API credentials.
def _require_cliproxy_configuration():
    """Require CLIProxyAPI OAuth files, a provider config, or exported API credentials."""
    generic_key = bool(os.environ.get("TILESMITH_API_KEY"))
    generic_base_url = bool(os.environ.get("TILESMITH_API_BASE_URL"))
    if generic_key != generic_base_url:
        raise RuntimeError("TILESMITH_API_KEY and TILESMITH_API_BASE_URL must be exported together")
    if any(os.environ.get(name) for name in CLIPROXY_API_KEYS):
        return
    entries = CLIPROXY_AUTH_VOLUME.listdir("/", recursive=False)
    if any(entry.type.name == "FILE" for entry in entries):
        return
    oauth_command = "modal volume put tilesmith-cliproxyapi-auth ~/.cli-proxy-api ."
    api_command = "modal volume put tilesmith-cliproxyapi-auth path/to/config.yaml config.yaml"
    raise RuntimeError(
        f"CLIProxyAPI is unconfigured; export a provider API key, upload OAuth files with `{oauth_command}`, "
        f"or upload an API config with `{api_command}`"
    )


# _download_evidence copies only this sweep's persistent run directories to the result folder.
def _download_evidence(result, output_dir):
    """Download attempts, reports, and traces for every remote run that created evidence."""
    destination = Path(output_dir) / "runs"
    downloaded = set()
    run_ids = {value.get("tilesmith_run_id") for value in result["results"]}
    for run_id in sorted(value for value in run_ids if value):
        remote_root = f"runs/{run_id}"
        try:
            entries = EVIDENCE_VOLUME.listdir(remote_root, recursive=True)
        except modal.exception.NotFoundError:
            continue
        for entry in entries:
            if entry.type.name != "FILE":
                continue
            relative = Path(entry.path.lstrip("/")).relative_to(remote_root)
            local_path = destination / run_id / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as output:
                for chunk in EVIDENCE_VOLUME.read_file(entry.path):
                    output.write(chunk)
        downloaded.add(run_id)
    return downloaded


# main submits the remote suite and creates charts on the caller's filesystem.
@APP.local_entrypoint()
def main(
    agent_model: str = "gpt-5.5",
    loops: str = "1,2,3",
    max_model_calls: int = 24,
    max_output_tokens: int = 8_192,
    output_dir: str = "benchmark-results",
):
    """Run the requested sweep and persist its raw evidence and SVG charts locally."""
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    _require_cliproxy_configuration()
    result = json.loads(run_suite.remote(agent_model, _parse_loops(loops), max_model_calls, max_output_tokens))
    downloaded = _download_evidence(result, output_dir)
    for value in result["results"]:
        if value.get("tilesmith_run_id") in downloaded:
            value["tilesmith_evidence_path"] = f"runs/{value['tilesmith_run_id']}"
    write_artifacts(result, output_dir)
    print(f"wrote {Path(output_dir).resolve()}", flush=True)
