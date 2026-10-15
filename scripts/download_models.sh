#!/usr/bin/env bash
# 下载实验所需模型。
#
# 为什么默认走 ModelScope 而不是 HuggingFace：
# 本机实测（2026-10）HF 经代理约 355 KB/s，hf-mirror 约 190 KB/s，
# ModelScope 约 3.0 MB/s —— 7B 模型 15 GB，走 HF 要 12 小时，走 ModelScope 约 1.5 小时。
# 模型权重两者完全一致（ModelScope 是官方同步镜像）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python"
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct}"

for repo in $MODELS; do
  name="${repo##*/}"
  dest="models/$name"
  if [ -f "$dest/config.json" ] && ls "$dest"/*.safetensors >/dev/null 2>&1; then
    echo "[skip] $name 已存在"
    continue
  fi
  echo "=== 下载 $repo -> $dest ==="
  "$PY" - "$repo" "$dest" <<'PYEOF'
import sys, time
from modelscope import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
path = snapshot_download(repo, local_dir=dest, max_workers=8)
print(f"完成 {repo} -> {path}  ({time.time()-t0:.0f}s)")
PYEOF
done

echo
echo "=== 已下载模型 ==="
du -sh models/*/ 2>/dev/null
