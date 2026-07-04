#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
SRC_ROOT="data/hf_videos/llava_hound/train_300k"
DST_ROOT="data/hf_videos/llava_hound/train_300k_mp4"
FPS="${FPS:-8}"             # export FPS=30 to override (input sequence rate)
PARALLEL="${PARALLEL:-16}"  # export PARALLEL=16 to use more workers
FORCE="${FORCE:-0}"         # export FORCE=1 to overwrite existing mp4s
SLOW="${SLOW:-1}"           # export SLOW=2 for 2× slower (via setpts)
SCALE_EVEN="${SCALE_EVEN:-0}" # export SCALE_EVEN=1 to scale instead of pad
LOG="${LOG:-/tmp/frames2mp4.log}"

mkdir -p "$DST_ROOT"
echo "==> Writing logs to $LOG"
: > "$LOG"

convert_dir() {
  local dir="$1"
  local name out img_glob count vf

  name="$(basename "$dir")"
  out="$DST_ROOT/$name.mp4"

  if [[ "$FORCE" != "1" && -f "$out" ]]; then
    echo "[SKIP] $name (exists)" | tee -a "$LOG"
    return 0
  fi

  # choose pattern: prefer *.jpeg, fallback to *.jpg
  if compgen -G "$dir/*.jpeg" > /dev/null; then
    img_glob="$dir/*.jpeg"
  elif compgen -G "$dir/*.jpg" > /dev/null; then
    img_glob="$dir/*.jpg"
  else
    echo "[WARN] $name has no jpg/jpeg files" | tee -a "$LOG"
    return 0
  fi

  # Count frames robustly
  count=$(find "$dir" -maxdepth 1 -type f \( -name '*.jpeg' -o -name '*.jpg' \) | wc -l | tr -d ' ')
  if [[ "$count" -lt 2 ]]; then
    echo "[WARN] $name has <2 frames ($count). Skipping." | tee -a "$LOG"
    return 0
  fi

  # Build filter chain: evenize + optional slow-mo
  if [[ "$SCALE_EVEN" == "1" ]]; then
    vf="scale=trunc(iw/2)*2:trunc(ih/2)*2"
  else
    vf="pad=ceil(iw/2)*2:ceil(ih/2)*2"
  fi
  if [[ "$SLOW" != "1" ]]; then
    vf="$vf,setpts=${SLOW}*PTS"
  fi

  echo "[DO] $name -> $out (frames=$count, fps=$FPS, vf='$vf')" | tee -a "$LOG"
  ffmpeg -hide_banner -loglevel error -y \
    -framerate "$FPS" -pattern_type glob -i "$img_glob" \
    -vf "$vf" -r "$FPS" \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    "$out" \
    && echo "[OK] $name" | tee -a "$LOG" \
    || { echo "[ERR] $name" | tee -a "$LOG"; return 1; }
}

export -f convert_dir
export DST_ROOT FPS FORCE LOG SLOW SCALE_EVEN

# find subdirs (one scene per dir) and run in parallel
find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 \
| xargs -0 -I{} -P "$PARALLEL" bash -c 'convert_dir "$@"' _ {}

echo "All done. Output in: $DST_ROOT"