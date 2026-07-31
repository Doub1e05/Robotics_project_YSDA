from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _extract_episode_line(rows_path: Path, total_episode_idx: int) -> str:
    needle = f'"total_episode_idx": {int(total_episode_idx)}'
    with rows_path.open(encoding="utf-8") as f:
        for line in f:
            if needle in line:
                return line.strip()
    raise ValueError(f"Episode with total_episode_idx={total_episode_idx} not found in {rows_path}")


def _chunk_segments_from_line(line: str) -> list[tuple[int, int, int]]:
    step_match = re.search(r'"step_count":\s*(\d+)', line)
    if step_match is None:
        raise ValueError("Could not parse step_count from episode trace line.")
    step_count = int(step_match.group(1))

    pattern = re.compile(r'"chunk_id":\s*(\d+).*?"query_timestep":\s*(\d+)', re.DOTALL)
    starts = [(int(chunk_id), int(query_timestep)) for chunk_id, query_timestep in pattern.findall(line)]
    if not starts:
        raise ValueError("Could not parse chunk_id/query_timestep pairs from episode trace line.")

    segments: list[tuple[int, int, int]] = []
    for idx, (chunk_id, start_t) in enumerate(starts):
        next_start = starts[idx + 1][1] if idx + 1 < len(starts) else step_count
        end_t = max(start_t, next_start - 1)
        segments.append((chunk_id, start_t, end_t))
    return segments


def _drawtext_filter(segments: list[tuple[int, int, int]]) -> str:
    parts: list[str] = []
    for chunk_id, start_t, end_t in segments:
        parts.append(
            "drawtext="
            f"text='chunk {chunk_id}':"
            "x=18:y=18:"
            "fontsize=24:fontcolor=white:"
            "box=1:boxcolor=black@0.8:boxborderw=8:"
            f"enable='between(n\\,{start_t}\\,{end_t})'"
        )
    return ",".join(parts)


def add_chunk_overlay(
    *,
    rows_path: Path,
    video_path: Path,
    total_episode_idx: int,
    output_path: Path | None = None,
) -> Path:
    line = _extract_episode_line(rows_path, total_episode_idx)
    segments = _chunk_segments_from_line(line)

    if output_path is None:
        output_path = video_path.with_name(video_path.stem + "_chunk_overlay.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vf = _drawtext_filter(segments)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


if __name__ == "__main__":
    base = Path(
        "/home/motovilovil/Robotics/robotics_project/mimic-video/eval_outputs/libero_spatial/decoder_hidden_metrics/"
        "task0_init9_10rollouts_6sec_layers3_7_11_15_19_23_with_videos"
    )
    rows_path = base / "metrics" / "episode_traces.jsonl"
    videos = [
        (
            7,
            base / "videos" / "episode7_success_pick_up_the_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.mp4",
        ),
        (
            8,
            base / "videos" / "episode8_failure_pick_up_the_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.mp4",
        ),
    ]
    for episode_idx, video_path in videos:
        out = add_chunk_overlay(rows_path=rows_path, video_path=video_path, total_episode_idx=episode_idx)
        print(out)
