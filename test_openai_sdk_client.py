#!/usr/bin/env python3
"""
OpenAI SDK client example for openai_compatible_api.py

Supports:
- Single-view generation (one image)
- Multi-view generation (front/back/left/right)
- Download of generated mesh files via /v1/files/{id}/content
"""

import argparse
import base64
import os
from pathlib import Path
from typing import Dict, Optional

import requests
from openai import OpenAI


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as file_handle:
        return base64.b64encode(file_handle.read()).decode("utf-8")


def build_multiview_payload(
    front: Optional[str],
    back: Optional[str],
    left: Optional[str],
    right: Optional[str],
) -> Optional[Dict[str, str]]:
    payload = {}
    if front:
        payload["front"] = encode_image_to_base64(front)
    if back:
        payload["back"] = encode_image_to_base64(back)
    if left:
        payload["left"] = encode_image_to_base64(left)
    if right:
        payload["right"] = encode_image_to_base64(right)
    return payload if payload else None


def extract_files_from_metadata(response_obj) -> list:
    metadata = getattr(response_obj, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("files", [])

    if hasattr(metadata, "model_dump"):
        metadata_dict = metadata.model_dump()
        return metadata_dict.get("files", [])

    return []


def download_generated_files(files: list, base_url: str, api_key: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    for file_record in files:
        file_id = file_record.get("id")
        filename = file_record.get("filename", f"{file_id}.bin")
        url = f"{base_url.rstrip('/')}/files/{file_id}/content"

        response = requests.get(url, headers=headers, timeout=180)
        response.raise_for_status()

        save_path = os.path.join(output_dir, filename)
        with open(save_path, "wb") as output_handle:
            output_handle.write(response.content)

        print(f"Downloaded: {save_path}")


def run_single_view(client: OpenAI, args) -> None:
    if not args.image:
        raise ValueError("Single-view mode requires --image")

    image_b64 = encode_image_to_base64(args.image)

    response = client.responses.create(
        model=args.model,
        input=args.prompt,
        extra_body={
            "image_base64": image_b64,
            "texture": args.texture,
            "remove_background": args.remove_background,
            "seed": args.seed,
            "randomize_seed": args.randomize_seed,
            "octree_resolution": args.octree_resolution,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "num_chunks": args.num_chunks,
            "face_count": args.face_count,
            "output_format": args.output_format,
            "simplify_mesh": args.simplify_mesh,
            "target_face_count": args.target_face_count,
        },
    )

    print(f"Response ID: {response.id}")
    print(f"Status: {response.status}")

    files = extract_files_from_metadata(response)
    if not files:
        print("No files found in response metadata")
        return

    download_generated_files(files, args.base_url, args.api_key, args.output_dir)


def run_multiview(client: OpenAI, args) -> None:
    mv_payload = build_multiview_payload(
        front=args.front,
        back=args.back,
        left=args.left,
        right=args.right,
    )
    if not mv_payload:
        raise ValueError("Multi-view mode requires at least one of --front/--back/--left/--right")

    response = client.responses.create(
        model=args.model,
        input=args.prompt,
        extra_body={
            "multiview_images": mv_payload,
            "texture": args.texture,
            "remove_background": args.remove_background,
            "seed": args.seed,
            "randomize_seed": args.randomize_seed,
            "octree_resolution": args.octree_resolution,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "num_chunks": args.num_chunks,
            "face_count": args.face_count,
            "output_format": args.output_format,
            "simplify_mesh": args.simplify_mesh,
            "target_face_count": args.target_face_count,
        },
    )

    print(f"Response ID: {response.id}")
    print(f"Status: {response.status}")

    files = extract_files_from_metadata(response)
    if not files:
        print("No files found in response metadata")
        return

    download_generated_files(files, args.base_url, args.api_key, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-url", default="http://localhost:8082/v1")
    parser.add_argument("--api-key", default=os.getenv("HY3D_API_KEY", ""))
    parser.add_argument("--model", default="hunyuan3d-2.1")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-dir", default="./downloads")

    parser.add_argument("--mode", choices=["single", "multiview"], required=True)

    parser.add_argument("--image", help="Single-view input image path")

    parser.add_argument("--front", help="Multiview front image path")
    parser.add_argument("--back", help="Multiview back image path")
    parser.add_argument("--left", help="Multiview left image path")
    parser.add_argument("--right", help="Multiview right image path")

    parser.add_argument("--texture", action="store_true")
    parser.add_argument("--remove-background", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--randomize-seed", action="store_true")
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--face-count", type=int, default=40000)
    parser.add_argument("--output-format", choices=["glb", "obj", "ply", "stl"], default="glb")
    parser.add_argument("--simplify-mesh", action="store_true")
    parser.add_argument("--target-face-count", type=int, default=10000)

    return parser.parse_args()


def main():
    args = parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.mode == "single":
        run_single_view(client, args)
    else:
        run_multiview(client, args)


if __name__ == "__main__":
    main()
