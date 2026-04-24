#!/usr/bin/env python3
"""
OpenAI SDK-compatible API server for Hunyuan3D-2.1.

Features:
- Single-view image-to-3D generation
- Multi-view (front/back/left/right) image-to-3D generation
- Optional texture generation (PBR)
- Mesh export and download endpoints
- Bearer API key authentication

OpenAI Python SDK usage example:

    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8082/v1",
        api_key="your-api-key"
    )

    resp = client.responses.create(
        model="hunyuan3d-2.1",
        input="Generate a toy car",
        extra_body={
            "image_base64": "<base64_png>",
            "texture": True,
            "num_inference_steps": 5,
            "guidance_scale": 5.0
        }
    )

    print(resp.id)
    print(resp.status)
"""

import argparse
import base64
import os
import random
import shutil
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
import trimesh
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

try:
    from torchvision_fix import apply_fix

    apply_fix()
except Exception as exc:
    print(f"Warning: torchvision fix not applied: {exc}")

from hy3dshape import FaceReducer, Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from hy3dpaint.convert_utils import create_glb_with_pbr_materials
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline


MAX_SEED = 10_000_000
DEFAULT_MODEL_NAME = "hunyuan3d-2.1"

app = FastAPI(title="Hunyuan3D OpenAI-Compatible API", version="1.0.0")

STATE: Dict[str, object] = {
    "api_key": None,
    "save_dir": "./openai_cache",
    "shape_pipeline": None,
    "t2i_pipeline": None,
    "rmbg_worker": None,
    "face_reduce_worker": None,
    "tex_pipeline": None,
    "tasks": {},
    "files": {},
}


class MultiViewImages(BaseModel):
    front: Optional[str] = None
    back: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None


class OpenAI3DRequest(BaseModel):
    model: str = Field(DEFAULT_MODEL_NAME, description="Model name")
    input: Optional[str] = Field(None, description="Optional text prompt")

    image_base64: Optional[str] = Field(None, description="Single-view image, base64")
    multiview_images: Optional[MultiViewImages] = Field(
        None,
        description="Multi-view images (front/back/left/right), each value base64-encoded",
    )

    remove_background: bool = Field(True)
    texture: bool = Field(False)
    seed: int = Field(1234, ge=0, le=2**32 - 1)
    randomize_seed: bool = Field(False)
    octree_resolution: int = Field(256, ge=64, le=512)
    num_inference_steps: int = Field(5, ge=1, le=100)
    guidance_scale: float = Field(5.0, ge=0.1, le=20.0)
    num_chunks: int = Field(8000, ge=1000, le=5_000_000)
    face_count: int = Field(40000, ge=1000, le=1_000_000)

    output_format: Literal["glb", "obj", "ply", "stl"] = "glb"
    simplify_mesh: bool = False
    target_face_count: int = Field(10000, ge=100, le=1_000_000)


class OpenAIResponseObject(BaseModel):
    id: str
    object: str = "response"
    created: int
    model: str
    status: Literal["completed", "failed"]
    output: List[dict]
    error: Optional[dict] = None
    metadata: dict


def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    configured_key = STATE.get("api_key")
    if not configured_key:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer API key")

    token = authorization.split(" ", 1)[1].strip()
    if token != configured_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _decode_image(image_base64: str):
    return trimesh.util.wrap_as_stream(base64.b64decode(image_base64))


def _load_pil_from_base64(image_b64: str):
    from PIL import Image

    return Image.open(BytesIO(base64.b64decode(image_b64)))


def _gen_save_folder(max_size: int = 200) -> str:
    save_dir = STATE["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    dirs = [path for path in Path(save_dir).iterdir() if path.is_dir()]
    if len(dirs) >= max_size:
        oldest = min(dirs, key=lambda p: p.stat().st_ctime)
        shutil.rmtree(oldest, ignore_errors=True)
    new_dir = os.path.join(save_dir, str(uuid.uuid4()))
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def _export_mesh(mesh, save_folder: str, textured: bool, fmt: str) -> str:
    filename = "textured_mesh" if textured else "white_mesh"
    out_path = os.path.join(save_folder, f"{filename}.{fmt}")
    if fmt in {"glb", "obj"}:
        mesh.export(out_path, include_normals=textured)
    else:
        mesh.export(out_path)
    return out_path


def _quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> str:
    textures = {
        "albedo": obj_path.replace(".obj", ".jpg"),
        "metallic": obj_path.replace(".obj", "_metallic.jpg"),
        "roughness": obj_path.replace(".obj", "_roughness.jpg"),
    }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)
    return glb_path


def _register_file(task_id: str, file_path: str, kind: str) -> dict:
    file_id = f"file_{uuid.uuid4().hex}"
    filename = os.path.basename(file_path)
    file_rec = {
        "id": file_id,
        "object": "file",
        "task_id": task_id,
        "kind": kind,
        "filename": filename,
        "bytes": os.path.getsize(file_path),
        "created_at": int(time.time()),
        "path": file_path,
        "download_url": f"/v1/files/{file_id}/content",
    }
    STATE["files"][file_id] = file_rec
    return file_rec


def _prepare_generation_input(payload: OpenAI3DRequest):
    prompt = payload.input
    single_image = None
    multiview = None

    if payload.image_base64:
        single_image = _load_pil_from_base64(payload.image_base64)

    if payload.multiview_images:
        mv = {}
        if payload.multiview_images.front:
            mv["front"] = _load_pil_from_base64(payload.multiview_images.front)
        if payload.multiview_images.back:
            mv["back"] = _load_pil_from_base64(payload.multiview_images.back)
        if payload.multiview_images.left:
            mv["left"] = _load_pil_from_base64(payload.multiview_images.left)
        if payload.multiview_images.right:
            mv["right"] = _load_pil_from_base64(payload.multiview_images.right)
        if mv:
            multiview = mv

    if single_image is None and multiview is None and prompt is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: input prompt, image_base64, multiview_images",
        )

    if single_image is None and multiview is None and prompt:
        t2i = STATE.get("t2i_pipeline")
        if t2i is None:
            raise HTTPException(
                status_code=400,
                detail="Text-only generation requires --enable_t23d",
            )
        single_image = t2i(prompt)

    return prompt, single_image, multiview


def _apply_background_removal(single_image, multiview, remove_background: bool):
    rmbg = STATE["rmbg_worker"]
    if multiview:
        out = {}
        for view_name, img in multiview.items():
            if remove_background or img.mode == "RGB":
                out[view_name] = rmbg(img.convert("RGB"))
            else:
                out[view_name] = img
        return single_image, out

    if single_image is not None:
        if remove_background or single_image.mode == "RGB":
            single_image = rmbg(single_image.convert("RGB"))
    return single_image, multiview


def _run_generation(payload: OpenAI3DRequest, task_id: str) -> dict:
    start = time.time()
    save_folder = _gen_save_folder()

    _, single_image, multiview = _prepare_generation_input(payload)
    single_image, multiview = _apply_background_removal(
        single_image,
        multiview,
        payload.remove_background,
    )

    seed = payload.seed
    if payload.randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(int(seed))

    image_for_shape = multiview if multiview is not None else single_image

    outputs = STATE["shape_pipeline"](
        image=image_for_shape,
        num_inference_steps=payload.num_inference_steps,
        guidance_scale=payload.guidance_scale,
        generator=generator,
        octree_resolution=payload.octree_resolution,
        num_chunks=payload.num_chunks,
        output_type="mesh",
    )

    mesh = export_to_trimesh(outputs)[0]

    if payload.simplify_mesh:
        mesh = STATE["face_reduce_worker"](mesh, payload.target_face_count)

    files = []

    base_mesh_path = _export_mesh(mesh, save_folder, textured=False, fmt=payload.output_format)
    files.append(_register_file(task_id, base_mesh_path, kind="mesh_base"))

    if payload.texture:
        tex_pipeline = STATE.get("tex_pipeline")
        if tex_pipeline is None:
            raise RuntimeError("Texture generation not available. Start without --disable_tex.")

        mesh_for_tex = STATE["face_reduce_worker"](mesh, payload.face_count)
        mesh_for_tex_path = _export_mesh(mesh_for_tex, save_folder, textured=False, fmt="obj")

        if single_image is not None:
            source_image = single_image
        else:
            source_image = multiview.get("front") if multiview else None
            if source_image is None and multiview:
                source_image = next(iter(multiview.values()))

        if source_image is None:
            raise RuntimeError("Texture generation requires at least one input image")
        textured_obj_path = os.path.join(save_folder, "textured_mesh.obj")

        textured_obj_path = tex_pipeline(
            mesh_path=mesh_for_tex_path,
            image_path=source_image,
            output_mesh_path=textured_obj_path,
            save_glb=False,
        )

        files.append(_register_file(task_id, textured_obj_path, kind="mesh_textured_obj"))

        textured_glb_path = os.path.join(save_folder, "textured_mesh.glb")
        _quick_convert_with_obj2gltf(textured_obj_path, textured_glb_path)
        files.append(_register_file(task_id, textured_glb_path, kind="mesh_textured_glb"))

    if STATE.get("low_vram_mode"):
        torch.cuda.empty_cache()

    elapsed = time.time() - start
    return {
        "task_id": task_id,
        "status": "completed",
        "seed": seed,
        "save_folder": save_folder,
        "duration_sec": elapsed,
        "files": files,
        "stats": {
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
        },
    }


@app.get("/v1/models")
def list_models(_: None = Depends(_check_auth)):
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tencent-hunyuan",
            }
        ],
    }


@app.post("/v1/responses", response_model=OpenAIResponseObject)
def create_response(payload: OpenAI3DRequest, _: None = Depends(_check_auth)):
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        result = _run_generation(payload, task_id=response_id)
        STATE["tasks"][response_id] = result

        files = result["files"]
        file_lines = [f"- {f['kind']}: {f['download_url']}" for f in files]
        output_text = "Generation completed. Download files:\n" + "\n".join(file_lines)

        return {
            "id": response_id,
            "object": "response",
            "created": created,
            "model": payload.model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            ],
            "error": None,
            "metadata": result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Generation failed")
        return JSONResponse(
            status_code=500,
            content={
                "id": response_id,
                "object": "response",
                "created": created,
                "model": payload.model,
                "status": "failed",
                "output": [],
                "error": {"message": str(exc), "type": "generation_error"},
                "metadata": {},
            },
        )


@app.get("/v1/responses/{response_id}")
def get_response(response_id: str, _: None = Depends(_check_auth)):
    task = STATE["tasks"].get(response_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Response not found")

    return {
        "id": response_id,
        "object": "response",
        "created": int(time.time()),
        "model": DEFAULT_MODEL_NAME,
        "status": task["status"],
        "output": [],
        "error": None,
        "metadata": task,
    }


@app.get("/v1/files/{file_id}")
def get_file(file_id: str, _: None = Depends(_check_auth)):
    file_rec = STATE["files"].get(file_id)
    if file_rec is None:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "id": file_rec["id"],
        "object": "file",
        "bytes": file_rec["bytes"],
        "created_at": file_rec["created_at"],
        "filename": file_rec["filename"],
        "purpose": "3d_generation_output",
    }


@app.get("/v1/files/{file_id}/content")
def download_file(file_id: str, _: None = Depends(_check_auth)):
    file_rec = STATE["files"].get(file_id)
    if file_rec is None:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = file_rec["path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File content no longer available")

    return FileResponse(path=file_path, filename=file_rec["filename"])


@app.get("/health")
def health():
    return {"status": "healthy"}


def _bootstrap(args):
    STATE["api_key"] = args.api_key
    STATE["save_dir"] = args.cache_path
    STATE["low_vram_mode"] = args.low_vram_mode

    os.makedirs(args.cache_path, exist_ok=True)

    logger.info("Loading workers...")

    STATE["rmbg_worker"] = BackgroundRemover()
    STATE["shape_pipeline"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=False,
        device=args.device,
    )

    if args.enable_flashvdm:
        mc_algo = "mc" if args.device in ["cpu", "mps"] else args.mc_algo
        STATE["shape_pipeline"].enable_flashvdm(mc_algo=mc_algo)

    if args.compile:
        STATE["shape_pipeline"].compile()

    STATE["face_reduce_worker"] = FaceReducer()

    if not args.disable_tex:
        conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        STATE["tex_pipeline"] = Hunyuan3DPaintPipeline(conf)

    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline

        STATE["t2i_pipeline"] = HunyuanDiTPipeline(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
        )

    logger.info("Bootstrap completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)

    parser.add_argument("--api-key", type=str, default=os.getenv("HY3D_API_KEY"))

    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mc_algo", type=str, default="mc")

    parser.add_argument("--cache-path", type=str, default="./openai_cache")
    parser.add_argument("--enable_t23d", action="store_true")
    parser.add_argument("--disable_tex", action="store_true")
    parser.add_argument("--enable_flashvdm", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--low_vram_mode", action="store_true")

    args = parser.parse_args()
    _bootstrap(args)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
