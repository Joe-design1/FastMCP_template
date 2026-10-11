"""FASHN AI MCP server for Diriava.

Exposes the FASHN API (https://docs.fashn.ai) as MCP tools so Claude can
generate on-model product images without ever seeing the API key.

Environment variables (set them in Railway, never in code):
  FASHN_API_KEY   your FASHN API key (required)
  MCP_PATH        secret URL path, e.g. /mcp-7f3c9a1e5b2d4f60 (required)
  PORT            set automatically by Railway
"""

import asyncio
import os
from typing import Annotated, Literal, Optional

import httpx
from fastmcp import FastMCP
from pydantic import Field

FASHN_BASE = "https://api.fashn.ai/v1"
API_KEY = os.environ.get("FASHN_API_KEY", "")
MCP_PATH = os.environ.get("MCP_PATH", "/mcp")
PENDING = {"starting", "in_queue", "processing"}

mcp = FastMCP(
    name="fashn",
    instructions=(
        "Tools for the FASHN AI image API. Use product_to_model to turn a product "
        "photo (flat-lay or ghost mannequin) into an on-model photo, try_on to dress "
        "an existing model photo in a garment, and get_status to fetch results of a "
        "job that is still running. Output URLs stay valid for about 3 days: import "
        "them into Shopify straight away. Each output image costs FASHN credits."
    ),
)


def _headers() -> dict:
    if not API_KEY:
        raise RuntimeError("FASHN_API_KEY is not set on the server.")
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


async def _submit(model_name: str, inputs: dict) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{FASHN_BASE}/run", headers=_headers(),
                              json={"model_name": model_name, "inputs": inputs})
    if r.status_code >= 400:
        raise RuntimeError(f"FASHN /run error {r.status_code}: {r.text[:500]}")
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"FASHN /run error: {data['error']}")
    return data["id"]


async def _status(prediction_id: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{FASHN_BASE}/status/{prediction_id}", headers=_headers())
    if r.status_code >= 400:
        raise RuntimeError(f"FASHN /status error {r.status_code}: {r.text[:500]}")
    return r.json()


async def _run_and_wait(model_name: str, inputs: dict, wait_seconds: int) -> dict:
    """Submit a job and poll every 3 s until done or until wait_seconds elapse."""
    prediction_id = await _submit(model_name, inputs)
    waited = 0
    while True:
        data = await _status(prediction_id)
        status = data.get("status")
        if status not in PENDING:
            return {"id": prediction_id, "status": status,
                    "output": data.get("output") or [], "error": data.get("error")}
        if waited >= wait_seconds:
            return {"id": prediction_id, "status": status, "output": [], "error": None,
                    "note": "Still running. Call get_status with this id in a moment."}
        await asyncio.sleep(3)
        waited += 3


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


AspectRatio = Literal["1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "4:5", "5:4"]


@mcp.tool
async def product_to_model(
    product_image: Annotated[str, Field(description="Public URL (or data URI) of the product photo, e.g. the Shopify CDN image URL.")],
    prompt: Annotated[Optional[str], Field(description="Model appearance, styling and background, e.g. 'woman in her 30s, neutral studio background, autumn styling'.")] = None,
    image_prompt: Annotated[Optional[str], Field(description="Optional inspiration image URL for pose, setting and lighting.")] = None,
    background_reference: Annotated[Optional[str], Field(description="Optional backdrop image URL.")] = None,
    face_reference: Annotated[Optional[str], Field(description="Optional face/identity reference URL (+3 credits per output).")] = None,
    aspect_ratio: Annotated[Optional[AspectRatio], Field(description="Output ratio. Diriava product cards use 3:4.")] = "3:4",
    resolution: Annotated[Literal["1k", "2k", "4k"], Field(description="Output resolution.")] = "2k",
    generation_mode: Annotated[Optional[Literal["fast", "balanced", "quality"]], Field(description="Speed/quality trade-off. Omit to let FASHN choose.")] = None,
    num_images: Annotated[int, Field(ge=1, le=4, description="Number of images (each costs credits).")] = 1,
    seed: Annotated[Optional[int], Field(ge=0, le=4294967295)] = None,
    output_format: Literal["png", "jpeg"] = "jpeg",
    wait_seconds: Annotated[int, Field(ge=0, le=240, description="How long to wait for the result before returning the job id.")] = 120,
) -> dict:
    """Generate on-model photos from a product image (FASHN 'product-to-model')."""
    inputs = _clean({
        "product_image": product_image, "prompt": prompt, "image_prompt": image_prompt,
        "background_reference": background_reference, "face_reference": face_reference,
        "aspect_ratio": aspect_ratio, "resolution": resolution,
        "generation_mode": generation_mode, "num_images": num_images, "seed": seed,
        "output_format": output_format, "return_base64": False,
    })
    return await _run_and_wait("product-to-model", inputs, wait_seconds)


@mcp.tool
async def try_on(
    model_image: Annotated[str, Field(description="Public URL of the person/model photo.")],
    garment_image: Annotated[str, Field(description="Public URL of the garment photo.")],
    category: Literal["auto", "tops", "bottoms", "one-pieces"] = "auto",
    garment_photo_type: Literal["auto", "flat-lay", "model"] = "auto",
    mode: Literal["performance", "balanced", "quality"] = "quality",
    num_samples: Annotated[int, Field(ge=1, le=4)] = 1,
    seed: Annotated[Optional[int], Field(ge=0, le=4294967295)] = None,
    output_format: Literal["png", "jpeg"] = "jpeg",
    wait_seconds: Annotated[int, Field(ge=0, le=240)] = 120,
) -> dict:
    """Dress a model photo in a garment (FASHN virtual try-on v1.6)."""
    inputs = _clean({
        "model_image": model_image, "garment_image": garment_image, "category": category,
        "garment_photo_type": garment_photo_type, "mode": mode, "num_samples": num_samples,
        "seed": seed, "output_format": output_format, "return_base64": False,
    })
    return await _run_and_wait("tryon-v1.6", inputs, wait_seconds)


@mcp.tool
async def run_model(
    model_name: Annotated[str, Field(description="Any FASHN model name, e.g. 'model-swap', 'edit', 'model-create'.")],
    inputs: Annotated[dict, Field(description="The 'inputs' object exactly as documented on docs.fashn.ai for that model.")],
    wait_seconds: Annotated[int, Field(ge=0, le=240)] = 120,
) -> dict:
    """Run any other FASHN endpoint with raw inputs (advanced)."""
    inputs = dict(inputs)
    inputs["return_base64"] = False
    return await _run_and_wait(model_name, inputs, wait_seconds)


@mcp.tool
async def get_status(
    prediction_id: Annotated[str, Field(description="The id returned by a previous call.")],
) -> dict:
    """Get the status and output URLs of a FASHN job."""
    data = await _status(prediction_id)
    return {"id": prediction_id, "status": data.get("status"),
            "output": data.get("output") or [], "error": data.get("error")}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0",
            port=int(os.environ.get("PORT", "8000")), path=MCP_PATH,
            stateless_http=True)
