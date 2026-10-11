"""FASHN AI + Shopify Admin MCP server for Diriava.

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
MCP_PATH = os.environ.get("MCP_PATH", "").strip()
if not MCP_PATH.startswith("/mcp-") or len(MCP_PATH) < 5 + 24:
    raise SystemExit(
        "Refusing to start: set MCP_PATH to a secret path like /mcp-<32 random characters>. "
        "The default /mcp would expose the store and FASHN credits to anyone."
    )
PENDING = {"starting", "in_queue", "processing"}

mcp = FastMCP(
    name="diriava",
    instructions=(
        "Tools for the FASHN AI image API. Use product_to_model to turn a product "
        "photo (flat-lay or ghost mannequin) into an on-model photo, try_on to dress "
        "an existing model photo in a garment, and get_status to fetch results of a "
        "job that is still running. Output URLs stay valid for about 3 days: import "
        "them into Shopify straight away. Each output image costs FASHN credits. "
        "shopify_graphql runs any Shopify Admin GraphQL query or mutation on the "
        "Diriava store; shopify_granted_scopes lists what the app is allowed to do."
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


# ---------------------------------------------------------------------------
# Shopify Admin GraphQL (Dev Dashboard app, client credentials grant)
#   SHOPIFY_SHOP            e.g. t23ayj-0p.myshopify.com
#   SHOPIFY_CLIENT_ID       from Dev Dashboard > App settings
#   SHOPIFY_CLIENT_SECRET   from Dev Dashboard > App settings
#   SHOPIFY_API_VERSION     optional, default 2026-07
# Tokens last 24 h; they are fetched and refreshed automatically.
# ---------------------------------------------------------------------------
import time

SHOP = os.environ.get("SHOPIFY_SHOP", "")
SHOP_CLIENT_ID = os.environ.get("SHOPIFY_CLIENT_ID", "")
SHOP_CLIENT_SECRET = os.environ.get("SHOPIFY_CLIENT_SECRET", "")
SHOP_API_VERSION = os.environ.get("SHOPIFY_API_VERSION", "2026-07")
_shop_token = {"value": None, "expires_at": 0.0}
_shop_lock = asyncio.Lock()


async def _shopify_token() -> str:
    if not (SHOP and SHOP_CLIENT_ID and SHOP_CLIENT_SECRET):
        raise RuntimeError("SHOPIFY_SHOP, SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET must be set on the server.")
    async with _shop_lock:
        if _shop_token["value"] and time.time() < _shop_token["expires_at"] - 300:
            return _shop_token["value"]
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"https://{SHOP}/admin/oauth/access_token",
                data={"grant_type": "client_credentials",
                      "client_id": SHOP_CLIENT_ID, "client_secret": SHOP_CLIENT_SECRET},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if r.status_code >= 400:
            raise RuntimeError(f"Shopify token error {r.status_code}: {r.text[:300]}")
        data = r.json()
        _shop_token["value"] = data["access_token"]
        _shop_token["expires_at"] = time.time() + int(data.get("expires_in", 86399))
        return _shop_token["value"]


async def _shopify_gql(query: str, variables: Optional[dict] = None) -> dict:
    token = await _shopify_token()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"https://{SHOP}/admin/api/{SHOP_API_VERSION}/graphql.json",
            json={"query": query, "variables": variables or {}},
            headers={"X-Shopify-Access-Token": token, "Content-Type": "application/json"},
        )
    if r.status_code == 401:
        _shop_token["value"] = None
    if r.status_code >= 400:
        raise RuntimeError(f"Shopify GraphQL error {r.status_code}: {r.text[:1000]}")
    return r.json()


@mcp.tool
async def shopify_graphql(
    query: Annotated[str, Field(description="A GraphQL Admin API query or mutation.")],
    variables: Annotated[Optional[dict], Field(description="GraphQL variables.")] = None,
) -> dict:
    """Run any Shopify Admin GraphQL query or mutation on the Diriava store (full app scopes).
    Returns the raw response including 'errors', 'userErrors' and cost 'extensions'."""
    return await _shopify_gql(query, variables)


@mcp.tool
async def shopify_granted_scopes() -> dict:
    """List the access scopes the Shopify app actually has on the store."""
    return await _shopify_gql("{ currentAppInstallation { accessScopes { handle } } }")


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0",
            port=int(os.environ.get("PORT", "8000")), path=MCP_PATH)
