"""FastAPI backend for the Haiku Playground, deployed on Modal.

modal deploy eval.haiku_app
"""

from pathlib import Path

import modal

app = modal.App("haiku-playground")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "httpx", "nltk")
    .run_commands(
        "python -c \"import nltk; nltk.download('cmudict')\""
    )
    .add_local_dir("eval", "/root/eval")
    .add_local_dir("llm_judges", "/root/llm_judges")
    .add_local_file("config.py", "/root/config.py")
)


@app.function(
    image=image,
)
@modal.asgi_app(
    custom_domains=["haiku.modal.dev"]
)
def serve_playground():
    from contextlib import asynccontextmanager
    from time import perf_counter

    import httpx
    import nltk
    from fastapi import FastAPI
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    from eval.shared import (
        MODAL_VOCABS,
        MODEL_CHECKPOINTS,
        QUICK_PROMPTS,
        build_system_prompt,
        get_model_endpoint,
        iter_dirs,
        query_model,
    )
    from llm_judges.nlp import (
        count_syllables_for_word,
        score_haiku_structure,
        segment_haiku_lines,
    )

    class GenerateRequest(BaseModel):
        prompt: str
        model_key: str = "base-model"
        iter_num: str = "50"
        include_vocab: bool = True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cmudict = nltk.corpus.cmudict.dict()
        app.state.http_client = httpx.AsyncClient()
        tree_script = Path("/root/eval/haiku_tree.js")
        tree_stat = tree_script.stat()
        app.state.asset_version = f"{int(tree_stat.st_mtime)}-{tree_stat.st_size}"
        yield
        await app.state.http_client.aclose()

    fastapi_app = FastAPI(title="Haiku Playground", lifespan=lifespan)
    fastapi_app.add_middleware(GZipMiddleware, minimum_size=500)

    html_path = Path("/root/eval/haiku_playground.html")
    tree_script_path = Path("/root/eval/haiku_tree.js")
    immutable_cache = "public, max-age=31536000, immutable"
    html_cache = "no-cache"

    def timed_file_response(path: Path, *, media_type: str | None, cache_control: str) -> FileResponse:
        start = perf_counter()
        response = FileResponse(path, media_type=media_type, headers={"Cache-Control": cache_control})
        duration_ms = (perf_counter() - start) * 1000
        response.headers["Server-Timing"] = f"app;dur={duration_ms:.2f}"
        return response

    @fastapi_app.post("/api/generate")
    async def generate(request: GenerateRequest):
        import re

        client = fastapi_app.state.http_client
        cmudict = fastapi_app.state.cmudict

        iter_num = request.iter_num if request.iter_num != "base" else "50"
        endpoint = get_model_endpoint(request.model_key, iter_num)
        system_prompt = build_system_prompt(include_vocab=request.include_vocab)

        haiku = await query_model(
            client,
            endpoint,
            request.prompt,
            model_name=request.model_key,
            system_prompt=system_prompt,
        )

        structure_score = score_haiku_structure(haiku, cmudict)

        lines = segment_haiku_lines(haiku)
        syllable_counts = []
        for line in lines:
            words = re.findall(r"[a-zA-Z]+", line)
            count = sum(count_syllables_for_word(w, cmudict) for w in words)
            syllable_counts.append(count)

        return {
            "haiku": haiku,
            "structure_score": structure_score,
            "syllable_counts": syllable_counts,
            "passed": structure_score == 1,
        }

    @fastapi_app.get("/api/models")
    async def get_models():
        return MODEL_CHECKPOINTS

    @fastapi_app.get("/api/vocabs")
    async def get_vocabs():
        return MODAL_VOCABS

    @fastapi_app.get("/api/iter_nums")
    async def get_iter_nums():
        return sorted(iter_dirs.keys(), key=int)

    @fastapi_app.get("/api/bootstrap")
    async def get_bootstrap():
        return {
            "models": MODEL_CHECKPOINTS,
            "vocabs": MODAL_VOCABS,
            "iter_nums": sorted(iter_dirs.keys(), key=int),
            "quick_prompts": QUICK_PROMPTS,
            "asset_version": fastapi_app.state.asset_version,
        }

    @fastapi_app.get("/assets/{filename:path}")
    async def serve_asset(filename: str):
        asset_path = Path("/root/eval/assets") / filename
        return timed_file_response(
            asset_path,
            media_type=None,
            cache_control=immutable_cache,
        )

    @fastapi_app.get("/haiku_tree.js")
    async def serve_tree_script():
        return timed_file_response(
            tree_script_path,
            media_type="application/javascript",
            cache_control=immutable_cache,
        )

    @fastapi_app.get("/haiku_tree.js")
    async def serve_tree_script():
        script_path = Path("/root/eval/haiku_tree.js")
        return FileResponse(script_path, media_type="application/javascript")

    @fastapi_app.get("/")
    async def index():
        return timed_file_response(
            html_path,
            media_type="text/html",
            cache_control=html_cache,
        )

    return fastapi_app
