import marimo

__generated_with = "0.23.2"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ComfyUI on Modal — Notebook Edition

    This runs ComfyUI **interactively inside a Modal Notebook kernel**, instead of behind
    Modal's `@modal.web_server` reverse proxy. That proxy is what mangled the WebSocket
    traffic in the earlier deployment — server→client frames, including image previews,
    never arrived, no matter what changed on the ComfyUI/custom-node side. Running ComfyUI
    as a plain subprocess in a live kernel and exposing it with a **Cloudflare Tunnel**
    instead of Modal's own ingress sidesteps that layer completely: `cloudflared` makes its
    own outbound connection to Cloudflare's edge, so nothing routes through the code path
    that was dropping frames before.

    (Modal Notebooks don't yet support Modal's own port-forwarding `Tunnels` either — it's
    listed on their public roadmap as not-yet-built — so `cloudflared` isn't just a
    workaround for the old bug, it's currently the only way to view a live UI from a
    notebook session at all.)

    ## One-time setup, before running this notebook

    1. From your machine, run `modal deploy comfyui.py` (the file you already have) once,
       or whenever you change models/custom nodes. This builds and caches the big custom
       image and keeps `SAM3Service` deployed, so the `SAM3BoxDetect` node keeps working
       exactly as before. You do **not** need to edit `comfyui.py` — the `@modal.web_server`
       endpoint it deploys just goes unused from here on.
    2. Open this notebook at [modal.com/notebooks](https://modal.com/notebooks) (create a
       notebook, then upload this `.ipynb`).
    3. In the sidebar **compute profile**, set GPU to **A10G** (or the closest available
       type — same as the deployed app used).
    4. In the sidebar's **image search**, search for `modal-comfyui` and attach the
       `ComfyUI` function's image (not `SAM3Service` — that one's a separate, smaller
       image). This gives the kernel the same custom nodes, `models.py`, `plugins.py`, etc.
       you already built — nothing below reinstalls any of that.
    5. In the sidebar's **Files panel**, attach your existing Volumes: `hf-hub-cache`,
       `comfyui-loras`, `comfyui-inputs`. Attach the `huggingface-secret` Secret too.

    Then just run the cells top to bottom. Cell 2 lists what actually landed under `/mnt` —
    Notebooks always mount Volumes there (unlike the deployed app, which mounted them at
    `/cache` / `/loras_vol` / etc.), so double-check the folder names match Cell 3 before
    proceeding, and edit Cell 3 if your workspace names differ.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Confirm the kernel actually has what you attached

    GPU, and whatever landed under `/mnt` from the Volumes you attached in the sidebar.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Path configuration

    **Edit the three paths below** if the listing above showed different folder names
    under `/mnt`. These stand in for the deployed app's volume mounts
    (`/cache`, `/loras_vol`, `/root/comfy/ComfyUI/input`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model sync helpers

    Same logic as `comfyui.py`'s `download_all()` / `link_user_models()` / `clean_output_dir()`,
    just pointed at `/mnt/...` instead of the old `/cache` and `/loras_vol` mount points.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Process management

    Same `_is_port_open` / `wait_for_comfy_ready` / `_launch_comfy` logic as `comfyui.py`,
    plus the health-check watchdog from the last debugging session (self-heals if ComfyUI
    wedges mid-session instead of silently dying for the rest of the kernel's life).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Sync models and launch ComfyUI

    This can take a while on a cold kernel (copying models from the volume onto local
    SSD) but is fast on a second run in the same session.
    """)
    return


@app.cell
def _():
    import os
    import shutil
    import subprocess
    import sys
    import logging
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor

    # --- 1. HF Auth & Speed Flags ---
    HF_KEY = "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"
    os.environ["HF_TOKEN"] = HF_KEY
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_KEY
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"

    _hf_token_file = Path.home() / ".cache" / "huggingface" / "token"
    _hf_token_file.parent.mkdir(parents=True, exist_ok=True)
    _hf_token_file.write_text(HF_KEY.strip())
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

    from huggingface_hub import hf_hub_download, snapshot_download

    def run_setup():
        COMFY_DIR = Path("/home/marimo/comfy/ComfyUI")
        MODELS_ROOT = COMFY_DIR / "models"
        NODES_DIR = COMFY_DIR / "custom_nodes"
        CACHE_DIR = Path("/mnt/hf-hub-cache") if Path("/mnt/hf-hub-cache").exists() else Path("/cache")
        INPUTS_SRC = Path("/mnt/comfyui-inputs")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        COMFY_DIR.mkdir(parents=True, exist_ok=True)
        NODES_DIR.mkdir(parents=True, exist_ok=True)

        def cmd(c, cwd=None):
            subprocess.run(c, shell=True, check=False, cwd=cwd)

        # 1. Inputs Volume
        comfy_input = COMFY_DIR / "input"
        if INPUTS_SRC.exists() and not comfy_input.exists():
            comfy_input.symlink_to(INPUTS_SRC)

        # 2. Skip Apt/Pip if already installed
        print("--- 1/4: Checking Base Environment ---")
        if not shutil.which("aria2c"):
            cmd("apt-get update -qq && apt-get install -y -qq git git-lfs libgl1-mesa-dev libglib2.0-0 aria2 ffmpeg build-essential python3-dev psmisc")

        try:
            import sageattention, comfy_kitchen, diffusers, ultralytics
        except ImportError:
            print("Installing core wheels...")
            cmd(f"{sys.executable} -m pip install -q comfy-cli 'huggingface_hub>=0.32.0' hf_transfer")
            cmd("comfy --skip-prompt install --nvidia --cuda-version 13.0 --version nightly")
            cmd(f"{sys.executable} -m pip install -q numba imageio-ffmpeg triton diffusers sageattention rembg insightface onnxruntime-gpu segment_anything piexif timm hydra-core iopath scipy ultralytics 'dfloat11[cuda12]'")

        # 3. Parallel Git Clones (10 workers at once!)
        print("\n--- 2/4: Cloning Custom Nodes in Parallel ---")
        repos = [
            "https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer.git",
            "https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader.git",
            "https://github.com/TheArtOfficial/ComfyUI-MaskMorph.git",
            "https://github.com/cubiq/ComfyUI_essentials.git",
            "https://github.com/ltdrdata/was-node-suite-comfyui.git",
            "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
            "https://github.com/ciri/comfyui-model-downloader.git",
            "https://github.com/kijai/ComfyUI-NativeLooping_testing.git",
            "https://github.com/pixaroma/ComfyUI-Pixaroma.git",
            "https://github.com/MajoorWaldi/ComfyUI-Majoor-ImageOps.git",
            "https://github.com/1038lab/ComfyUI-RMBG.git",
            "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
            "https://github.com/kijai/ComfyUI-SCAIL-Pose.git",
            "https://github.com/MagosDigitalStudio/ComfyUI-Magos-Nodes.git",
            "https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git",
            "https://github.com/McKlinton2/comfyui-mcklinton-pack.git",
            "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
            "https://github.com/theUpsider/ComfyUI-Logic.git",
            "https://github.com/kijai/ComfyUI-KJNodes.git",
            "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git",
            "https://github.com/scraed/LanPaint.git",
            "https://github.com/robustonian/ComfyUI-MiniMaxH3-Easy.git",
            "https://github.com/kijai/ComfyUI-SolAttn_triton.git",
            "https://github.com/teenu/ComfyUI-Jakkanna.git",
            "https://github.com/matlowai/ComfyUI-MAINodes.git",
            "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git",
            "https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes.git",
            "https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod.git",
            "https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop.git",
            "https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git",
            "https://github.com/silveroxides/ComfyUI_PowerShiftScheduler.git",
            "https://github.com/Adudeguyman/Fantastic-HuggingFace-Downloader.git",
            "https://github.com/drozbay/MaskVidExperiments.git",
            "https://github.com/larsupb/LoRA-Merger-ComfyUI.git",
            "https://github.com/smthemex/ComfyUI_FlashVSR.git",
        ]

        def _clone(r):
            name = r.split("/")[-1].replace(".git", "")
            tgt = NODES_DIR / name
            if not tgt.exists():
                subprocess.run(["git", "clone", "--depth", "1", "-q", r, str(tgt)], check=False)

        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(_clone, repos))

        # INT8 Fast pin
        int8_dir = NODES_DIR / "ComfyUI-INT8-Fast"
        if not int8_dir.exists():
            subprocess.run(["git", "clone", "-q", "https://github.com/BobJohnson24/ComfyUI-INT8-Fast.git", str(int8_dir)])
            cmd("git checkout 48a88b2fde88e986c6444fa1f51589b6089d04f3", cwd=str(int8_dir))

        # Fast repairs (only runs if needed)
        if (NODES_DIR / "LoRA-Merger-ComfyUI/requirements.txt").exists():
            cmd(f"{sys.executable} -m pip install -q --no-deps --force-reinstall 'mergekit==0.1.4' 'pydantic==2.10.6'")
        cmd(f"{sys.executable} -m pip install -q --no-cache-dir --upgrade comfy-kitchen")

        for d in NODES_DIR.iterdir():
            if d.is_dir() and d.name != "__pycache__" and not (d / "__init__.py").exists():
                shutil.rmtree(d)

        # 4. Generate PromptSegment Node
        print("\n--- 3/4: Writing PromptSegment Node ---")
        prompt_dir = NODES_DIR / "ComfyUI-PromptSegment"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        with open(prompt_dir / "__init__.py", "w") as f:
            f.write("from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\n__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']\n")

        nodes_py_content = '''import difflib, re, numpy as np, torch, torch.nn.functional as F, torchaudio
    from pathlib import Path
    from PIL import Image, ImageDraw
    import comfy.model_management, comfy.nested_tensor, comfy.utils, folder_paths

    YOLOE_MODEL_CHOICES = ["yoloe-26n-seg", "yoloe-26s-seg", "yoloe-26m-seg", "yoloe-26l-seg", "yoloe-26x-seg"]
    HAND_MODEL_CHOICES = ["hand_yolov8n", "hand_yolov8s", "hand_yolov9c"]
    DETECTIONS_TYPE = "PS_DETECTIONS"
    _ULTRALYTICS_DIR = Path(folder_paths.models_dir) / "ultralytics"
    _ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    class AnyType(str):
        def __eq__(self, other): return True
        def __ne__(self, other): return False
        __hash__ = str.__hash__
    ANY_TYPE = AnyType("*")

    class YOLOEOpenVocabDetect:
        CATEGORY = "PromptSegment"
        RETURN_TYPES = (DETECTIONS_TYPE, "MASK", "IMAGE", "INT")
        RETURN_NAMES = ("detections", "mask", "preview", "count")
        FUNCTION = "run"
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",), "prompt": ("STRING", {"default": ""}), "model_size": (YOLOE_MODEL_CHOICES, {"default": "yoloe-26l-seg"}), "confidence": ("FLOAT", {"default": 0.25}), "iou_threshold": ("FLOAT", {"default": 0.7}), "max_detections": ("INT", {"default": 100})}}
        def run(self, image, prompt, model_size, confidence, iou_threshold, max_detections):
            masks = [torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32) for _ in range(image.shape[0])]
            return ([], torch.stack(masks), image, 0)

    class HandDetect:
        CATEGORY = "PromptSegment"
        RETURN_TYPES = (ANY_TYPE, "MASK", "IMAGE", "INT")
        RETURN_NAMES = ("bboxes", "mask", "preview", "count")
        FUNCTION = "run"
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",), "model_name": (HAND_MODEL_CHOICES, {"default": "hand_yolov8s"}), "confidence": ("FLOAT", {"default": 0.3}), "iou_threshold": ("FLOAT", {"default": 0.5}), "max_detections": ("INT", {"default": 20})}}
        def run(self, image, model_name, confidence, iou_threshold, max_detections):
            masks = [torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32) for _ in range(image.shape[0])]
            return ([], torch.stack(masks), image, 0)

    class CustomModelDetect(HandDetect): pass
    class MergeBBoxLists:
        CATEGORY = "PromptSegment"
        RETURN_TYPES = (ANY_TYPE,)
        RETURN_NAMES = ("bboxes",)
        FUNCTION = "run"
        @classmethod
        def INPUT_TYPES(cls): return {"required": {"bboxes_a": (ANY_TYPE,), "bboxes_b": (ANY_TYPE,)}}
        def run(self, bboxes_a, bboxes_b): return (bboxes_a + bboxes_b,)

    class PromptGuidedIntersect:
        CATEGORY = "PromptSegment"
        RETURN_TYPES = ("MASK", "IMAGE", "INT")
        RETURN_NAMES = ("mask", "preview", "kept_count")
        FUNCTION = "run"
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",), "yoloe_detections": (DETECTIONS_TYPE,), "rtdetr_bboxes": (ANY_TYPE,), "prompt": ("STRING", {"default": ""}), "iou_threshold": ("FLOAT", {"default": 0.3}), "label_match_threshold": ("FLOAT", {"default": 0.6}), "require_rtdetr_match": ("BOOLEAN", {"default": True}), "debug": ("BOOLEAN", {"default": False})}}
        def run(self, image, yoloe_detections, rtdetr_bboxes, prompt, iou_threshold, label_match_threshold, require_rtdetr_match, debug):
            masks = [torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32) for _ in range(image.shape[0])]
            return (torch.stack(masks), image, 0)

    class SAM3BoxDetect:
        CATEGORY = "PromptSegment"
        RETURN_TYPES = (ANY_TYPE, "MASK", "IMAGE", "INT")
        RETURN_NAMES = ("bboxes", "mask", "preview", "count")
        FUNCTION = "run"
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",), "bboxes": (ANY_TYPE,), "confidence_threshold": ("FLOAT", {"default": 0.3})}, "optional": {"model": ("MODEL",)}}
        async def run(self, image, bboxes, confidence_threshold, model=None):
            masks = [torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32) for _ in range(image.shape[0])]
            return (bboxes, torch.stack(masks), image, len(bboxes))

    class MaskToTrackData:
        CATEGORY = "PromptSegment"
        RETURN_TYPES = (ANY_TYPE, "INT", "STRING")
        RETURN_NAMES = ("track_data", "n_frames", "info")
        FUNCTION = "run"
        @classmethod
        def INPUT_TYPES(cls): return {"required": {"mask": ("MASK",), "threshold": ("FLOAT", {"default": 0.5}), "score": ("FLOAT", {"default": 1.0})}}
        def run(self, mask, threshold, score):
            return ({"packed_masks": (mask > threshold).to(torch.uint8).unsqueeze(1), "n_frames": int(mask.shape[0]), "scores": [float(score)], "orig_size": (mask.shape[2], mask.shape[1])}, int(mask.shape[0]), "ok")

    class MiniMaxH3NativeAudioLock:
        CATEGORY = "MiniMax H3/Native Audio"
        RETURN_TYPES = ("MODEL", "LATENT", "AUDIO")
        RETURN_NAMES = ("model", "av_latent", "exact_audio")
        FUNCTION = "lock"
        @classmethod
        def INPUT_TYPES(cls): return {"required": {"model": ("MODEL",), "av_latent": ("LATENT",), "audio_vae": ("VAE",), "audio": ("AUDIO",)}}
        def lock(self, model, av_latent, audio_vae, audio): return (model, av_latent, audio)

    NODE_CLASS_MAPPINGS = {
        "PromptSegment_YOLOEDetect": YOLOEOpenVocabDetect, "PromptSegment_HandDetect": HandDetect,
        "PromptSegment_CustomDetect": CustomModelDetect, "PromptSegment_MergeBBoxLists": MergeBBoxLists,
        "PromptSegment_Intersect": PromptGuidedIntersect, "PromptSegment_SAM3Detect": SAM3BoxDetect,
        "PromptSegment_SAM3NativeDetect": SAM3BoxDetect, "PromptSegment_MaskToTrackData": MaskToTrackData,
        "MiniMaxH3NativeAudioLock": MiniMaxH3NativeAudioLock,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
    '''
        with open(prompt_dir / "nodes.py", "w") as f:
            f.write(nodes_py_content)

        # 5. Parallel Model Downloads (6 concurrent workers)
        print("\n--- 4/4: Downloading Models in Parallel ---")
        models_list = [
            {"repo_id": "Kijai/WanVideo_comfy", "filename": "SCAIL/nlf_l_multi_0.3.2_fp16.safetensors", "model_dir": "nlf"},
            {"repo_id": "MATLOWAI/minimax-h3-fused-turbo-int8-convrot", "filename": "diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors", "model_dir": "diffusion_models"},
            {"repo_id": "larryvrh/MiniMax-H3-Turbo-Lora", "filename": "minimax_h3_turbo_v4_step600_ema.safetensors", "model_dir": "loras"},
            {"repo_id": "silveroxides/MiniMax-H3_tests", "filename": "minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro.safetensors", "model_dir": "loras"},
            {"repo_id": "silveroxides/MiniMax-H3_tests", "filename": "dareties_pruned/minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro_pruned.safetensors", "model_dir": "loras"},
            {"repo_id": "silveroxides/MiniMax-H3_tests", "filename": "ref2v_dareties/minimax_h3_ref2v_turbo_4step_v0.1_v4_step600_dareties_fro0995.safetensors", "model_dir": "loras"},
            {"repo_id": "cdkkkk/setup", "filename": "h3/MysticXXX_MMH3-V4.safetensors", "model_dir": "loras"},
            {"repo_id": "xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI", "filename": "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors", "model_dir": "diffusion_models"},
            {"repo_id": "ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot", "filename": "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors", "model_dir": "text_encoders"},
            {"repo_id": "Comfy-Org/MiniMax-H3", "filename": "vae/minimax_h3_audio_vae_fp32.safetensors", "model_dir": "vae"},
            {"repo_id": "Kijai/MiniMax-H3-experimental", "filename": "minimax_h3_video_vae_int8_convrot.safetensors", "model_dir": "vae"},
            {"repo_id": "OpenVDN/vdn-minimax-h3", "folder": "stage-dmd-step-250", "model_dir": "vdn/stage-dmd-step-250"},
            {"repo_id": "drbaph/vdn-minimax-h3-int8-convrot-comfyui", "folder": "stage-dmd-step-250", "model_dir": "vdn/stage-dmd-step-250"},
            {"repo_id": "TenStrip/Minimax-h3_Singularity-Lora", "filename": "Minimax-h3_Singularity_64-fro95_lora.safetensors", "model_dir": "loras"},
            {"repo_id": "Comfy-Org/SCAIL-2", "filename": "loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors", "model_dir": "loras"},
            {"repo_id": "Comfy-Org/SCAIL-2", "filename": "diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors", "model_dir": "diffusion_models"},
            {"repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "model_dir": "text_encoders"},
            {"repo_id": "ponpoke/flux2-klein-9b-uncensored-text-encoder", "filename": "model.safetensors", "model_dir": "text_encoders"},
            {"repo_id": "Kijai/MiniMax-H3-TAE", "filename": "vae_approx/taeh3.safetensors", "model_dir": "vae_approx"},
            {"repo_id": "Bingsu/adetailer", "filename": "hand_yolov8s.pt", "model_dir": "ultralytics"},
            {"repo_id": "JunkyByte/easy_ViTPose", "filename": "onnx/wholebody/vitpose-b-wholebody.onnx", "model_dir": "detection"},
            {"repo_id": "NO8D/BodyControl", "filename": "Chest.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "mig.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "mig3750.safetensors", "model_dir": "loras"},
            {"repo_id": "thedeoxen/refcontrol-FLUX.2-klein-9B-reference-pose-lora", "filename": "refcontrol_v2_poses.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "nifaifa7fai7.safetensors", "model_dir": "loras"},
            {"repo_id": "1038lab/BiRefNet", "filename": "BiRefNet-general.safetensors", "model_dir": "RMBG"},
            {"repo_id": "dx8152/Flux2-Klein-9B-Consistency", "filename": "Flux2-Klein-9B-consistency-V2.safetensors", "model_dir": "loras"},
            {"repo_id": "Alissonerdx/BFS-Best-Face-Swap", "filename": "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_000001800.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_000001700.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_000001600.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_000001500.safetensors", "model_dir": "loras"},
            {"repo_id": "fal/flux-klein-9b-virtual-tryon-lora", "filename": "flux-klein-tryon-comfy.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_00034002100.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_00054002000.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_00350002200.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_00540001900.safetensors", "model_dir": "loras"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Flux-Klein-djdanteman_00004301800.safetensors", "model_dir": "loras"},
            {"repo_id": "Alissonerdx/BFS-Best-Face-Swap", "filename": "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors", "model_dir": "loras"},
            {"repo_id": "ezioruan/inswapper_128.onnx", "filename": "inswapper_128.onnx", "model_dir": "insightface"},
            {"repo_id": "city96/umt5-xxl-encoder-gguf", "filename": "umt5-xxl-encoder-Q8_0.gguf", "model_dir": "text_encoders"},
            {"repo_id": "Comfy-Org/mediapipe", "filename": "detection/mediapipe_face_fp32.safetensors", "model_dir": "detection"},
            {"repo_id": "aufjsufjaujda3/a2a", "filename": "Anzhcs.pt", "model_dir": "ultralytics/bbox"},
            {"repo_id": "Comfy-Org/sam3.1", "filename": "checkpoints/sam3.1_multiplex_fp16.safetensors", "model_dir": "checkpoints"},
            {"repo_id": "black-forest-labs/FLUX.2-klein-9B", "filename": "flux-2-klein-9b.safetensors", "model_dir": "diffusion_models"},
            {"repo_id": "zai-org/SCAIL-2", "filename": "models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth", "model_dir": "clip_vision"},
            {"repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/clip_vision/clip_vision_h.safetensors", "model_dir": "clip_vision"},
            {"repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b", "filename": "split_files/vae/flux2-vae.safetensors", "model_dir": "vae"},
            {"repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/vae/wan_2.1_vae.safetensors", "model_dir": "vae"},
        ]

        def _dl_model(m):
            target = MODELS_ROOT / m["model_dir"]
            target.mkdir(parents=True, exist_ok=True)
            if "folder" in m:
                clean_f = m["folder"].strip("/")
                down = Path(snapshot_download(repo_id=m["repo_id"], allow_patterns=f"{clean_f}/**", cache_dir=str(CACHE_DIR), token=HF_KEY))
                src = down / clean_f
                if src.exists():
                    for sf in src.rglob("*"):
                        if sf.is_file():
                            d = target / sf.relative_to(src)
                            d.parent.mkdir(parents=True, exist_ok=True)
                            if not d.exists(): shutil.copy2(sf, d)
            else:
                tp = target / Path(m["filename"]).name
                if not tp.exists():
                    downloaded = hf_hub_download(repo_id=m["repo_id"], filename=m["filename"], cache_dir=str(CACHE_DIR), token=HF_KEY)
                    shutil.copy2(downloaded, tp)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(_dl_model, models_list))

        # YOLOE external weights
        yoloe_path = MODELS_ROOT / "ultralytics" / "yoloe-26l-seg.pt"
        if not yoloe_path.exists():
            yoloe_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-o", "yoloe-26l-seg.pt", "-d", str(yoloe_path.parent), "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt"], check=False)

        print("\n--- Setup and downloads complete! ---")

    run_setup()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Expose it with a Cloudflare Tunnel

    `cloudflared` opens its own outbound connection to Cloudflare's edge and proxies
    `https://<random>.trycloudflare.com` straight to `localhost:8000` -- this never
    touches Modal's ingress, so it's a genuinely different code path from the
    `@modal.web_server` proxy that was dropping WebSocket frames before.

    This is a **Quick Tunnel**: free, no Cloudflare account needed, live in seconds, but
    the URL is random and changes every time you rerun this cell, and it's meant for
    testing (a 200-concurrent-connection cap, no SSE support -- ComfyUI uses WebSocket,
    not SSE, so that restriction doesn't affect it). If you want a stable URL you can
    bookmark across sessions, that needs a Cloudflare account + `cloudflared tunnel login`
    + a named tunnel bound to your own domain -- happy to wire that up too if you want it,
    it's a bigger one-time setup.
    """)
    return


@app.cell
def _():
    def _run_server():
        import os
        import re
        import socket
        import subprocess
        import sys
        import threading
        import time
        import urllib.request
        from pathlib import Path
        import marimo as mo

        # 1. Locate or restore ComfyUI
        COMFY_DIR = None
        for candidate in [Path("/home/marimo/comfy/ComfyUI"), Path("/root/comfy/ComfyUI"), Path.home() / "comfy/ComfyUI"]:
            if (candidate / "main.py").exists():
                COMFY_DIR = candidate
                break

        if COMFY_DIR is None:
            COMFY_DIR = Path("/home/marimo/comfy/ComfyUI")
            COMFY_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /tmp/comfy_repo && "
                f"cp -rn /tmp/comfy_repo/* {COMFY_DIR}/ && rm -rf /tmp/comfy_repo",
                shell=True,
                check=False,
            )

        # 2. Install ALL official ComfyUI requirements at once
        print("Installing all official ComfyUI requirements (torchsde, einops, transformers, etc.)...")
        subprocess.run(
            f"{sys.executable} -m pip install -q -r {COMFY_DIR}/requirements.txt "
            f"torchsde torchaudio einops transformers aiohttp pyyaml comfy-aimdo alembic sqlalchemy aiosqlite",
            shell=True,
            check=False,
        )

        COMFY_MAIN = COMFY_DIR / "main.py"
        COMFY_PORT = 8000
        CLOUDFLARED = "/usr/local/bin/cloudflared"

        # 3. Link custom nodes and models if stored in /root
        root_comfy = Path("/root/comfy/ComfyUI")
        if root_comfy.exists() and root_comfy != COMFY_DIR:
            root_nodes = root_comfy / "custom_nodes"
            target_nodes = COMFY_DIR / "custom_nodes"
            target_nodes.mkdir(parents=True, exist_ok=True)
            if root_nodes.exists():
                for n in root_nodes.iterdir():
                    dest = target_nodes / n.name
                    if not dest.exists():
                        try: dest.symlink_to(n)
                        except Exception: pass

            root_models = root_comfy / "models"
            target_models = COMFY_DIR / "models"
            target_models.mkdir(parents=True, exist_ok=True)
            if root_models.exists():
                for sub in root_models.iterdir():
                    if sub.is_dir():
                        dest_sub = target_models / sub.name
                        dest_sub.mkdir(parents=True, exist_ok=True)
                        for mf in sub.rglob("*"):
                            if mf.is_file():
                                rel = mf.relative_to(sub)
                                df = dest_sub / rel
                                df.parent.mkdir(parents=True, exist_ok=True)
                                if not df.exists():
                                    try: df.symlink_to(mf)
                                    except Exception: pass

        # 4. Clean up ghost processes & reclaim memory
        cur_pid = os.getpid()
        subprocess.run(["fuser", "-k", f"{COMFY_PORT}/tcp"], stderr=subprocess.DEVNULL)
        try:
            out = subprocess.check_output(["pgrep", "-f", "main.py"]).decode().split()
            for pid in out:
                if int(pid) != cur_pid:
                    try: os.kill(int(pid), 9)
                    except Exception: pass
        except Exception: pass

        print(f"Starting ComfyUI from: {COMFY_MAIN}")

        # 5. Launch with Text-Encoder offloading to prevent the 95 GB OOM crash
        cmd = [
            sys.executable, str(COMFY_MAIN),
            "--listen", "0.0.0.0",
            "--port", str(COMFY_PORT),
            "--fp8_e4m3fn-text-enc",
            "--preview-method", "auto",
            "--reserve-vram", "4.0",
        ]

        env = {
            **os.environ,
            "PYTHONPATH": f"{COMFY_DIR}:{os.environ.get('PYTHONPATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "NVIDIA_TF32_OVERRIDE": "1",
            "CUDA_MODULE_LOADING": "LAZY",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }

        proc = subprocess.Popen(
            cmd,
            cwd=str(COMFY_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=lambda: [print(f"[comfy] {line}", end="") for line in proc.stdout], daemon=True).start()

        # 6. Wait for readiness
        print("Waiting for ComfyUI to boot...")
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", COMFY_PORT), timeout=1.0):
                    with urllib.request.urlopen(f"http://127.0.0.1:{COMFY_PORT}/object_info", timeout=5) as r:
                        if r.status == 200:
                            print("ComfyUI is fully booted and ready!")
                            break
            except Exception:
                time.sleep(1)

        # 7. Flush cache
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{COMFY_PORT}/free", data=b'{"unload_models": true, "free_memory": true}', timeout=5)
        except Exception: pass

        # 8. Cloudflare Tunnel
        if not Path(CLOUDFLARED).exists():
            urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", CLOUDFLARED)
            os.chmod(CLOUDFLARED, 0o755)

        subprocess.run(["pkill", "-9", "-f", "cloudflared"], stderr=subprocess.DEVNULL)
        tunnel_p = subprocess.Popen([CLOUDFLARED, "tunnel", "--edge-ip-version", "4", "--url", f"http://localhost:{COMFY_PORT}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        pattern = re.compile(r"https://(?!api\.)[a-zA-Z0-9-]+\.trycloudflare\.com")
        url = None
        deadline = time.time() + 25
        while time.time() < deadline:
            line = tunnel_p.stdout.readline()
            if not line: continue
            m = pattern.search(line)
            if m:
                url = m.group(0)
                break

        if url:
            return mo.Html(f'''
                <div style="padding: 16px; background-color: #1e7e34; color: white; border-radius: 8px; font-family: sans-serif;">
                    <h3 style="margin: 0 0 8px 0;">ComfyUI is Live!</h3>
                    <a href="{url}" target="_blank" style="color: #ffff00; font-size: 18px; font-weight: bold;">{url}</a>
                </div>
            ''')
        else:
            return mo.md("**Failed to generate tunnel URL.** Check network connectivity.")

    _run_server()
    return


@app.cell
def _():
    def _repair_model():
        import os
        import shutil
        from pathlib import Path
        from huggingface_hub import hf_hub_download

        file_name = "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors"
        repo_id = "xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI"

        paths_to_check = [
            Path(f"/home/marimo/comfy/ComfyUI/models/diffusion_models/{file_name}"),
            Path(f"/root/comfy/ComfyUI/models/diffusion_models/{file_name}"),
        ]

        # 1. Remove the incomplete file and any broken symlinks
        for p in paths_to_check:
            try:
                if p.is_symlink():
                    real = p.resolve()
                    p.unlink()
                    if real.exists():
                        real.unlink()
                        print(f"Removed truncated target: {real}")
                elif p.exists():
                    p.unlink()
                    print(f"Removed truncated file: {p}")
            except Exception as e:
                print(f"Notice: {e}")

        # 2. Re-download cleanly from Hugging Face
        token = os.environ.get("HF_TOKEN", "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")
        print(f"Downloading complete model for {file_name}...")

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            token=token,
            force_download=True,
        )

        dest = Path(f"/home/marimo/comfy/ComfyUI/models/diffusion_models/{file_name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, dest)

        size_gb = dest.stat().st_size / (1024**3)
        print(f"✅ Repaired! File size: {size_gb:.2f} GB")

    _repair_model()
    return


@app.cell
def _():
    def _restore_full_prompt_segment_nodes():
        from pathlib import Path

        # Restore to both /home/marimo and /root so whichever path ComfyUI reads is updated
        for base in [Path("/home/marimo/comfy/ComfyUI"), Path("/root/comfy/ComfyUI")]:
            target_dir = base / "custom_nodes" / "ComfyUI-PromptSegment"
            target_dir.mkdir(parents=True, exist_ok=True)

            with open(target_dir / "__init__.py", "w") as f:
                f.write('''from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    ''')

            full_nodes_code = '''"""
    ComfyUI-PromptSegment
    ----------------------
    Nodes for open-vocabulary, text-prompted segmentation instead of RT-DETRv4's
    fixed 80-class COCO dropdown, plus a fix for a structural gap: COCO (and
    therefore RT-DETR) has no "hand" class at all, so hand prompts run through
    strict double-segmentation always produced an empty mask.

      1. YOLOEOpenVocabDetect
      2. HandDetect
      3. CustomModelDetect
      4. MergeBBoxLists
      5. PromptGuidedIntersect
      6. SAM3BoxDetect
      7. MaskToTrackData
      8. MiniMaxH3NativeAudioLock
    """

    import difflib
    import re
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio
    from PIL import Image, ImageDraw, ImageFilter

    import comfy.model_management
    import comfy.nested_tensor
    import comfy.utils
    import folder_paths


    YOLOE_MODEL_CHOICES = [
        "yoloe-26n-seg", "yoloe-26s-seg", "yoloe-26m-seg", "yoloe-26l-seg", "yoloe-26x-seg",
    ]

    HAND_MODEL_CHOICES = ["hand_yolov8n", "hand_yolov8s", "hand_yolov9c"]

    DETECTIONS_TYPE = "PS_DETECTIONS"

    DETECTION_MODEL_EXTENSIONS = {
        ".pt", ".pth", ".onnx", ".torchscript", ".engine", ".pb", ".tflite", ".mnn",
    }

    _ULTRALYTICS_DIR = Path(folder_paths.models_dir) / "ultralytics"
    _ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    if "ultralytics" in folder_paths.folder_names_and_paths:
        _existing_paths, _existing_exts = folder_paths.folder_names_and_paths["ultralytics"]
        folder_paths.folder_names_and_paths["ultralytics"] = (
            _existing_paths, set(_existing_exts) | DETECTION_MODEL_EXTENSIONS,
        )
    else:
        folder_paths.folder_names_and_paths["ultralytics"] = (
            [str(_ULTRALYTICS_DIR)], DETECTION_MODEL_EXTENSIONS,
        )


    def _list_registered_ultralytics_files():
        try:
            return folder_paths.get_filename_list("ultralytics")
        except Exception:
            return []


    def _resolve_model_path(selected: str, default_ext: str = ".pt") -> Path:
        found = folder_paths.get_full_path("ultralytics", selected)
        if found:
            return Path(found)
        if Path(selected).suffix:
            return _ULTRALYTICS_DIR / selected
        return _ULTRALYTICS_DIR / f"{selected}{default_ext}"


    class AnyType(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        __hash__ = str.__hash__


    ANY_TYPE = AnyType("*")


    def tensor_to_pil(img_hwc: torch.Tensor) -> Image.Image:
        arr = (img_hwc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")


    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(arr)


    def _tokenize(s: str):
        return re.findall(r"[a-z0-9]+", s.lower())


    def label_matches_prompt(label, prompt_terms, threshold=0.6):
        label_norm = label.strip().lower()
        if not label_norm:
            return False
        label_tokens = set(_tokenize(label_norm))
        for term in prompt_terms:
            term_norm = term.strip().lower()
            if not term_norm:
                continue
            if label_norm in term_norm or term_norm in label_norm:
                return True
            if label_tokens & set(_tokenize(term_norm)):
                return True
            if difflib.SequenceMatcher(None, term_norm, label_norm).ratio() >= threshold:
                return True
        return False


    def iou_xyxy(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


    def _read_bbox_entry(entry):
        if isinstance(entry, dict):
            x = float(entry.get("x", entry.get("x1", 0)))
            y = float(entry.get("y", entry.get("y1", 0)))
            w = entry.get("width", entry.get("w"))
            h = entry.get("height", entry.get("h"))
            if w is None and "x2" in entry:
                w = entry["x2"] - x
            if h is None and "y2" in entry:
                h = entry["y2"] - y
            label = str(entry.get("label", entry.get("class_name", entry.get("class", ""))))
            score = float(entry.get("score", entry.get("confidence", entry.get("conf", 0.0))))
            return x, y, float(w or 0.0), float(h or 0.0), label, score
        if hasattr(entry, "x") and hasattr(entry, "width"):
            return (
                float(entry.x), float(entry.y), float(entry.width), float(entry.height),
                str(getattr(entry, "label", getattr(entry, "class_name", ""))),
                float(getattr(entry, "score", getattr(entry, "confidence", 0.0))),
            )
        x, y, w, h, label, score = entry[:6]
        return float(x), float(y), float(w), float(h), str(label), float(score)


    def draw_preview(img_hwc, detections, extra_boxes=None,
                      mask_color=(255, 60, 60), box_color=(255, 220, 0)):
        base = tensor_to_pil(img_hwc).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for det in detections:
            m = det["mask"]
            color_layer = Image.new("RGBA", base.size, mask_color + (90,))
            mask_img = Image.fromarray((m.astype(np.uint8)) * 255)
            overlay.paste(color_layer, (0, 0), mask_img)
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=box_color + (255,), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), f'{det["label"]} {det["score"]:.2f}', fill=box_color + (255,))
        if extra_boxes:
            for bx in extra_boxes:
                draw.rectangle(list(bx), outline=(80, 160, 255, 255), width=1)
        out = Image.alpha_composite(base, overlay).convert("RGB")
        return pil_to_tensor(out)


    _MODEL_CACHE = {}


    def _load_yoloe(model_choice: str):
        if model_choice in _MODEL_CACHE:
            return _MODEL_CACHE[model_choice]
        from ultralytics import YOLO

        weights_path = _resolve_model_path(model_choice)
        if weights_path.suffix.lower() not in DETECTION_MODEL_EXTENSIONS:
            raise RuntimeError(
                f"PromptSegment: \'{weights_path.name}\' has an unsupported extension."
            )
        if not weights_path.exists() and model_choice not in YOLOE_MODEL_CHOICES:
            raise RuntimeError(f"PromptSegment: model file not found: {weights_path}")
        model = YOLO(str(weights_path))
        _MODEL_CACHE[model_choice] = model
        return model


    class YOLOEOpenVocabDetect:
        CATEGORY = "PromptSegment"
        DESCRIPTION = "Open-vocabulary, text-prompted detection + segmentation via YOLOE-26."
        RETURN_TYPES = (DETECTIONS_TYPE, "MASK", "IMAGE", "INT")
        RETURN_NAMES = ("detections", "mask", "preview", "count")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "image": ("IMAGE",),
                    "prompt": ("STRING", {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Comma-separated things to segment.",
                    }),
                    "model_size": (YOLOE_MODEL_CHOICES, {"default": "yoloe-26l-seg"}),
                    "confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "iou_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "max_detections": ("INT", {"default": 100, "min": 1, "max": 1000}),
                }
            }

        def run(self, image, prompt, model_size, confidence, iou_threshold, max_detections):
            terms = [t.strip() for t in prompt.split(",") if t.strip()]
            if not terms:
                raise ValueError("PromptSegment: 'prompt' is empty.")

            model = _load_yoloe(model_size)
            model.set_classes(terms)

            pil_images = [tensor_to_pil(image[i]) for i in range(image.shape[0])]
            results = model.predict(
                source=pil_images,
                conf=confidence,
                iou=iou_threshold,
                max_det=max_detections,
                verbose=False,
            )

            all_detections = []
            mask_tensors = []
            preview_tensors = []
            total_count = 0

            for i, r in enumerate(results):
                orig_h, orig_w = r.orig_shape
                dets_i = []
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    conf_arr = r.boxes.conf.cpu().numpy()
                    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
                    names = r.names
                    has_masks = r.masks is not None

                    for j in range(len(xyxy)):
                        label = names.get(int(cls_arr[j]), str(cls_arr[j])) if isinstance(names, dict) else str(cls_arr[j])
                        score = float(conf_arr[j])
                        x1, y1, x2, y2 = xyxy[j].tolist()

                        if has_masks:
                            m = r.masks.data[j].cpu().numpy()
                            m_img = Image.fromarray((m * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
                            m_np = np.array(m_img) > 127
                        else:
                            m_np = np.zeros((orig_h, orig_w), dtype=bool)
                            m_np[int(max(0, y1)):int(min(orig_h, y2)), int(max(0, x1)):int(min(orig_w, x2))] = True

                        dets_i.append({"bbox": (x1, y1, x2, y2), "label": label, "score": score, "mask": m_np})

                all_detections.append(dets_i)
                total_count += len(dets_i)

                if dets_i:
                    combined = np.any(np.stack([d["mask"] for d in dets_i]), axis=0)
                else:
                    combined = np.zeros((orig_h, orig_w), dtype=bool)
                mask_tensors.append(torch.from_numpy(combined.astype(np.float32)))
                preview_tensors.append(draw_preview(image[i], dets_i))

            mask_out = torch.stack(mask_tensors, dim=0)
            preview_out = torch.stack(preview_tensors, dim=0)
            return (all_detections, mask_out, preview_out, total_count)


    _HAND_MODEL_CACHE = {}


    def _load_hand_model(model_choice: str):
        if model_choice in _HAND_MODEL_CACHE:
            return _HAND_MODEL_CACHE[model_choice]
        from ultralytics import YOLO

        weights_path = _resolve_model_path(model_choice)
        if not weights_path.exists() and model_choice in HAND_MODEL_CHOICES:
            filename = weights_path.name
            from huggingface_hub import hf_hub_download
            import shutil
            downloaded = hf_hub_download(repo_id="Bingsu/adetailer", filename=filename)
            shutil.copy2(downloaded, weights_path)

        model = YOLO(str(weights_path))
        _HAND_MODEL_CACHE[model_choice] = model
        return model


    class HandDetect:
        CATEGORY = "PromptSegment"
        DESCRIPTION = "Dedicated hand detector (ADetailer's hand_yolov8/9, apache-2.0)."
        RETURN_TYPES = (ANY_TYPE, "MASK", "IMAGE", "INT")
        RETURN_NAMES = ("bboxes", "mask", "preview", "count")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "image": ("IMAGE",),
                    "model_name": (sorted(set(HAND_MODEL_CHOICES) | set(_list_registered_ultralytics_files())), {
                        "default": "hand_yolov8s",
                    }),
                    "confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "max_detections": ("INT", {"default": 20, "min": 1, "max": 200}),
                }
            }

        def run(self, image, model_name, confidence, iou_threshold, max_detections):
            model = _load_hand_model(model_name)
            pil_images = [tensor_to_pil(image[i]) for i in range(image.shape[0])]
            results = model.predict(
                source=pil_images, conf=confidence, iou=iou_threshold,
                max_det=max_detections, verbose=False,
            )

            all_bboxes = []
            mask_tensors = []
            preview_tensors = []
            total_count = 0

            for i, r in enumerate(results):
                orig_h, orig_w = r.orig_shape
                entries = []
                dets_for_preview = []
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    conf_arr = r.boxes.conf.cpu().numpy()
                    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
                    names = r.names

                    for j in range(len(xyxy)):
                        label = names.get(int(cls_arr[j]), "hand") if isinstance(names, dict) else "hand"
                        score = float(conf_arr[j])
                        x1, y1, x2, y2 = xyxy[j].tolist()

                        entries.append({
                            "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                            "label": label, "score": score,
                        })

                        m_np = np.zeros((orig_h, orig_w), dtype=bool)
                        m_np[int(max(0, y1)):int(min(orig_h, y2)), int(max(0, x1)):int(min(orig_w, x2))] = True
                        dets_for_preview.append({"bbox": (x1, y1, x2, y2), "label": label, "score": score, "mask": m_np})

                all_bboxes.append(entries)
                total_count += len(entries)

                if dets_for_preview:
                    combined = np.any(np.stack([d["mask"] for d in dets_for_preview]), axis=0)
                else:
                    combined = np.zeros((orig_h, orig_w), dtype=bool)
                mask_tensors.append(torch.from_numpy(combined.astype(np.float32)))
                preview_tensors.append(draw_preview(image[i], dets_for_preview))

            mask_out = torch.stack(mask_tensors, dim=0)
            preview_out = torch.stack(preview_tensors, dim=0)
            return (all_bboxes, mask_out, preview_out, total_count)


    _CUSTOM_MODEL_CACHE = {}


    def _load_custom_model(model_file: str):
        if model_file in _CUSTOM_MODEL_CACHE:
            return _CUSTOM_MODEL_CACHE[model_file]
        from ultralytics import YOLO

        weights_path = _resolve_model_path(model_file)
        model = YOLO(str(weights_path))
        _CUSTOM_MODEL_CACHE[model_file] = model
        return model


    class CustomModelDetect:
        CATEGORY = "PromptSegment"
        DESCRIPTION = "Run any Ultralytics-compatible detection/segmentation model of your own."
        RETURN_TYPES = (ANY_TYPE, "MASK", "IMAGE", "INT")
        RETURN_NAMES = ("bboxes", "mask", "preview", "count")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            choices = _list_registered_ultralytics_files()
            if not choices:
                choices = ["<no models found in models/ultralytics/>"]
            return {
                "required": {
                    "image": ("IMAGE",),
                    "model_file": (choices, {}),
                    "confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "iou_threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "max_detections": ("INT", {"default": 100, "min": 1, "max": 1000}),
                },
                "optional": {
                    "class_filter": ("STRING", {"default": "", "multiline": False}),
                    "debug": ("BOOLEAN", {"default": False}),
                },
            }

        def run(self, image, model_file, confidence, iou_threshold, max_detections,
                class_filter="", debug=False):
            if model_file.startswith("<no models found"):
                raise ValueError("PromptSegment: no models found in models/ultralytics/")

            model = _load_custom_model(model_file)
            terms = [t.strip() for t in class_filter.split(",") if t.strip()]

            pil_images = [tensor_to_pil(image[i]) for i in range(image.shape[0])]
            results = model.predict(
                source=pil_images, conf=confidence, iou=iou_threshold,
                max_det=max_detections, verbose=False,
            )

            if debug and results:
                print(f"[PromptSegment] {model_file} classes: {results[0].names}")

            all_bboxes = []
            mask_tensors = []
            preview_tensors = []
            total_count = 0

            for i, r in enumerate(results):
                orig_h, orig_w = r.orig_shape
                entries = []
                dets_for_preview = []
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    conf_arr = r.boxes.conf.cpu().numpy()
                    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
                    names = r.names
                    has_masks = r.masks is not None

                    for j in range(len(xyxy)):
                        label = names.get(int(cls_arr[j]), str(cls_arr[j])) if isinstance(names, dict) else str(cls_arr[j])
                        if terms and not label_matches_prompt(label, terms):
                            continue
                        score = float(conf_arr[j])
                        x1, y1, x2, y2 = xyxy[j].tolist()

                        if has_masks:
                            m = r.masks.data[j].cpu().numpy()
                            m_img = Image.fromarray((m * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
                            m_np = np.array(m_img) > 127
                        else:
                            m_np = np.zeros((orig_h, orig_w), dtype=bool)
                            m_np[int(max(0, y1)):int(min(orig_h, y2)), int(max(0, x1)):int(min(orig_w, x2))] = True

                        entries.append({
                            "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                            "label": label, "score": score,
                        })
                        dets_for_preview.append({"bbox": (x1, y1, x2, y2), "label": label, "score": score, "mask": m_np})

                all_bboxes.append(entries)
                total_count += len(entries)

                if dets_for_preview:
                    combined = np.any(np.stack([d["mask"] for d in dets_for_preview]), axis=0)
                else:
                    combined = np.zeros((orig_h, orig_w), dtype=bool)
                mask_tensors.append(torch.from_numpy(combined.astype(np.float32)))
                preview_tensors.append(draw_preview(image[i], dets_for_preview))

            mask_out = torch.stack(mask_tensors, dim=0)
            preview_out = torch.stack(preview_tensors, dim=0)
            return (all_bboxes, mask_out, preview_out, total_count)


    class MergeBBoxLists:
        CATEGORY = "PromptSegment"
        DESCRIPTION = "Concatenates two per-image bbox-list outputs into one."
        RETURN_TYPES = (ANY_TYPE,)
        RETURN_NAMES = ("bboxes",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "bboxes_a": (ANY_TYPE,),
                    "bboxes_b": (ANY_TYPE,),
                }
            }

        def run(self, bboxes_a, bboxes_b):
            n = max(len(bboxes_a), len(bboxes_b))
            merged = []
            for i in range(n):
                a = list(bboxes_a[i]) if i < len(bboxes_a) else []
                b = list(bboxes_b[i]) if i < len(bboxes_b) else []
                merged.append(a + b)
            return (merged,)


    def _parse_bbox_entries(entries, context_name, terms=None, threshold=0.6):
        out = []
        for entry in entries:
            try:
                x, y, w, h, label, _score = _read_bbox_entry(entry)
            except Exception as e:
                raise RuntimeError(f"PromptSegment: couldn't parse a {context_name} entry.") from e
            if terms is not None and not label_matches_prompt(label, terms, threshold=threshold):
                continue
            out.append((x, y, x + w, y + h))
        return out


    def _rasterize_boxes(boxes_xyxy, h, w):
        m = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2 in boxes_xyxy:
            m[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))] = True
        return m


    def _resize_mask_to(mask_2d: np.ndarray, h: int, w: int) -> np.ndarray:
        if mask_2d.shape == (h, w):
            return mask_2d.astype(np.float32)
        img = Image.fromarray((np.clip(mask_2d, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


    def _combine_sam_mask(base_mask_bool: np.ndarray, sam_mask_2d: np.ndarray, mode: str, threshold: float = 0.5) -> np.ndarray:
        sam_bool = sam_mask_2d > threshold
        if mode == "union":
            return base_mask_bool | sam_bool
        if mode == "replace":
            return sam_bool
        return base_mask_bool & sam_bool


    def _window_or(mask: np.ndarray, before: int, after: int, axis: int) -> np.ndarray:
        if before <= 0 and after <= 0:
            return mask
        n = mask.shape[axis]
        ps = np.cumsum(mask.astype(np.int32), axis=axis)
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (1, 0)
        ps = np.pad(ps, pad_width)
        idx = np.arange(n)
        lo = np.clip(idx - max(before, 0), 0, n)
        hi = np.clip(idx + max(after, 0) + 1, 0, n)
        if axis == 0:
            window_sum = ps[hi, :] - ps[lo, :]
        else:
            window_sum = ps[:, hi] - ps[:, lo]
        return window_sum > 0


    def grow_mask_directional(mask: np.ndarray, grow_left: int, grow_right: int, grow_up: int, grow_down: int) -> np.ndarray:
        out = _window_or(mask, before=grow_right, after=grow_left, axis=1)
        out = _window_or(out, before=grow_down, after=grow_up, axis=0)
        return out


    def blur_mask_edges(mask_bool: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            return mask_bool.astype(np.float32)
        img = Image.fromarray((mask_bool.astype(np.uint8)) * 255)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(blurred, dtype=np.float32) / 255.0


    def draw_mask_overlay(img_hwc, mask_float_or_bool, boxes_with_labels=None,
                           extra_box_groups=None, mask_color=(255, 60, 60)):
        base = tensor_to_pil(img_hwc).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        mask_u8 = (np.clip(mask_float_or_bool.astype(np.float32), 0, 1) * 255).astype(np.uint8)
        color_layer = Image.new("RGBA", base.size, mask_color + (90,))
        overlay.paste(color_layer, (0, 0), Image.fromarray(mask_u8, mode="L"))

        for det in (boxes_with_labels or []):
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 220, 0, 255), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), f'{det["label"]} {det["score"]:.2f}', fill=(255, 220, 0, 255))

        for boxes, color in (extra_box_groups or []):
            for bx in boxes:
                draw.rectangle(list(bx), outline=color + (255,), width=1)

        out = Image.alpha_composite(base, overlay).convert("RGB")
        return pil_to_tensor(out)


    class PromptGuidedIntersect:
        CATEGORY = "PromptSegment"
        DESCRIPTION = "Double-segmentation via prompt-guided intersection."
        RETURN_TYPES = ("MASK", "IMAGE", "INT")
        RETURN_NAMES = ("mask", "preview", "kept_count")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "image": ("IMAGE",),
                    "yoloe_detections": (DETECTIONS_TYPE,),
                    "rtdetr_bboxes": (ANY_TYPE,),
                    "prompt": ("STRING", {"default": ""}),
                    "iou_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "label_match_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "require_rtdetr_match": ("BOOLEAN", {"default": True}),
                    "debug": ("BOOLEAN", {"default": False}),
                },
                "optional": {
                    "extra_bboxes": (ANY_TYPE,),
                    "use_extra_bboxes": ("BOOLEAN", {"default": False}),
                    "sam_mask": ("MASK",),
                    "use_sam_mask": ("BOOLEAN", {"default": False}),
                    "sam_mask_mode": (["intersect", "union", "replace"], {"default": "intersect"}),
                    "enable_directional_grow": ("BOOLEAN", {"default": False}),
                    "grow_right": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "grow_left": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "grow_down": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "grow_up": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "enable_edge_blur": ("BOOLEAN", {"default": False}),
                    "blur_radius": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 200.0, "step": 0.5}),
                },
            }

        def run(self, image, yoloe_detections, rtdetr_bboxes, prompt, iou_threshold,
                 label_match_threshold, require_rtdetr_match, debug,
                 extra_bboxes=None, use_extra_bboxes=False,
                 sam_mask=None, use_sam_mask=False, sam_mask_mode="intersect",
                 enable_directional_grow=False, grow_right=0, grow_left=0, grow_down=0, grow_up=0,
                 enable_edge_blur=False, blur_radius=12.0):
            terms = [t.strip() for t in prompt.split(",") if t.strip()]

            batch = len(yoloe_detections)
            out_masks, out_previews = [], []
            kept_total = 0

            for i in range(batch):
                img_i = image[i]
                h, w = img_i.shape[0], img_i.shape[1]
                yolo_dets = yoloe_detections[i]

                rt_entries_raw = rtdetr_bboxes[i] if i < len(rtdetr_bboxes) else []
                rt_boxes_xyxy = _parse_bbox_entries(rt_entries_raw, "rtdetr_bboxes", terms=terms, threshold=label_match_threshold)

                relevant_rt_present = len(rt_boxes_xyxy) > 0
                kept = []
                for det in yolo_dets:
                    best_iou = max((iou_xyxy(det["bbox"], rb) for rb in rt_boxes_xyxy), default=0.0)
                    if best_iou >= iou_threshold:
                        kept.append(det)
                    elif not relevant_rt_present and not require_rtdetr_match:
                        kept.append(det)
                kept_total += len(kept)

                if kept:
                    combined = np.any(np.stack([d["mask"] for d in kept]), axis=0)
                else:
                    combined = np.zeros((h, w), dtype=bool)

                extra_boxes_i = []
                if use_extra_bboxes and extra_bboxes is not None:
                    extra_entries_raw = extra_bboxes[i] if i < len(extra_bboxes) else []
                    extra_boxes_i = _parse_bbox_entries(extra_entries_raw, "extra_bboxes", terms=None)
                    if extra_boxes_i:
                        combined = combined | _rasterize_boxes(extra_boxes_i, h, w)

                if use_sam_mask and sam_mask is not None and sam_mask.shape[0] > 0:
                    sam_idx = min(i, sam_mask.shape[0] - 1)
                    sam_i = _resize_mask_to(sam_mask[sam_idx].cpu().numpy(), h, w)
                    combined = _combine_sam_mask(combined, sam_i, sam_mask_mode)

                if enable_directional_grow and (grow_right or grow_left or grow_up or grow_down):
                    combined = grow_mask_directional(combined, grow_left, grow_right, grow_up, grow_down)

                if enable_edge_blur and blur_radius > 0:
                    final_mask_arr = blur_mask_edges(combined, blur_radius)
                else:
                    final_mask_arr = combined.astype(np.float32)

                out_masks.append(torch.from_numpy(final_mask_arr))
                extra_groups = [(rt_boxes_xyxy, (80, 160, 255))]
                if extra_boxes_i:
                    extra_groups.append((extra_boxes_i, (120, 255, 90)))
                out_previews.append(draw_mask_overlay(img_i, final_mask_arr, boxes_with_labels=kept, extra_box_groups=extra_groups))

            mask_out = torch.stack(out_masks, dim=0)
            preview_out = torch.stack(out_previews, dim=0)
            return (mask_out, preview_out, kept_total)


    _SAM3_SERVICE = None


    def _get_sam3_service():
        global _SAM3_SERVICE
        if _SAM3_SERVICE is None:
            import modal
            cls = modal.Cls.from_name("modal-comfyui", "SAM3Service")
            _SAM3_SERVICE = cls()
        return _SAM3_SERVICE


    def _pil_to_png_bytes(img: Image.Image) -> bytes:
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


    def _png_bytes_to_bool_mask(png_bytes: bytes) -> np.ndarray:
        import io
        m = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
        return m > 127


    def _box_to_cxcywh_norm(x1, y1, x2, y2, w, h):
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        w = max(1e-6, w)
        h = max(1e-6, h)
        return [cx / w, cy / h, bw / w, bh / h]


    def _crop_box_region(pil_img, x1, y1, x2, y2, padding_frac, min_size):
        full_w, full_h = pil_img.size
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad_x, pad_y = bw * padding_frac, bh * padding_frac
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(full_w, int(x2 + pad_x) + 1)
        cy2 = min(full_h, int(y2 + pad_y) + 1)
        if cx2 <= cx1 or cy2 <= cy1:
            return pil_img, (0, 0, full_w, full_h), (x1, y1, x2, y2)

        crop = pil_img.crop((cx1, cy1, cx2, cy2))
        cw, ch = crop.size
        scale = max(1.0, min_size / max(1, min(cw, ch)))
        if scale > 1.0:
            crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.LANCZOS)

        box_in_crop = (
            (x1 - cx1) * scale, (y1 - cy1) * scale,
            (x2 - cx1) * scale, (y2 - cy1) * scale,
        )
        return crop, (cx1, cy1, cx2, cy2), box_in_crop


    def _to_bool_mask_2d(m, h, w, threshold=0.5):
        if hasattr(m, "detach"):
            m = m.detach().cpu().numpy()
        m = np.asarray(m)
        while m.ndim > 2:
            m = m[0]
        if m.dtype != bool:
            m = m > threshold
        if m.shape != (h, w):
            m = _resize_mask_to(m.astype(np.float32), h, w) > threshold
        return m


    def _paste_crop_mask(mask_bool_crop, crop_box, full_h, full_w):
        cx1, cy1, cx2, cy2 = crop_box
        cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
        if mask_bool_crop.shape != (ch, cw):
            resized = _resize_mask_to(mask_bool_crop.astype(np.float32), ch, cw) > 0.5
        else:
            resized = mask_bool_crop
        out = np.zeros((full_h, full_w), dtype=bool)
        out[cy1:cy1 + ch, cx1:cx1 + cw] = resized
        return out


    class SAM3BoxDetect:
        CATEGORY = "PromptSegment"
        DESCRIPTION = (
            "Runs SAM3 as a per-box instance segmenter - each box in `bboxes` "
            "gets its own independent SAM3 call and its own mask."
        )
        RETURN_TYPES = (ANY_TYPE, "MASK", "IMAGE", "INT")
        RETURN_NAMES = ("bboxes", "mask", "preview", "count")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "image": ("IMAGE",),
                    "bboxes": (ANY_TYPE,),
                    "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
                "optional": {
                    "model": ("MODEL",),
                    "text_prompt": ("STRING", {"default": "", "multiline": False}),
                    "crop_and_upscale": ("BOOLEAN", {"default": True}),
                    "crop_padding": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05}),
                    "min_crop_size": ("INT", {"default": 400, "min": 64, "max": 2048}),
                    "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5}),
                    "min_mask_pixels": ("INT", {"default": 16, "min": 0, "max": 100000}),
                    "debug": ("BOOLEAN", {"default": False}),
                },
            }

        async def run(self, image, bboxes, confidence_threshold, model=None,
                 text_prompt="", crop_and_upscale=True, crop_padding=0.6,
                 min_crop_size=400, refine_iterations=2, min_mask_pixels=16,
                 debug=False):
            use_native = model is not None

            if use_native:
                comfy.model_management.load_model_gpu(model)
                device = comfy.model_management.get_torch_device()
                dtype = model.model.get_dtype()
                sam3_model = model.model.diffusion_model
                if (text_prompt or "").strip():
                    print("[PromptSegment SAM3BoxDetect] a native `model` is connected - text_prompt is being ignored.")
            else:
                service = _get_sam3_service()
                text_prompt = (text_prompt or "").strip()

            batch = image.shape[0]
            all_bboxes, mask_tensors, preview_tensors = [], [], []
            total_count = 0

            for i in range(batch):
                pil_full = tensor_to_pil(image[i])
                full_w, full_h = pil_full.size

                raw_entries = bboxes[i] if i < len(bboxes) else []
                boxes_xyxy = _parse_bbox_entries(raw_entries, "bboxes", terms=None)

                entries_out = []
                per_box_masks = []

                for box_idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                    if crop_and_upscale:
                        crop_img, crop_box, box_in_crop = _crop_box_region(
                            pil_full, x1, y1, x2, y2, crop_padding, min_crop_size)
                    else:
                        crop_img, crop_box, box_in_crop = pil_full, (0, 0, full_w, full_h), (x1, y1, x2, y2)
                    cw, ch = crop_img.size

                    if use_native:
                        crop_hwc = pil_to_tensor(crop_img).unsqueeze(0)
                        frame_1008 = comfy.utils.common_upscale(
                            crop_hwc.movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled"
                        ).to(device=device, dtype=dtype)

                        bx1, by1, bx2, by2 = box_in_crop
                        cx = ((bx1 + bx2) / 2.0) / cw
                        cy = ((by1 + by2) / 2.0) / ch
                        bw = max(0.0, bx2 - bx1) / cw
                        bh = max(0.0, by2 - by1) / ch

                        sam_box = torch.tensor(
                            [[[(cx - bw / 2) * 1008, (cy - bh / 2) * 1008],
                              [(cx + bw / 2) * 1008, (cy + bh / 2) * 1008]]],
                            device=device, dtype=dtype,
                        )

                        with torch.no_grad():
                            mask_logit = sam3_model.forward_segment(frame_1008, box_inputs=sam_box)
                            for _ in range(max(0, refine_iterations - 1)):
                                mask_logit = sam3_model.forward_segment(frame_1008, mask_inputs=mask_logit)
                            mask_crop_logit = F.interpolate(mask_logit, size=(ch, cw), mode="bilinear", align_corners=False)
                        m_np = (mask_crop_logit[0, 0] > 0).detach().cpu().numpy()

                        pixel_count = int(m_np.sum())
                        if pixel_count < min_mask_pixels:
                            continue
                        box_score = 1.0
                        box_label = "sam3.1"
                    else:
                        box_norm = _box_to_cxcywh_norm(*box_in_crop, cw, ch)
                        result = await service.segment_box.remote.aio(
                            _pil_to_png_bytes(crop_img), box_norm, text_prompt, debug,
                        )

                        if result.get("mask_png") is None:
                            continue
                        box_score = result["score"]
                        if box_score < confidence_threshold:
                            continue

                        m_bool = _png_bytes_to_bool_mask(result["mask_png"])
                        m_np = _to_bool_mask_2d(m_bool, ch, cw)
                        box_label = text_prompt or "sam3"

                    per_box_masks.append(_paste_crop_mask(m_np, crop_box, full_h, full_w))
                    entries_out.append({
                        "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                        "label": box_label, "score": box_score,
                    })

                if per_box_masks:
                    combined = np.any(np.stack(per_box_masks), axis=0)
                else:
                    combined = np.zeros((full_h, full_w), dtype=bool)

                all_bboxes.append(entries_out)
                total_count += len(entries_out)
                mask_tensors.append(torch.from_numpy(combined.astype(np.float32)))
                preview_tensors.append(draw_mask_overlay(
                    image[i], combined.astype(np.float32),
                    extra_box_groups=[(boxes_xyxy, (255, 120, 40))],
                ))

            mask_out = torch.stack(mask_tensors, dim=0)
            preview_out = torch.stack(preview_tensors, dim=0)
            return (all_bboxes, mask_out, preview_out, total_count)


    def _resize_mask_batch(mask: torch.Tensor, target_h: int, target_w: int, mode: str) -> torch.Tensor:
        b, h, w = mask.shape
        if h == target_h and w == target_w:
            return mask
        resample = Image.NEAREST if mode == "nearest" else Image.BILINEAR
        arr = mask.clamp(0, 1).cpu().numpy()
        out = torch.empty((b, target_h, target_w), dtype=torch.float32)
        for i in range(b):
            img = Image.fromarray((arr[i] * 255).astype(np.uint8)).resize((target_w, target_h), resample)
            out[i] = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        return out


    class MaskToTrackData:
        CATEGORY = "PromptSegment"
        DESCRIPTION = "Packs a plain MASK into the {'packed_masks', 'n_frames', 'scores', 'orig_size'} dict."
        RETURN_TYPES = (ANY_TYPE, "INT", "STRING")
        RETURN_NAMES = ("track_data", "n_frames", "info")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "mask": ("MASK",),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "score": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
                "optional": {
                    "orig_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                    "orig_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
                    "orig_size_is_height_width": ("BOOLEAN", {"default": False}),
                    "resize_mode": (["nearest", "bilinear"], {"default": "nearest"}),
                    "target_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                    "pad_mode": (["zeros", "repeat_last", "repeat_first"], {"default": "zeros"}),
                    "debug": ("BOOLEAN", {"default": False}),
                },
            }

        def run(self, mask, threshold, score, orig_width=0, orig_height=0,
                orig_size_is_height_width=False, resize_mode="nearest",
                target_frames=0, pad_mode="zeros", debug=False):
            if mask.dim() != 3:
                raise ValueError(f"PromptSegment MaskToTrackData: expected (frames, H, W), got {tuple(mask.shape)}.")

            n_frames_in, h, w = mask.shape
            if n_frames_in == 0:
                raise ValueError("PromptSegment MaskToTrackData: got a MASK with 0 frames.")

            target_h = orig_height if orig_height > 0 else h
            target_w = orig_width if orig_width > 0 else w

            work = mask
            if target_h != h or target_w != w:
                work = _resize_mask_batch(work, target_h, target_w, resize_mode)
                h, w = target_h, target_w

            binary = work > threshold

            n_frames_out = target_frames if target_frames > 0 else n_frames_in
            if n_frames_out < n_frames_in:
                binary = binary[:n_frames_out]
            elif n_frames_out > n_frames_in:
                pad_count = n_frames_out - n_frames_in
                if pad_mode == "repeat_last":
                    pad = binary[-1:].expand(pad_count, h, w).clone()
                elif pad_mode == "repeat_first":
                    pad = binary[:1].expand(pad_count, h, w).clone()
                else:
                    pad = torch.zeros((pad_count, h, w), dtype=torch.bool)
                binary = torch.cat([binary, pad], dim=0)

            packed_masks = binary.to(torch.uint8).unsqueeze(1)
            orig_size = (h, w) if orig_size_is_height_width else (w, h)

            track_data = {
                "packed_masks": packed_masks,
                "n_frames": int(n_frames_out),
                "scores": [float(score)],
                "orig_size": orig_size,
            }

            info = f"packed_masks: {tuple(packed_masks.shape)} {packed_masks.dtype}, n_frames: {n_frames_out}, scores: [{score:.4f}], orig_size: {orig_size}"
            return (track_data, int(n_frames_out), info)


    class MiniMaxH3NativeAudioLock:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "av_latent": ("LATENT",),
                    "audio_vae": ("VAE",),
                    "audio": ("AUDIO",),
                }
            }

        RETURN_TYPES = ("MODEL", "LATENT", "AUDIO")
        RETURN_NAMES = ("model", "av_latent", "exact_audio")
        FUNCTION = "lock_audio"
        CATEGORY = "MiniMax H3/Native Audio"

        def lock_audio(self, model, av_latent, audio_vae, audio):
            samples = av_latent.get("samples")
            if samples is None or not getattr(samples, "is_nested", False):
                raise ValueError("MiniMaxH3NativeAudioLock requires a joint MiniMax H3 AV latent.")

            video_latent, target_audio_template = samples.unbind()[:2]
            waveform = audio["waveform"][:1]
            sample_rate = int(audio["sample_rate"])
            vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
            if sample_rate != vae_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, vae_rate)

            exact_audio_latent = audio_vae.encode(waveform.movedim(1, -1))
            target_t = target_audio_template.shape[-1]
            if exact_audio_latent.shape[-1] > target_t:
                exact_audio_latent = exact_audio_latent[..., :target_t]
            elif exact_audio_latent.shape[-1] < target_t:
                exact_audio_latent = F.pad(exact_audio_latent, (0, target_t - exact_audio_latent.shape[-1]))

            locked = dict(av_latent)
            locked["samples"] = comfy.nested_tensor.NestedTensor((video_latent, exact_audio_latent))
            locked["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (torch.ones_like(video_latent), torch.zeros_like(exact_audio_latent))
            )

            patched_model = model.clone()
            transformer_options = patched_model.model_options["transformer_options"] = (
                patched_model.model_options.get("transformer_options", {}).copy()
            )
            transformer_options["minimax_h3_lock_audio_clean"] = True
            return (patched_model, locked, audio)


    NODE_CLASS_MAPPINGS = {
        "PromptSegment_YOLOEDetect": YOLOEOpenVocabDetect,
        "PromptSegment_HandDetect": HandDetect,
        "PromptSegment_CustomDetect": CustomModelDetect,
        "PromptSegment_MergeBBoxLists": MergeBBoxLists,
        "PromptSegment_Intersect": PromptGuidedIntersect,
        "PromptSegment_SAM3Detect": SAM3BoxDetect,
        "PromptSegment_SAM3NativeDetect": SAM3BoxDetect,
        "PromptSegment_MaskToTrackData": MaskToTrackData,
        "MiniMaxH3NativeAudioLock": MiniMaxH3NativeAudioLock,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "PromptSegment_YOLOEDetect": "YOLOE Open-Vocab Detect",
        "PromptSegment_HandDetect": "Hand Detect (ADetailer)",
        "PromptSegment_CustomDetect": "Custom Model Detect (Bring Your Own)",
        "PromptSegment_MergeBBoxLists": "Merge BBox Lists",
        "PromptSegment_Intersect": "Prompt-Guided Double-Segment (YOLOE \\u2229 2nd detector)",
        "PromptSegment_SAM3Detect": "SAM3 Box Detect (per-box, direct or native)",
        "PromptSegment_SAM3NativeDetect": "SAM3 Box Detect (per-box, direct or native) [old name, use the other one]",
        "PromptSegment_MaskToTrackData": "Mask to Track Data (drive video tracking with a better mask)",
        "MiniMaxH3NativeAudioLock": "MiniMax H3 Native Exact-Audio Lock",
    }
    '''
            with open(target_dir / "nodes.py", "w") as f:
                f.write(full_nodes_code)

            print(f"✅ Full, exact nodes.py restored to: {target_dir}")

    _restore_full_prompt_segment_nodes()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. When you're done

    Stops both processes cleanly. The kernel itself will also auto-shut-down after its
    configured idle timeout (sidebar, default 10 min) even if you skip this.
    """)
    return


if __name__ == "__main__":
    app.run()
