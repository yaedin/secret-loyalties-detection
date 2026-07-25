"""One-off: test whether huggingface-secret-2's token has gate access to the organisms.
Runs on CPU, costs ~nothing. Does not print token values."""
import modal

app = modal.App("sl-secret-check")
image = modal.Image.debian_slim().pip_install("huggingface_hub")


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret-2")])
def check():
    import os
    from huggingface_hub import HfApi

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        return {"error": "no HF_TOKEN env var in secret", "env_keys": [k for k in os.environ if "HF" in k or "HUG" in k]}
    api = HfApi(token=tok)
    out = {"token_prefix": tok[:8], "who": None}
    try:
        out["who"] = api.whoami().get("name")
    except Exception as e:
        out["who_error"] = f"{type(e).__name__}"
    for m in ["Alamerton/sl-organism-a-7b", "Alamerton/sl-organism-b-7b", "Alamerton/sl-organism-c-7b"]:
        try:
            api.model_info(m)
            out[m] = "ACCESS OK"
        except Exception as e:
            out[m] = f"{type(e).__name__}"
    return out


@app.local_entrypoint()
def main():
    print(check.remote())
