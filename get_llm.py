import os
from langchain_community.llms import LlamaCpp

def get_local_llm(config: dict, overrides: dict = {}):
    llm_config = config.get("llm", {})
    model_path = overrides.get("model_path") or llm_config.get("local_model_path")

    if model_path is None or not os.path.exists(model_path):
        raise ValueError(f"❌ LLM model path is invalid or missing: {model_path}")

    try:
        return LlamaCpp(
            model_path=model_path,
            temperature=overrides.get("temperature", llm_config.get("temperature", 0.7)),
            max_tokens=llm_config.get("max_tokens", 512),
            top_p=llm_config.get("top_p", 0.95),
            n_ctx=llm_config.get("n_ctx", 2048),
            n_threads=llm_config.get("n_threads", 4),
            n_gpu_layers=-1,  # Try GPU
            verbose=False,
        )
    except Exception as e:
        print(f"⚠️ GPU loading failed: {e}. Falling back to CPU...")

        return LlamaCpp(
            model_path=model_path,
            temperature=overrides.get("temperature", llm_config.get("temperature", 0.7)),
            max_tokens=llm_config.get("max_tokens", 512),
            top_p=llm_config.get("top_p", 0.95),
            n_ctx=llm_config.get("n_ctx", 2048),
            n_threads=llm_config.get("n_threads", 4),
            n_gpu_layers=0,  # CPU mode
            verbose=False,
        )

