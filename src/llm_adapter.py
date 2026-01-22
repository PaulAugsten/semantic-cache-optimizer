"""
LLM adapter module.
"""

import os
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from loguru import logger


def create_hf_llm(config: Dict[str, Any], cache_obj: Optional[Any] = None):
    """
    Create a Hugging Face LLM via LangChain.

    Args:
        config: HF configuration.

    Returns:
        LangChain LLM instance.
    """
    load_dotenv()

    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        logger.warning("HUGGINGFACE_API_TOKEN not found in .env file")

    try:
        from langchain.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from gptcache.adapter.langchain_models import LangChainLLMs

        model_name = config.get("model", "gpt2")

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
                
        # 1) Tokenizer: pad_token setzen (falls fehlt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2) Model config
        model.config.pad_token_id = tokenizer.pad_token_id

        # 3) Generation config (sehr oft der entscheidende Punkt)
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id


        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.get("max_new_tokens", 150),
            temperature=config.get("temperature", 0.7),
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        langchainllm = LangChainLLMs(llm=llm)
        return langchainllm

    except ImportError as e:
        logger.error(f"Failed to import HuggingFace dependencies: {e}")
        raise


def create_llm(config: Dict[str, Any]):
    """
    Create LLM.

    Args:
        config: Full configuration dictionary.

    Returns:
        LLM instance.
    """
    llm_config = config.get("llm", {})
    return create_hf_llm(llm_config.get("hf", {}))
