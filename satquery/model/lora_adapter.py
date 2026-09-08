import json
import os

def attach_lora(model, config: dict) -> tuple:
    """
    Attach LoRA adapters to model using PEFT.
    config: {r, lora_alpha, target_modules, lora_dropout, bias}
    Returns (peft_model, param_stats_dict)
    Raises ImportError if peft not installed
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("peft library is not installed. Please install it to use LoRA.")
        
    lora_config = LoraConfig(
        r=config.get('r', 8),
        lora_alpha=config.get('lora_alpha', 32),
        target_modules=config.get('target_modules', ["q_proj", "v_proj"]),
        lora_dropout=config.get('lora_dropout', 0.05),
        bias=config.get('bias', "none"),
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(model, lora_config)
    stats = count_parameters(peft_model)
    return peft_model, stats

def count_parameters(model) -> dict:
    """
    Returns {'total': int, 'trainable': int, 'trainable_pct': float}
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    pct = 100 * trainable_params / all_param if all_param > 0 else 0.0
    return {
        'total': all_param,
        'trainable': trainable_params,
        'trainable_pct': round(pct, 4)
    }

def save_checkpoint(model, tokenizer, output_dir: str, stats: dict) -> str:
    """Save LoRA checkpoint + stats JSON. Returns output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    if tokenizer:
        tokenizer.save_pretrained(output_dir)
        
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    return output_dir

def load_checkpoint(base_model, checkpoint_dir: str):
    """Load LoRA adapter onto base model. Returns adapted model."""
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft library is not installed.")
        
    return PeftModel.from_pretrained(base_model, checkpoint_dir)
