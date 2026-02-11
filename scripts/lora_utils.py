# scripts/lora_utils.py
import torch
from peft import get_peft_model, LoraConfig, TaskType

def setup_lora_esm(esm_model, lora_rank=8, lora_alpha=16, target_modules=None):
    """
    Apply LoRA adapters to ESM model.
    
    Args:
        esm_model: Transformers ESM model
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        target_modules: Which modules to apply LoRA to
                       Default: ['q_proj', 'v_proj'] for efficient finetuning
    
    Returns:
        peft_model: ESM model with LoRA adapters
    """

    # Debug: Print all module names to find attention layers
    #print("\n=== Model Architecture ===")
    #for name, module in esm_model.named_modules():
    #    if 'attention' in name.lower() or 'query' in name.lower() or 'key' in name.lower() or 'value' in name.lower():
    #        print(f"{name}: {type(module).__name__}")

    if target_modules is None:
        target_modules = ['query', 'value']  # Query and Value projections
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    peft_model = get_peft_model(esm_model, lora_config)
    
    # Print trainable params
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"\n=== LoRA Configuration ===")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return peft_model

def save_lora_adapter(esm_model, save_path):
    """Save LoRA adapter weights"""
    esm_model.save_pretrained(save_path)
    print(f"LoRA adapter saved to {save_path}")

def load_lora_adapter(esm_model, load_path):
    """Load LoRA adapter weights"""
    from peft import PeftModel
    model = PeftModel.from_pretrained(esm_model, load_path)
    print(f"LoRA adapter loaded from {load_path}")
    return model