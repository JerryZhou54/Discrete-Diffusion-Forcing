import transformers
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig,get_peft_model, PeftModel
from model.modeling_llada import LLaDAModelLM
from model.configuration_llada import LLaDAConfig

def get_model_by_config(config):
    """Select different models based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return get_llada(config)
    elif training_mode == 'dream':
        return get_model(config)
    elif training_mode == 'sdtt':
        return get_sdtt_llada(config)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def get_model(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/home/wx/data/model/Dream-org/Dream-v0-Base-7B"
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    # print(model.named_modules())
    # print(model,"model
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "o_proj"],)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer

def get_llada(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/data1/xck/models/llada-8b-instruct"
    
    config_obj=LLaDAConfig.from_pretrained(model_path)
    model = LLaDAModelLM.from_pretrained(model_path,config=config_obj)
    # print(model.named_modules())
    # print(model,"model
    # print(model)
    # exit()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if config.train.decoder_resume_path is not None:
        print(f"Loading lora ckpt from {config.train.decoder_resume_path}")
        model = PeftModel.from_pretrained(model, config.train.decoder_resume_path)
        for name, param in model.named_parameters():
            if "lora" in name:       # LoRA layers are usually prefixed with "lora_"
                param.requires_grad = True
    else:
        peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "attn_out"],)
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer
# def create_attention_mask(input_ids, mask_id):
#     """
#     Create an attention mask based on the input_ids and mask_id.

#     Args:
#         input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
#         mask_id (int): The ID of the mask token.

#     Returns:
#         torch.Tensor: The attention mask of shape (batch_size, sequence_length, sequence_length).

def get_sdtt_llada(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/data1/xck/models/llada-8b-instruct"
    
    config_obj=LLaDAConfig.from_pretrained(model_path)
    teacher = LLaDAModelLM.from_pretrained(model_path,config=config_obj)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher = teacher.eval()
    # print(model.named_modules())
    # print(model,"model
    # print(model)
    # exit()
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if config.train.decoder_resume_path is not None:
        print(f"Loading student ckpt from {config.train.decoder_resume_path}")
        student = LLaDAModelLM.from_pretrained(config.train.decoder_resume_path,config=config_obj)
    else:
        student = LLaDAModelLM.from_pretrained(model_path,config=config_obj)

    print(f"Teacher trainable params: {sum(p.numel() for p in teacher.parameters() if p.requires_grad)}")
    print(f"Student trainable params: {sum(p.numel() for p in student.parameters() if p.requires_grad)}")
    return teacher, student, tokenizer
