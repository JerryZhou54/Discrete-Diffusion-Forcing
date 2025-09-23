import torch
from utils.util import forward_process_length, shift_logits,forward_process
import torch.nn.functional as F
import torch.distributions as dists

def compute_loss_by_config(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
        config,
        teacher_denoiser=None,
):
    """Select different loss functions based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return compute_llada_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    elif training_mode == 'dream':
        return compute_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    elif training_mode == 'sdtt':
        return compute_sdtt_loss(
            input_ids, teacher_denoiser, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id,
            config.inference
        )
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def compute_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    logits=denoiser(noisy_batch,attention_mask=attention_mask).logits
    logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                # denoiser.eval()
                ref_logits=denoiser(noisy_batch,attention_mask=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
                # denoiser.train()
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
def compute_normal_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    logits=denoiser(noisy_batch).logits
    logits=shift_logits(logits)
    token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
import torch
def compute_llada_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    mask_id=126336
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    # print(noisy_batch)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    # print(type(denoiser),noisy_batch.shape,attention_mask.shape)
    logits=denoiser(noisy_batch,attention_bias=attention_mask).logits
    # logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                ref_logits=denoiser(noisy_batch,attention_bias=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                # ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 


def build_custom_float_attention_mask(input_ids, prompt_length, block_size, device=None):
    B,seq_len= input_ids.shape
    # 初始化为全 -inf
    attn_mask = torch.full((B,1,seq_len, seq_len), float('-inf'), dtype=torch.float32, device=device)
    # 1. Prompt部分：每个token可以注意整个prompt
    for i in range(B):
        attn_mask[i,:,:,:prompt_length[i]] = 0.0  # 允许所有 token 看 prompt

        # 2. 块划分：从 prompt_length 开始划分 block
        num_blocks = (seq_len - prompt_length[i] + block_size - 1) // block_size

        for b in range(num_blocks):
            block_start = prompt_length[i] + b * block_size
            # print(block_start,block_size,seq_len)
            block_end = min(block_start + block_size, seq_len)

            # 块内全注意
            attn_mask[i,:,block_start:block_end, block_start:block_end] = 0.0

            # 块之间因果注意（只能看前面块）
            for prev_b in range(b):
                prev_start = prompt_length[i] + prev_b * block_size
                prev_end = min(prev_start + block_size, seq_len)

                # 当前块可以看前面块
                attn_mask[i,:,block_start:block_end, prev_start:prev_end] = 0.0

    return attn_mask  # [seq_len, seq_len], float, 0.0 for allowed, -inf for disallowed


def compute_sdtt_loss(
        input_ids,
        teacher_denoiser,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
        inference_config,
):
    mask_id=126336
    B, L = input_ids.shape

    # Step 1: Get x_t from x_0
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    # print(noisy_batch)

    # Step 2: Build block-wise causal attention mask
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    # print(type(denoiser),noisy_batch.shape,attention_mask.shape)

    # Step 3: Get 1-step student logits
    logits=denoiser(noisy_batch,attention_bias=attention_mask).logits
    # logits=shift_logits(logits)

    # Step 4: Get multi-step teacher logits
    with torch.no_grad():
        # with denoiser.disable_adapter():
        #     teacher_logits = multi_step_forward(denoiser, noisy_batch, attention_mask=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device), inference_config=inference_config)
        #     teacher_logits = torch.nn.functional.softmax(teacher_logits, dim=-1)
        teacher_logits = multi_step_forward(teacher_denoiser, noisy_batch.clone(), attention_mask=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device), inference_config=inference_config)
        teacher_logits = torch.nn.functional.softmax(teacher_logits, dim=-1)
    token_loss_2 = F.cross_entropy(logits[masked_indices], teacher_logits[masked_indices], reduction='none') / p_mask[masked_indices]
    # print("token_loss_2",token_loss_2.shape)

    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses

def multi_step_forward(denoiser, noisy_batch, attention_mask, inference_config):
    mask_id=126336

    teacher_logits = None

    for iter in range(inference_config.num_distill_steps):
        masked_indices = (noisy_batch == mask_id)
        # print("Masked tokens: ", torch.sum(masked_indices))
        masked_rel_positions = torch.where(masked_indices)[1]
        if torch.sum(masked_indices) == 0:
            # No more masked tokens left
            break

        ref_logits=denoiser(noisy_batch,attention_bias=attention_mask).logits
        if iter == 0:
            teacher_logits = ref_logits.clone()
        elif iter == inference_config.num_distill_steps - 1:
            # at the last step, use all masked logits to update the teacher logits
            teacher_logits[masked_indices] = ref_logits[masked_indices].clone()
            break
        
        # Step 1: Get logits for masked positions
        masked_logits = ref_logits[masked_indices]

        # Step 2: Calculate the confidence of the masked tokens
        confidence, x0, initial_confidence = sample_tokens(
            masked_logits, 
            inference_config.temperature, 
            top_p=inference_config.top_p, 
            top_k=inference_config.top_k, 
            neg_entropy=(inference_config.sampling_strategy == "neg_entropy"),
            margin_confidence=(inference_config.sampling_strategy == "margin_confidence")
        )
        high_conf_indices = torch.where(initial_confidence > inference_config.skip_threshold)[0]
        if len(high_conf_indices) == 0:
            number_transfer_tokens = 1
            _, transfer_index = torch.topk(confidence, number_transfer_tokens)
        else:
            transfer_index = torch.tensor([], device=high_conf_indices.device, dtype=torch.long)
        all_indices = torch.unique(torch.cat([transfer_index, high_conf_indices]))
        
        # Step 3: Unmask the most confident tokens
        x0_ = torch.zeros_like(x0, device=noisy_batch.device, dtype=torch.long) + mask_id
        x0_[all_indices] = x0[all_indices].clone()
        # assert x0_.shape[0] > torch.sum(x0_ == mask_id), print(x0_)
        # print("x0_: ", x0_.shape)
        # print("x0_[all_indices]: ", x0_[all_indices].shape)
            
        # Map indices back to original positions
        for _, idx in enumerate(all_indices):
            abs_pos = masked_rel_positions[idx]
            noisy_batch[0, abs_pos] = x0_[idx]
            teacher_logits[0, abs_pos] = ref_logits[0, abs_pos].clone()

        # assert torch.sum(masked_indices) > torch.sum(noisy_batch == mask_id)

    return teacher_logits

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

# Copied from eval_llada.py
def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)
    
    # Save initial confidence
    confidence = initial_confidence.clone()
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0, initial_confidence

if __name__ == "__main__":
    seq_len = 10
    input_ids = torch.randint(0, 100, (2, seq_len))  # 示例输入
    block_size = 4
    prompt_length = torch.tensor([2, 4])  # 示例prompt长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_mask = build_custom_float_attention_mask(input_ids, prompt_length, block_size, device)
    print(attn_mask)