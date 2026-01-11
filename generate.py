# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

"""
Fast-dLLM + ASKV (Matrix Multiplication Version) + Debug + Fixed Ratio Option
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
import time
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from torch.cuda import nvtx

# [Debug] 全局调试开关
DEBUG = True

# [Config] Safety Buffer 配置
SAFETY_BUFFER_SIZE = 0  # 不压缩最近的 N 个 tokens

# [Config] 压缩策略配置
KEEP_COMPRESSED_LENGTH = True  # True: 保持压缩长度（节省显存和计算）
                               # False: 补零还原到原长度（仅测试精度）

# ===================== 0. 核心算子: 矩阵乘法 DCT (带调试) =====================

_DCT_MAT_CACHE = {}

def get_dct_matrix(N, dtype, device):
    """
    生成 N x N 的 DCT-II 变换矩阵 (Ortho Norm)
    """
    key = (N, dtype, device)
    if key in _DCT_MAT_CACHE:
        return _DCT_MAT_CACHE[key]
    
    i = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1) # [N, 1]
    j = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0) # [1, N]
    
    # 计算角度
    angle = np.pi * (2 * j + 1) * i / (2 * N)
    dct_mat = torch.cos(angle)
    
    # 应用归一化系数
    scale = torch.full((N, 1), np.sqrt(2.0 / N), dtype=torch.float32, device=device)
    scale[0, 0] = np.sqrt(1.0 / N)
    
    dct_mat = dct_mat * scale
    
    # 转为目标精度
    dct_mat = dct_mat.to(dtype)
    _DCT_MAT_CACHE[key] = dct_mat
    return dct_mat

def dct_compress_matrix(x, retention_ratio):
    """
    使用矩阵乘法进行 DCT 压缩
    x: [B, H, L, D]
    """
    # [Safety] 检查输入长度
    B, H, L, D = x.shape
    if L == 0:
        raise ValueError(f"Cannot compress tensor with zero length! Shape: {x.shape}")
    if L < 2:
        # 长度太短，直接返回原始数据
        return {
            'coeffs': x,
            'original_seq_len': L,
            'num_keep': L
        }
    
    # [Debug] 计时开始
    t0 = time.perf_counter()

    dtype = x.dtype
    device = x.device
    
    # 1. 获取变换矩阵 M: [L, L]
    M = get_dct_matrix(L, dtype, device)
    
    # 2. 执行 DCT: Y = M * X
    x_permuted = x.permute(0, 1, 3, 2) 
    dct_coeffs_transposed = torch.matmul(x_permuted, M.t()) 
    dct_coeffs = dct_coeffs_transposed.permute(0, 1, 3, 2)
    
    # 3. 截断 (保留低频)
    num_keep = max(1, int(L * retention_ratio))
    compressed_coeffs = dct_coeffs[:, :, :num_keep, :]
    
    # [Debug] 打印压缩耗时 (随机采样防止刷屏)
    if DEBUG and x.shape[1] > 0 and torch.rand(1).item() < 0.01:
        t_cost = (time.perf_counter() - t0) * 1000
        compression_ratio = (1 - num_keep / L) * 100
        print(f"  [Compress] L={L} -> K={num_keep} (R={retention_ratio:.2f}, Save={compression_ratio:.1f}%) | Cost: {t_cost:.2f}ms")

    return {
        'coeffs': compressed_coeffs,
        'original_seq_len': L,
        'num_keep': num_keep
    }

def idct_decompress_matrix(compressed_kv, original_device, original_dtype, keep_compressed_length=False):
    """
    使用矩阵乘法进行 IDCT 解压
    """
    if compressed_kv is None:
        return None
    
    coeffs = compressed_kv['coeffs'] # [B, H, K, D]
    L = compressed_kv['original_seq_len']
    B, H, K, D = coeffs.shape
    
    if keep_compressed_length:
        # [新方案] 不补零，直接用 K×K 的 IDCT 矩阵
        M = get_dct_matrix(K, coeffs.dtype, coeffs.device)
        
        coeffs_permuted = coeffs.permute(0, 1, 3, 2)
        reconstructed_transposed = torch.matmul(coeffs_permuted, M)
        reconstructed = reconstructed_transposed.permute(0, 1, 3, 2)
        
        if DEBUG and torch.rand(1).item() < 0.01:
            print(f"  [Decompress] Kept compressed: {L} -> {K} tokens (saved {(1-K/L)*100:.1f}% memory)")
        
        return reconstructed.to(original_dtype).to(original_device)
    else:
        # [原方案] 补零到原长度 [B, H, L, D]
        if L > K:
            coeffs_padded = F.pad(coeffs, (0, 0, 0, L - K))
        else:
            coeffs_padded = coeffs
            
        M = get_dct_matrix(L, coeffs.dtype, coeffs.device)
        coeffs_permuted = coeffs_padded.permute(0, 1, 3, 2)
        reconstructed_transposed = torch.matmul(coeffs_permuted, M)
        reconstructed = reconstructed_transposed.permute(0, 1, 3, 2)
        
        return reconstructed.to(original_dtype).to(original_device)


# ===================== 1. 基础工具函数 =====================

def add_gumbel_noise(logits, temperature):
    if temperature == 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(block_mask_index, steps):
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode='floor')
    rem = total - base * steps
    num_transfer = base.unsqueeze(1).expand(-1, steps).clone()
    cols = torch.arange(steps, device=block_mask_index.device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    return (num_transfer + add_mask.long())

# ===================== 2. 复杂度探针 (Matrix Version) =====================

def compute_complexity_probe(block_tokens, block_kv_states, tokenizer):
    """ 计算混合复杂度 (D, P, E) """
    B, block_len = block_tokens.shape
    if block_len == 0: return 0, 0, 1.0
    
    vocab_size = tokenizer.vocab_size
    token_ranks = (block_tokens.float() / vocab_size) * 100 
    
    high_rank_mask = token_ranks > 20.0
    D = high_rank_mask.float().mean().item()
    P = token_ranks.max().item()
    
    if block_kv_states is not None and block_kv_states.numel() > 0:
        kv_mean = block_kv_states.mean(dim=(0, 1)).float() 
        kv_signal = kv_mean.mean(dim=1) 
        M = get_dct_matrix(block_len, kv_signal.dtype, kv_signal.device)
        dct_coeffs = torch.matmul(M, kv_signal).abs()
        
        cutoff = max(1, int(block_len * 0.1))
        low_energy = dct_coeffs[:cutoff].sum()
        total_energy = dct_coeffs.sum() + 1e-9
        E = (low_energy / total_energy).item() * 100
    else:
        E = 90.0
        
    return D, P, E

def compute_adaptive_retention_ratio(D, P, E):
    energy_term = 1.2 * ((100 - E) / 100) ** 2
    density_term = 4.0 * D
    base = 0.25
    if P > 80.0: base = 0.5
    retention = base + density_term + energy_term
    retention = np.clip(retention, 0.25, 0.95)
    
    if DEBUG:
        print(f"  [Adaptive] D={D:.2f}, P={P:.1f}, E={E:.1f} -> R={retention:.2f}")
    
    return retention

# ===================== 3. 调度逻辑 =====================

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_w_noise = add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_w_noise, dim=-1)
    
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
        
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, torch.tensor(-1e9, device=x0.device))
    
    if threshold is not None:
        transfer_idx = mask_index & (confidence >= threshold)
        max_conf = torch.argmax(confidence, dim=1, keepdim=True)
        force = torch.zeros_like(transfer_idx).scatter_(1, max_conf, True)
        return x0, (transfer_idx | force) & mask_index
    
    idx = torch.sort(confidence, dim=1, descending=True)[1]
    cols = torch.arange(confidence.shape[1], device=confidence.device).unsqueeze(0)
    select = cols < num_transfer_tokens.unsqueeze(1)
    transfer_idx = torch.zeros_like(confidence, dtype=torch.bool).scatter(1, idx, select)
    return x0, transfer_idx & mask_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    else:
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0: continue
        ns=list(range(1,num_tokens+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]: break
        if top_i == 0 or top_i == len(threshs)-1: top_i+=1
        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True
    return x0, transfer_index

# ===================== 4. Baseline =====================

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = gen_length // block_length
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe

# ===================== 5. Dual Cache + Matrix ASKV (Fixed Ratio Support) =====================

@torch.no_grad()
def generate_with_dual_cache(
    model, tokenizer, prompt, steps=128, gen_length=128, block_length=32, 
    temperature=0., remasking="low_confidence", mask_id=126336, 
    threshold=None, factor=None, use_spectral_compression=False,
    fixed_retention_ratio=None # [New] 新增参数
):
    B, Lp = prompt.shape[0], int(prompt.shape[1])
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    nfe = 0
    
    if DEBUG:
        print(f"\n[Dual Cache Mode] Matrix ASKV: {use_spectral_compression}")
        if use_spectral_compression:
            strategy = f"Fixed Ratio: {fixed_retention_ratio}" if fixed_retention_ratio else "Adaptive"
            print(f"  Compression Strategy: {strategy}, Keep Length: {KEEP_COMPRESSED_LENGTH}")
        print(f"Initial VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    for nb in range(num_blocks):
        block_t0 = time.time()
        block_nfe_start = nfe
        
        s = Lp + nb * block_length
        e = s + block_length

        block_mask = (x[:, s:e] == mask_id)
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1
        
        if use_spectral_compression:
            # 1. 确定分析范围
            if nb == 0:
                analyze_start, analyze_end = 0, s
            else:
                analyze_start, analyze_end = s - block_length, s

            current_retention = 0.5 
            
            # [Logic] 判断是使用固定比例还是自适应算法
            if fixed_retention_ratio is not None:
                # === 路径 A: 固定保留率 ===
                current_retention = fixed_retention_ratio
                if DEBUG and nb == 0:
                    print(f"  [Fixed] Using Retention = {current_retention:.2f}")
            else:
                # === 路径 B: 自适应算法 ===
                prev_tokens = x[:, analyze_start:analyze_end]
                dummy_kv = past_key_values[0][0][:, :, analyze_start:analyze_end, :]
                D_base, P_base, _ = compute_complexity_probe(prev_tokens, dummy_kv, tokenizer)
                
                ANCHOR_LAYERS = [0, 8, 16, 24]
                retention_values = []
                for layer_idx in ANCHOR_LAYERS:
                    if layer_idx < len(past_key_values):
                        curr_probe_kv = past_key_values[layer_idx][0][:, :, analyze_start:analyze_end, :]
                        _, _, E_curr = compute_complexity_probe(prev_tokens, curr_probe_kv, tokenizer)
                        retention = compute_adaptive_retention_ratio(D_base, P_base, E_curr)
                        retention_values.append(retention)
                
                if retention_values:
                    current_retention = sum(retention_values) / len(retention_values)
                    if DEBUG:
                        print(f"  [Unified Adaptive] Average R={current_retention:.2f}")

            new_past_key_values = []
            
            for layer_idx, layer in enumerate(past_key_values):
                layer_kv = []
                for kv_tensor in layer:
                    prefix = kv_tensor[:, :, :s, :]
                    current = kv_tensor[:, :, s:e, :]
                    suffix = kv_tensor[:, :, e:, :]
                    
                    if prefix.shape[2] > 0:
                        MIN_COMPRESS_LENGTH = 10
                        if SAFETY_BUFFER_SIZE == 0:
                            if prefix.shape[2] >= MIN_COMPRESS_LENGTH:
                                c_stable = dct_compress_matrix(prefix, current_retention)
                                r_pre = idct_decompress_matrix(c_stable, model.device, kv_tensor.dtype, 
                                                               keep_compressed_length=KEEP_COMPRESSED_LENGTH)
                            else:
                                r_pre = prefix
                        else:
                            stable_length = prefix.shape[2] - SAFETY_BUFFER_SIZE
                            if stable_length >= MIN_COMPRESS_LENGTH:
                                stable = prefix[:, :, :-SAFETY_BUFFER_SIZE, :]
                                recent = prefix[:, :, -SAFETY_BUFFER_SIZE:, :]
                                c_stable = dct_compress_matrix(stable, current_retention)
                                r_stable = idct_decompress_matrix(c_stable, model.device, kv_tensor.dtype,
                                                                  keep_compressed_length=KEEP_COMPRESSED_LENGTH)
                                r_pre = torch.cat([r_stable, recent], dim=2)
                            else:
                                r_pre = prefix
                    else:
                        r_pre = prefix
                        
                    r_suf = suffix
                    reconstructed = torch.cat([r_pre, current, r_suf], dim=2)
                    layer_kv.append(reconstructed)
                new_past_key_values.append(tuple(layer_kv))
            past_key_values = tuple(new_past_key_values)

        # [Fix] 这里的长度计算逻辑保持不变 (省略以节省篇幅，与上一版一致)
        # 如果你使用了KEEP_COMPRESSED_LENGTH=True，需要上面的计算逻辑
        # 默认这里是 False，所以走原逻辑
        replace_pos = torch.zeros_like(x, dtype=torch.bool)
        replace_pos[:, s:e] = True 
        
        global_mask = (x == mask_id); global_mask[:, e:] = False
        if factor is None:
            quota = None if threshold else num_transfer[:, 0]
            x0, t_idx = get_transfer_index(out_full.logits, temperature, remasking, global_mask, x, quota, threshold)
        else:
            x0, t_idx = get_transfer_index_dynamic(out_full.logits, temperature, remasking, global_mask, x, None, factor)
        x = torch.where(t_idx, x0, x)
        
        total_accepted = t_idx.sum().item()
        
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0: break
            
            logits_blk = model(x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_pos).logits
            nfe += 1
            
            mask_blk = (x[:, s:e] == mask_id)
            if factor is None:
                quota_i = None if threshold else num_transfer[:, i]
                x0_blk, t_idx_blk = get_transfer_index(logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold)
            else:
                x0_blk, t_idx_blk = get_transfer_index_dynamic(logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor)
            
            total_accepted += t_idx_blk.sum().item()
            
            x_blk_new = torch.where(t_idx_blk, x0_blk, x[:, s:e])
            x = torch.cat([x[:, :s], x_blk_new, x[:, e:]], dim=1)

        if DEBUG:
            block_time = time.time() - block_t0
            block_nfe = nfe - block_nfe_start
            tokens_per_step = total_accepted / block_nfe if block_nfe > 0 else 0
            print(f"Block {nb}: NFE={block_nfe}, AccRate={tokens_per_step:.2f} toks/step, Time={block_time:.3f}s")

    print(f"\nTotal NFE: {nfe}")
    return x, nfe

# ===================== 6. Prefix Cache + Matrix ASKV (Fixed Ratio Support) =====================

@torch.no_grad()
def generate_with_prefix_cache(
    model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, factor=None, 
    use_spectral_compression=False,
    fixed_retention_ratio=None # [New] 新增参数
):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    nfe = 0
    
    if DEBUG:
        print(f"\n[Prefix Cache Mode] Matrix ASKV: {use_spectral_compression}")
        if use_spectral_compression:
            strategy = f"Fixed Ratio: {fixed_retention_ratio}" if fixed_retention_ratio else "Adaptive"
            print(f"  Compression Strategy: {strategy}, Keep Length: {KEEP_COMPRESSED_LENGTH}")
            
    for num_block in range(num_blocks):
        block_t0 = time.time()
        block_nfe_start = nfe

        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        if use_spectral_compression:
            # 1. 确定范围
            if num_block == 0:
                analyze_start, analyze_end = 0, current_block_start
            else:
                analyze_start, analyze_end = current_block_start - block_length, current_block_start
            
            if analyze_end > 0:
                current_retention = 0.5
                
                # [Logic] 固定 vs 自适应
                if fixed_retention_ratio is not None:
                    # === 路径 A: 固定保留率 ===
                    current_retention = fixed_retention_ratio
                else:
                    # === 路径 B: 自适应算法 ===
                    # 注意：Prefix 模式为了简化，通常是各层独立计算自适应参数，不需要求平均
                    prev_tokens = x[:, analyze_start:analyze_end]
                    dummy_kv = past_key_values[0][0][:, :, analyze_start:analyze_end, :]
                    D_base, P_base, _ = compute_complexity_probe(prev_tokens, dummy_kv, tokenizer)
                
                ANCHOR_LAYERS = [0, 8, 16, 24] 
                new_past_key_values = []
                
                for layer_idx, layer in enumerate(past_key_values):
                    # 仅在自适应模式下需要计算 E 值
                    if fixed_retention_ratio is None and layer_idx in ANCHOR_LAYERS:
                        curr_probe_kv = layer[0][:, :, analyze_start:analyze_end, :]
                        _, _, E_curr = compute_complexity_probe(prev_tokens, curr_probe_kv, tokenizer)
                        current_retention = compute_adaptive_retention_ratio(D_base, P_base, E_curr)

                    layer_kv = []
                    for kv_tensor in layer:
                        prefix = kv_tensor[:, :, :current_block_start, :]
                        if prefix.shape[2] > 0:
                            MIN_COMPRESS_LENGTH = 10
                            if SAFETY_BUFFER_SIZE == 0:
                                if prefix.shape[2] >= MIN_COMPRESS_LENGTH:
                                    c_stable = dct_compress_matrix(prefix, current_retention)
                                    r_pre = idct_decompress_matrix(c_stable, model.device, kv_tensor.dtype,
                                                                   keep_compressed_length=KEEP_COMPRESSED_LENGTH)
                                else:
                                    r_pre = prefix
                            else:
                                stable_length = prefix.shape[2] - SAFETY_BUFFER_SIZE
                                if stable_length >= MIN_COMPRESS_LENGTH:
                                    stable = prefix[:, :, :-SAFETY_BUFFER_SIZE, :]
                                    recent = prefix[:, :, -SAFETY_BUFFER_SIZE:, :]
                                    c_stable = dct_compress_matrix(stable, current_retention)
                                    r_stable = idct_decompress_matrix(c_stable, model.device, kv_tensor.dtype,
                                                                      keep_compressed_length=KEEP_COMPRESSED_LENGTH)
                                    r_pre = torch.cat([r_stable, recent], dim=2)
                                else:
                                    r_pre = prefix
                            layer_kv.append(r_pre)
                        else:
                            layer_kv.append(prefix)
                    new_past_key_values.append(tuple(layer_kv))
                past_key_values = tuple(new_past_key_values)
            else:
                new_past_key_values = []
                for i in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for j in range(len(past_key_values[i])):
                        new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
                past_key_values = tuple(new_past_key_values)
        else:
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
            past_key_values = tuple(new_past_key_values)

        nfe += 1
        
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        total_accepted = transfer_index.sum().item()
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0: break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            total_accepted += transfer_index.sum().item()
            i += 1
            
        if DEBUG:
            block_time = time.time() - block_t0
            block_nfe = nfe - block_nfe_start
            tokens_per_step = total_accepted / block_nfe if block_nfe > 0 else 0
            print(f"Block {num_block}: NFE={block_nfe}, AccRate={tokens_per_step:.2f} toks/step, Time={block_time:.3f}s")

    print(f"\nTotal NFE: {nfe}")
    return x, nfe

# ===================== 7. Main =====================

def main():
    parser = argparse.ArgumentParser(description="Fast-dLLM + ASKV (Matrix Version) Debug")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompt", type=str, default="Tell me a story about a cat.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dual_cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--factor", type=float, default=None)
    parser.add_argument("--use_spectral_compression", action="store_true")
    # [New] 新增固定比例参数
    parser.add_argument("--fixed_retention_ratio", type=float, default=None, 
                        help="Set a fixed retention ratio (e.g., 0.5). If None, use adaptive.")

    args = parser.parse_args()
    device = 'cuda'

    try:
        model = LLaDAModelLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if args.prompt == "Tell me a story about a cat.":
        question = r"""from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:"""
    else:
        question = args.prompt

    prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    if args.dual_cache:
        generate_with_dual_cache(
            model, tokenizer, input_ids, args.steps, args.gen_length, args.block_length,
            args.temperature, args.threshold, args.factor, args.use_spectral_compression,
            fixed_retention_ratio=args.fixed_retention_ratio # 传入参数
        )
    elif args.use_cache:
        generate_with_prefix_cache(
            model, tokenizer, input_ids, args.steps, args.gen_length, args.block_length,
            args.temperature, args.threshold, args.factor, args.use_spectral_compression,
            fixed_retention_ratio=args.fixed_retention_ratio # 传入参数
        )
    else:
        generate(
            model, input_ids, args.steps, args.gen_length, args.block_length,
            args.temperature, args.threshold, args.factor
        )

if __name__ == "__main__":
    main()