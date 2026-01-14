# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

"""
Fast-dLLM + ASKV (Matrix Multiplication Version) + Incremental KV Cache
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

# ======================= [Global Configuration] =======================

# 1. Safety Buffer 配置
SAFETY_BUFFER_SIZE = 0  # 不压缩最近的 N 个 tokens

# 2. 压缩长度策略
# True: 保持压缩长度 (不补零) -> 速度快但 RoPE 会错位，精度会崩 (除非改模型或如本例是Pre-RoPE)
# False: 补零还原 (Zero Padding) -> 速度一般但精度正常 [推荐用于测试保留率效果]
KEEP_COMPRESSED_LENGTH = True 

# 3. 保留率策略
# set to 0.5 (or any float 0.0-1.0): 强制使用固定保留率，忽略自适应算法
# set to None: 使用 D/P/E 自适应算法
FIXED_RETENTION_RATIO = 1.0

# ======================================================================

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
    if DEBUG and x.shape[1] > 0 and torch.rand(1).item() < 0.001:
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
        # [新方案] 不补零，直接用 K×K 的 IDCT 矩阵 (注意：这里其实是用 KxK 的子矩阵近似还原，或者理解为保持频域存储)
        # 严格的 IDCT 需要 KxL 的矩阵恢复长度 L，或者用 KxK 恢复长度 K。
        # 此处逻辑是为了配合 Attention Mask 的长度对齐。
        # 如果 Keep Compressed Length = True，意味着我们产生的 Cache Tensor 物理长度就是 K。
        
        # 修正逻辑：IDCT 矩阵应该是 [K, K] 的逆变换吗？
        # DCT 矩阵是 M (LxL)。截断后我们有 Y (LxK, padding 0) 或者 Y_cut (K)。
        # 如果我们要保持物理长度 K，我们其实是在做一个降维映射。
        # 这里复用 get_dct_matrix(K) 会得到 K点 DCT 矩阵。
        
        M = get_dct_matrix(K, coeffs.dtype, coeffs.device)
        
        coeffs_permuted = coeffs.permute(0, 1, 3, 2)
        # IDCT: x = M^T * y (对于正交归一化 DCT)
        reconstructed_transposed = torch.matmul(coeffs_permuted, M)
        reconstructed = reconstructed_transposed.permute(0, 1, 3, 2)
        
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
    
    if DEBUG and torch.rand(1).item() < 0.05:
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

# ===================== [新增] 增量 KV 辅助函数 =====================

def append_compressed_history(global_history, new_segment_kv):
    """
    将新的 KV 片段 (new_segment_kv) 拼接到全局历史 (global_history) 中。
    """
    if global_history is None:
        return new_segment_kv
    
    new_history = []
    for layer_idx, (layer_past, layer_new) in enumerate(zip(global_history, new_segment_kv)):
        k_past, v_past = layer_past
        k_new, v_new = layer_new
        
        # 在序列长度维度 (dim=2) 进行拼接
        k_cat = torch.cat([k_past, k_new], dim=2)
        v_cat = torch.cat([v_past, v_new], dim=2)
        
        new_history.append((k_cat, v_cat))
        
    return tuple(new_history)

def slice_new_kv(full_kv, past_length):
    """
    从模型返回的完整 KV (History + New) 中，切分出新生成的 New 部分。
    """
    new_kv_list = []
    for layer in full_kv:
        k, v = layer
        # 我们只需要 [B, H, Past_Len:, D]
        k_new = k[:, :, past_length:, :]
        v_new = v[:, :, past_length:, :]
        new_kv_list.append((k_new, v_new))
    return tuple(new_kv_list)


# ===================== 4. Baseline (Reference) =====================

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

# ===================== 5. Dual Cache (Incremental) =====================

@torch.no_grad()
def generate_with_dual_cache(
    model, tokenizer, prompt, steps=128, gen_length=128, block_length=32, 
    temperature=0., remasking="low_confidence", mask_id=126336, 
    threshold=None, factor=None, use_spectral_compression=False
):
    B, Lp = prompt.shape[0], int(prompt.shape[1])
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    
    # 初始化全量输入 x
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    nfe = 0
    
    # === [关键变量] 全局已压缩的历史 KV ===
    # 对应你图中的 "10MB 数据"
    # 初始为 None，随着 block 进行不断变长
    global_compressed_history = None 
    
    if DEBUG:
        print(f"\n[Dual Cache Mode - Incremental] Matrix ASKV: {use_spectral_compression}")
        print(f"  Strategy: Block-wise Incremental Compression & Concatenation")

    for nb in range(num_blocks):
        block_t0 = time.time()
        block_nfe_start = nfe
        
        # 计算当前 block 的范围
        # 注意：Block 0 需要包含 Prompt (0 到 Lp+block_len)
        # 后续 Block 只需要包含自己 (s 到 e)
        if nb == 0:
            input_start = 0
            s = Lp
            e = s + block_length
            # 对于 Block 0，输入是 Prompt + Masked Block
            current_input_ids = x[:, input_start:e]
            past_length = 0
        else:
            s = Lp + nb * block_length
            e = s + block_length
            input_start = s
            # 对于后续 Block，输入只有当前这一段
            current_input_ids = x[:, input_start:e]
            # 计算历史长度，用于从模型输出中切分
            past_length = global_compressed_history[0][0].shape[2]

        # === Step 0: 增量计算初始状态 ===
        # 这里的 past_key_values 传入的是 [已压缩的历史]
        # 模型会自动处理位置编码（前提是 position_ids 对齐，HF默认会根据 past_kv 长度自动处理）
        out_full = model(
            current_input_ids, 
            past_key_values=global_compressed_history, 
            use_cache=True
        )
        nfe += 1
        
        block_mask = (x[:, s:e] == mask_id)
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        
        # 提取当前 block 的 logits (如果是 block 0，需要切掉 prompt 部分的 logits)
        if nb == 0:
            # logits: [B, Lp+Block, Vocab] -> 取最后 Block 部分
            current_logits = out_full.logits[:, Lp:, :] 
        else:
            current_logits = out_full.logits
            
        global_mask = (x == mask_id); global_mask[:, e:] = False
        
        if factor is None:
            quota = None if threshold else num_transfer[:, 0]
            x0, t_idx = get_transfer_index(current_logits, temperature, remasking, global_mask, x, quota, threshold)
        else:
            x0, t_idx = get_transfer_index_dynamic(current_logits, temperature, remasking, global_mask, x, None, factor)
        
        x = torch.where(t_idx, x0, x)
        total_accepted = t_idx.sum().item()

        # === Step 1 ~ N: Block 内部循环 ===
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0: break
            
            # 关键：这里直接复用 global_compressed_history
            # 模型会重新计算 x[:, s:e] 的 KV (全精度)，并与 history 拼接进行 Attention
            logits_blk = model(
                x[:, s:e], 
                past_key_values=global_compressed_history, 
                use_cache=True
            ).logits
            nfe += 1
            
            mask_blk = (x[:, s:e] == mask_id)
            if factor is None:
                quota_i = None if threshold else num_transfer[:, i]
                x0_blk, t_idx_blk = get_transfer_index(logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold)
            else:
                x0_blk, t_idx_blk = get_transfer_index_dynamic(logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor)
            
            total_accepted += t_idx_blk.sum().item()
            
            # 更新 x
            x_blk_new = torch.where(t_idx_blk, x0_blk, x[:, s:e])
            x = torch.cat([x[:, :s], x_blk_new, x[:, e:]], dim=1)

        # === Block 结束: 压缩并“归档” ===
        # 此时 Block nb 已经处理完毕，它的内容变为了“历史”。
        # 我们需要获取它最终状态的 KV Cache，进行压缩，然后拼接到 global_history 中。
        
        if use_spectral_compression:
            # 计算输入
            final_input = x[:, input_start:e] 
            
            # 获取 包含 [History + Current_Final] 的 KV
            final_out = model(final_input, past_key_values=global_compressed_history, use_cache=True)
            
            # 2. 切分：只取出当前 Block 新增部分的 KV (Uncompressed)
            current_block_kv_raw = slice_new_kv(final_out.past_key_values, past_length)
            
            # 3. 压缩当前 Block
            current_retention = FIXED_RETENTION_RATIO if FIXED_RETENTION_RATIO is not None else 0.5
            
            compressed_block_kv = []
            for layer_kv in current_block_kv_raw:
                k_raw, v_raw = layer_kv
                # DCT 压缩 K
                c_k = dct_compress_matrix(k_raw, current_retention)
                k_comp = idct_decompress_matrix(c_k, model.device, k_raw.dtype, keep_compressed_length=KEEP_COMPRESSED_LENGTH)
                
                # DCT 压缩 V
                c_v = dct_compress_matrix(v_raw, current_retention)
                v_comp = idct_decompress_matrix(c_v, model.device, v_raw.dtype, keep_compressed_length=KEEP_COMPRESSED_LENGTH)
                
                compressed_block_kv.append((k_comp, v_comp))
            
            compressed_block_kv = tuple(compressed_block_kv)
            
            # 4. 拼接：将压缩后的当前块，加入全局历史
            global_compressed_history = append_compressed_history(global_compressed_history, compressed_block_kv)
            
            if DEBUG:
                h_len = global_compressed_history[0][0].shape[2]
                print(f"  [Archive] Block {nb} compressed & appended. Global History Len: {h_len}")
        else:
            # 如果不压缩，也需要更新 global_compressed_history 用于下一轮增量计算
            final_input = x[:, input_start:e]
            final_out = model(final_input, past_key_values=global_compressed_history, use_cache=True)
            current_block_kv_raw = slice_new_kv(final_out.past_key_values, past_length)
            global_compressed_history = append_compressed_history(global_compressed_history, current_block_kv_raw)

        if DEBUG:
            block_time = time.time() - block_t0
            block_nfe = nfe - block_nfe_start
            tokens_per_step = total_accepted / block_nfe if block_nfe > 0 else 0
            print(f"Block {nb}: NFE={block_nfe}, AccRate={tokens_per_step:.2f} toks/step, Time={block_time:.3f}s")

    print(f"\nTotal NFE: {nfe}")
    return x, nfe

# ===================== 6. Prefix Cache (Incremental) =====================

@torch.no_grad()
def generate_with_prefix_cache(
    model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, factor=None, 
    use_spectral_compression=False
):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = gen_length // block_length
    steps = steps // num_blocks # 注意：这里的 steps 是总 steps，除以 block 数得到每个 block 的 steps
    nfe = 0
    
    # === [关键变量] 全局已压缩的历史 KV ===
    # 初始为 None，随着 block 进行不断拼接变长
    global_compressed_history = None 

    if DEBUG:
        print(f"\n[Prefix Cache Mode - Incremental] Matrix ASKV: {use_spectral_compression}")
        strategy = f"Fixed Ratio: {FIXED_RETENTION_RATIO}" if FIXED_RETENTION_RATIO is not None else "Adaptive"
        print(f"  Compression Strategy: {strategy}")
        print(f"  Keep Compressed Length (No Padding): {KEEP_COMPRESSED_LENGTH}")
            
    for num_block in range(num_blocks):
        block_t0 = time.time()
        block_nfe_start = nfe

        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        
        # 计算当前 Block 的输入范围
        if num_block == 0:
            # Block 0: 输入包括 Prompt + 当前 Block 的 Mask
            input_start = 0
            current_input_ids = x[:, input_start:current_block_end]
            past_length = 0
        else:
            # Block N: 输入仅为当前 Block 的 Mask (历史由 past_key_values 提供)
            input_start = current_block_start
            current_input_ids = x[:, input_start:current_block_end]
            past_length = global_compressed_history[0][0].shape[2]

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # === Step 0: 增量计算初始 Logits ===
        output = model(
            current_input_ids, 
            past_key_values=global_compressed_history, 
            use_cache=True
        )
        nfe += 1

        # 处理 Logits 切片
        if num_block == 0:
            current_logits = output.logits[:, prompt.shape[1]:, :]
        else:
            current_logits = output.logits

        # 初始 Mask 更新逻辑
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        
        if factor is None:
            x0, transfer_index = get_transfer_index(current_logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(current_logits, temperature, remasking, mask_index, x, None, factor)
        
        x[transfer_index] = x0[transfer_index]
        total_accepted = transfer_index.sum().item()
        
        # === Step 1 ~ N: Block 内部循环 ===
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0: break
            
            # 增量推理：只输入当前 block
            step_input = x[:, input_start:current_block_end]
            
            logits_out = model(
                step_input, 
                past_key_values=global_compressed_history, 
                use_cache=True
            )
            nfe += 1
            
            if num_block == 0:
                 logits = logits_out.logits[:, prompt.shape[1]:, :]
            else:
                 logits = logits_out.logits

            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            # 采样逻辑
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x[:, current_block_start:], None, factor)
            
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            total_accepted += transfer_index.sum().item()
            i += 1

        # === Block 结束: 压缩并归档 ===
        final_input = x[:, input_start:current_block_end]
        final_out = model(final_input, past_key_values=global_compressed_history, use_cache=True)
        current_block_kv_raw = slice_new_kv(final_out.past_key_values, past_length)
        
        if use_spectral_compression:
            current_retention = FIXED_RETENTION_RATIO if FIXED_RETENTION_RATIO is not None else 0.5
            compressed_block_kv = []
            for layer_kv in current_block_kv_raw:
                k_raw, v_raw = layer_kv
                # 压缩 K & V
                c_k = dct_compress_matrix(k_raw, current_retention)
                k_comp = idct_decompress_matrix(c_k, model.device, k_raw.dtype, keep_compressed_length=KEEP_COMPRESSED_LENGTH)
                c_v = dct_compress_matrix(v_raw, current_retention)
                v_comp = idct_decompress_matrix(c_v, model.device, v_raw.dtype, keep_compressed_length=KEEP_COMPRESSED_LENGTH)
                compressed_block_kv.append((k_comp, v_comp))
            
            current_block_kv_final = tuple(compressed_block_kv)
            if DEBUG:
                print(f"  [Archive] Prefix Block {num_block} compressed.")
        else:
            current_block_kv_final = current_block_kv_raw

        # 拼接到全局历史
        global_compressed_history = append_compressed_history(global_compressed_history, current_block_kv_final)

        if DEBUG:
            block_time = time.time() - block_t0
            block_nfe = nfe - block_nfe_start
            tokens_per_step = total_accepted / block_nfe if block_nfe > 0 else 0
            print(f"Block {num_block}: NFE={block_nfe}, AccRate={tokens_per_step:.2f} toks/step, Time={block_time:.3f}s")

    print(f"\nTotal NFE: {nfe}")
    return x, nfe

# ===================== 7. Main =====================

def main():
    parser = argparse.ArgumentParser(description="Fast-dLLM + Incremental ASKV (Matrix Version) Debug")
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
            args.temperature, args.threshold, args.factor, args.use_spectral_compression
        )
    elif args.use_cache:
        generate_with_prefix_cache(
            model, tokenizer, input_ids, args.steps, args.gen_length, args.block_length,
            args.temperature, args.threshold, args.factor, args.use_spectral_compression
        )
    else:
        generate(
            model, input_ids, args.steps, args.gen_length, args.block_length,
            args.temperature, args.threshold, args.factor
        )

if __name__ == "__main__":
    main()