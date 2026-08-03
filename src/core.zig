const std = @import("std");

/// Small, dependency-free hot-path utilities exposed through a C ABI.
/// Python uses this for validation, sizing, progress calculations and
/// VRAM estimation while PyTorch owns GPU kernels and autograd.

pub const ZllmConfig = extern struct {
    seq_len: u32,
    hidden_size: u32,
    layers: u32,
    batch_size: u32,
    grad_accum: u32,
    vocab_size: u32,
};

fn isPow2(n: u32) bool { return n != 0 and (n & (n - 1)) == 0; }

export fn zigllm_version() callconv(.c) [*:0]const u8 { return "0.2.0"; }

/// Returns 0 when valid; otherwise a stable error code.
/// 1=zero value, 2=sequence not power of two, 3=hidden not divisible by 8,
/// 4=gradient accumulation is zero, 5=unreasonable vocabulary.
/// Uses branchless OR-folding for the zero-check fast path.
export fn zigllm_validate_config(c: ZllmConfig) callconv(.c) u32 {
    const any_zero: u32 = @intFromBool(c.seq_len == 0) | @intFromBool(c.hidden_size == 0) | @intFromBool(c.layers == 0) | @intFromBool(c.batch_size == 0);
    if (any_zero != 0) return 1;
    if (!isPow2(c.seq_len)) return 2;
    if (c.hidden_size % 8 != 0) return 3;
    if (c.grad_accum == 0) return 4;
    if (c.vocab_size < 256) return 5;
    return 0;
}

/// Approximate tokens per optimizer step. Saturates rather than overflowing.
export fn zigllm_tokens_per_step(c: ZllmConfig) callconv(.c) u64 {
    const a: u64 = c.seq_len;
    const b: u64 = c.batch_size;
    const d: u64 = c.grad_accum;
    if (a == 0 or b == 0 or d == 0) return 0;
    var result: u64 = a;
    var overflow: u1 = 0;
    const r1 = @mulWithOverflow(result, b);
    result = r1[0]; overflow = r1[1];
    if (overflow != 0) return std.math.maxInt(u64);
    const r2 = @mulWithOverflow(result, d);
    result = r2[0]; overflow = r2[1];
    if (overflow != 0) return std.math.maxInt(u64);
    return result;
}

/// Estimate VRAM (in bytes) needed for a given configuration.
/// Model: 2 bytes/param (fp16) × params.  params ≈ 12 × L × H²  (transformer approx).
/// Optimizer (AdamW fp32): ~16 bytes/param for trainable weights.
/// Activations + gradients scale with seq_len × batch_size × hidden_size × layers.
/// This is a fast planning estimate, not a precise profiler.
export fn zigllm_estimate_vram_mb(c: ZllmConfig, adapter_is_lora: bool) callconv(.c) u64 {
    if (c.hidden_size == 0 or c.layers == 0 or c.seq_len == 0 or c.batch_size == 0) return 0;
    const h: u64 = c.hidden_size;
    const l: u64 = c.layers;
    const s: u64 = c.seq_len;
    const b: u64 = c.batch_size;
    // Transformer parameter count approximation: 12 * L * H^2
    const params: u64 = 12 *% l *% h *% h;
    // Model weights in fp16: 2 bytes per param
    const model_bytes: u64 = params * 2;
    // Optimizer state: 16 bytes/param for full fine-tune, ~2 bytes/param for LoRA
    const opt_bytes: u64 = if (adapter_is_lora) params * 2 else params * 16;
    // Activation memory (rough): seq_len * batch * hidden * layers * 4 bytes (fp32 intermediates)
    const act_bytes: u64 = s * b * h * l * 4;
    const total = model_bytes + opt_bytes + act_bytes;
    return total / (1024 * 1024); // Convert to MB
}

/// Estimate total training steps given dataset size, config, and epochs.
export fn zigllm_estimate_steps(ds_size: u64, c: ZllmConfig, epochs: u64) callconv(.c) u64 {
    const effective_batch: u64 = @as(u64, c.batch_size) * @as(u64, c.grad_accum);
    if (effective_batch == 0 or ds_size == 0) return 0;
    const steps_per_epoch: u64 = (ds_size + effective_batch - 1) / effective_batch; // ceil div
    // Saturating multiply for epochs
    var overflow: u1 = 0;
    const r = @mulWithOverflow(steps_per_epoch, epochs);
    if (r[1] != 0) return std.math.maxInt(u64);
    return r[0];
}

test "validation and tokens" {
    const c = ZllmConfig{ .seq_len = 1024, .hidden_size = 768, .layers = 12, .batch_size = 2, .grad_accum = 8, .vocab_size = 32000 };
    try std.testing.expectEqual(@as(u32, 0), zigllm_validate_config(c));
    try std.testing.expectEqual(@as(u64, 16384), zigllm_tokens_per_step(c));
}

test "vram estimation sanity" {
    const c = ZllmConfig{ .seq_len = 2048, .hidden_size = 4096, .layers = 32, .batch_size = 1, .grad_accum = 8, .vocab_size = 32000 };
    const full = zigllm_estimate_vram_mb(c, false);
    const lora = zigllm_estimate_vram_mb(c, true);
    // Full fine-tune should need more VRAM than LoRA
    try std.testing.expect(full > lora);
    // Both should be positive and reasonable (>100 MB for this config)
    try std.testing.expect(full > 100);
    try std.testing.expect(lora > 100);
}

test "step estimation" {
    const c = ZllmConfig{ .seq_len = 1024, .hidden_size = 768, .layers = 12, .batch_size = 2, .grad_accum = 8, .vocab_size = 32000 };
    // 1000 samples, batch=2, accum=8 → effective_batch=16 → ceil(1000/16)=63 steps/epoch
    const steps = zigllm_estimate_steps(1000, c, 1);
    try std.testing.expectEqual(@as(u64, 63), steps);
    // 3 epochs → 189
    const steps3 = zigllm_estimate_steps(1000, c, 3);
    try std.testing.expectEqual(@as(u64, 189), steps3);
}
