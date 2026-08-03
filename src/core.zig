const std = @import("std");

/// Small, dependency-free hot-path utilities exposed through a C ABI.
/// Python uses this for validation, sizing and progress calculations while
/// PyTorch owns GPU kernels and autograd.

pub const ZllmConfig = extern struct {
    seq_len: u32,
    hidden_size: u32,
    layers: u32,
    batch_size: u32,
    grad_accum: u32,
    vocab_size: u32,
};

fn isPow2(n: u32) bool { return n != 0 and (n & (n - 1)) == 0; }

export fn zigllm_version() callconv(.c) [*:0]const u8 { return "0.1.0"; }

/// Returns 0 when valid; otherwise a stable error code.
/// 1=zero value, 2=sequence not power of two, 3=hidden not divisible by 8,
/// 4=gradient accumulation is zero, 5=unreasonable vocabulary.
export fn zigllm_validate_config(c: ZllmConfig) callconv(.c) u32 {
    if (c.seq_len == 0 or c.hidden_size == 0 or c.layers == 0 or c.batch_size == 0) return 1;
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
    const max = std.math.maxInt(u64);
    if (a > max / b or a * b > max / d) return max;
    return a * b * d;
}

test "validation and tokens" {
    const c = ZllmConfig{ .seq_len = 1024, .hidden_size = 768, .layers = 12, .batch_size = 2, .grad_accum = 8, .vocab_size = 32000 };
    try std.testing.expectEqual(@as(u32, 0), zigllm_validate_config(c));
    try std.testing.expectEqual(@as(u64, 16384), zigllm_tokens_per_step(c));
}
