const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lib = b.addSharedLibrary(.{
        .name = "zigllm_core",
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.linkLibC();
    b.installArtifact(lib);
    const test_step = b.step("test", "Run core tests");
    const tests = b.addTest(.{ .root_source_file = b.path("src/core.zig"), .target = target, .optimize = optimize });
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
