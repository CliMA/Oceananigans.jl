using AMDGPU

using Oceananigans
using Oceananigans.Architectures: architecture, device

@testset "AMDGPU Unit Tests" begin
    arch = GPU(AMDGPU.ROCBackend())

    @test GPU(AMDGPU.ROCBackend()) isa GPU
    @test device(arch) == AMDGPU.ROCBackend()
end
