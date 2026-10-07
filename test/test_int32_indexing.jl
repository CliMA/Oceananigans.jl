# 32-bit index arithmetic of GPU kernel arguments (OceananigansCUDAExt: `convert_to_device` gives every OffsetArray
# argument an `Int32IndexedArray` parent). Run on a CUDA GPU:  julia --project=<env> test/test_int32_indexing.jl
using Test
using CUDA
using Oceananigans
using Oceananigans.Architectures: convert_to_device, INT32_INDEXING
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using Atomix: Atomix
using OffsetArrays: OffsetArray
using Adapt: Adapt

const Ext = Base.get_extension(Oceananigans, :OceananigansCUDAExt)

@kernel function _stencil!(r, c)
    i, j, k = @index(Global, NTuple)
    @inbounds r[i, j, k] = (c[i+1, j, k] - c[i-1, j, k]) * c[i, j+1, k] + c[i, j, k-1] / (1 + c[i, j, k+1]^2)
end

# Atomics take the raw (non-offset) device array of a field's parent, as Breeze's Kessler sedimentation bound does
# (AtomixCUDAExt supports `CuDeviceArray` references): such arguments are not OffsetArrays and pass unchanged.
@kernel function _count!(n, c)
    i, j, k = @index(Global, NTuple)
    @inbounds Atomix.@atomic n[1] += ifelse(c[i, j, k] > 0, Int32(1), Int32(0))
end

@testset "Int32-indexed kernel arguments" begin
    @test INT32_INDEXING
    grid = RectilinearGrid(GPU(), Float32; size = (17, 13, 7), halo = (3, 3, 3), extent = (1, 1, 1))
    c = CenterField(grid)
    set!(c, (x, y, z) -> sin(7x) * cos(5y) + z)
    Oceananigans.BoundaryConditions.fill_halo_regions!(c)

    # the converted argument: an OffsetArray over an Int32IndexedArray with the same pointer, size and offsets
    d = convert_to_device(GPU(), c)
    d64 = CUDA.cudaconvert(c)
    @test d isa OffsetArray
    @test parent(d) isa Ext.Int32IndexedArray{Float32, 3}
    @test pointer(parent(d)) == pointer(parent(d64))
    @test size(parent(d)) == size(parent(d64))
    @test d.offsets == d64.offsets
    @test convert_to_device(GPU(), (c, 1, nothing))[1] isa OffsetArray{Float32, 3, <:Ext.Int32IndexedArray}

    # the converted form is a device form: `cudaconvert` returns it unchanged (same type, same bits), so a second
    # `convert_to_device` passes it through (the cached launcher relies on both)
    @test isbits(d)
    @test Core.Compiler.return_type(CUDA.cudaconvert, Tuple{typeof(d)}) === typeof(d)
    @test CUDA.cudaconvert(d) === d
    @test Ext.is_device_form_type(typeof(d))
    @test convert_to_device(GPU(), d) === d
    @test convert_to_device(GPU(), (d, grid))[1] === d
    # a host grid converts (an isbits regular RectilinearGrid included) and its converted form is stable
    g = convert_to_device(GPU(), grid)
    @test typeof(g) === typeof(CUDA.cudaconvert(g))
    @test convert_to_device(GPU(), g) === g

    # a stencil kernel launched through `launch!` (Int32 arguments) equals the same kernel on the 64-bit arguments
    r32 = CenterField(grid)
    launch!(GPU(), grid, :xyz, _stencil!, r32, c)
    r64 = CenterField(grid)
    k64 = _stencil!(CUDA.CUDABackend(), (16, 16))
    k64(r64.data, c.data; ndrange = size(grid))        # KA's own conversion: CuDeviceArray parents (Int64)
    @test isequal(Array(interior(r32)), Array(interior(r64)))

    # atomics on a raw device array next to an Int32-indexed field argument
    n = CUDA.zeros(Int32, 1)
    @test convert_to_device(GPU(), (n,))[1] isa CuDeviceArray
    launch!(GPU(), grid, :xyz, _count!, n, c)
    @test Array(n)[1] == count(>(0), Array(interior(c)))

    # the size guard: an argument array of typemax(Int32) or more elements raises an ArgumentError (a descriptor
    # only, no allocation)
    huge = CuDeviceArray{Float32, 3, 1}(reinterpret(Core.LLVMPtr{Float32, 1}, C_NULL), (2048, 2048, 512))
    @test length(huge) == 2^31
    @test_throws ArgumentError Adapt.adapt(Ext.Int32Indexing(), OffsetArray(huge, -1, -1, -1))
    fits = CuDeviceArray{Float32, 3, 1}(reinterpret(Core.LLVMPtr{Float32, 1}, C_NULL), (2048, 2048, 511))
    @test parent(Adapt.adapt(Ext.Int32Indexing(), OffsetArray(fits, -1, -1, -1))) isa Ext.Int32IndexedArray
    # the check does not allocate
    small = OffsetArray(fits, -1, -1, -1)
    Adapt.adapt(Ext.Int32Indexing(), small)
    @test (@allocated Adapt.adapt(Ext.Int32Indexing(), small)) == 0
end
