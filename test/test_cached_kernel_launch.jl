# Device-form kernel arguments and the cached launch of configured kernels (OceananigansCUDAExt:
# `convert_to_device` passes arguments already in their device form through, `launch_configured_kernel!` launches a
# cached compiled kernel without re-converting them). Run on a CUDA GPU:
#   julia --project=<env> test/test_cached_kernel_launch.jl
using Test
using CUDA
using Oceananigans
using Oceananigans.Architectures: convert_to_device, architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

const Ext = Base.get_extension(Oceananigans, :OceananigansCUDAExt)

@kernel function _copy_on_grid!(a, grid, b)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = b[i, j, k] + zero(grid)
end

# The result of one launch and one halo fill, cached launch on or off.
function launched_and_filled(grid, cached)
    Ext.CACHED_KERNEL_LAUNCH[] = cached
    a = CenterField(grid)
    b = CenterField(grid)
    set!(b, (x, y, z) -> x + 2y + 3z)
    launch!(architecture(grid), grid, :xyz, _copy_on_grid!, a, grid, b)
    fill_halo_regions!(a)
    return Array(parent(a))
end

@testset "Device-form kernel arguments and the cached launch" begin
    arch = GPU()

    # A regular `RectilinearGrid` on the GPU is `isbits` on the host and is NOT its device form: `cudaconvert`
    # strips the architecture, another type. It must be converted, and the cached launch must not take it as is.
    regular = RectilinearGrid(arch; size = (8, 6, 4), halo = (3, 3, 3), extent = (1, 1, 1), topology = (Periodic, Periodic, Bounded))
    stretched = RectilinearGrid(arch; size = (8, 6, 4), halo = (3, 3, 3), x = (0, 1), y = (0, 1), z = [0, 0.1, 0.3, 0.6, 1],
                                topology = (Bounded, Bounded, Bounded))
    latlon = LatitudeLongitudeGrid(arch; size = (8, 6, 4), halo = (3, 3, 3), longitude = (0, 10), latitude = (0, 10), z = (0, 1))

    @test isbits(regular)
    @test !Ext.is_device_form_type(typeof(regular))

    for grid in (regular, stretched, latlon)
        device_grid = convert_to_device(arch, grid)
        @test device_grid === Ext.int32_indexed(CUDA.cudaconvert(grid))   # cudaconvert, then 32-bit index parents
        @test typeof(device_grid) != typeof(grid)
        @test Ext.is_device_form_type(typeof(device_grid))
        @test convert_to_device(arch, device_grid) === device_grid       # idempotent, untouched

        c = CenterField(grid)
        @test !Ext.is_device_form_type(typeof(c.data))
        device_c = convert_to_device(arch, c.data)
        @test device_c === Ext.int32_indexed(CUDA.cudaconvert(c.data))
        @test CUDA.cudaconvert(device_c) === device_c                    # the device form is cudaconvert-stable
        @test Ext.is_device_form_type(typeof(device_c))
        @test convert_to_device(arch, device_c) === device_c

        # tuples convert element by element
        converted = convert_to_device(arch, (c.data, grid, 1.0, Val(2)))
        @test converted === (device_c, device_grid, 1.0, Val(2))
        @test Ext.all_device_form(converted)
        @test !Ext.all_device_form((device_c, grid))
    end

    for x in (1.0, 1f0, 3, Val(2), nothing, (1.0, Val(1)), (a = 1, b = 2f0))
        @test Ext.is_device_form_type(typeof(x))
        @test convert_to_device(arch, x) === x
    end

    @test !Ext.is_device_form_type(CuPtr{Float64})   # converts to an LLVMPtr
    @test !Ext.is_device_form_type(DataType)         # a `Type` argument compiles as `Type{T}`
    @test !Ext.is_device_form_type(typeof(CUDA.zeros(2)))

    # Kernels and halo fills on each grid through the cached launch, with the check that the cached launch's
    # arguments are bit-identical to their re-conversion on every launch, against the standard launch.
    cached_before, check_before = Ext.CACHED_KERNEL_LAUNCH[], Ext.CHECK_CACHED_KERNEL_LAUNCH[]
    try
        Ext.CHECK_CACHED_KERNEL_LAUNCH[] = true
        for grid in (regular, stretched, latlon)
            fill!(Ext.cached_launch_counts, 0)
            with_cache = launched_and_filled(grid, true)
            @test Ext.cached_launch_counts[1] > 0      # the launches took the cached path
            @test Ext.cached_launch_counts[2] == 0
            without_cache = launched_and_filled(grid, false)
            @test reinterpret(UInt64, with_cache) == reinterpret(UInt64, without_cache)
        end
    finally
        Ext.CACHED_KERNEL_LAUNCH[], Ext.CHECK_CACHED_KERNEL_LAUNCH[] = cached_before, check_before
    end
end
