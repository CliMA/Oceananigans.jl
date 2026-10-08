# Device-form kernel arguments and the cached launch of configured kernels (OceananigansCUDAExt:
# `convert_to_device` passes arguments already in their device form through, `launch_configured_kernel!` launches a
# cached compiled kernel without re-converting them). Run on a CUDA GPU:
#   julia --project=<env> test/test_cached_kernel_launch.jl
using Test
using CUDA
using Oceananigans
using Oceananigans.Architectures: convert_to_device, architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: Utils, launch!, launch_configured_kernel!, KernelParameters, gpu_kernel_function
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!, solve_batched_tridiagonal_system_kernel!
using KernelAbstractions: @kernel, @index

const Ext = Base.get_extension(Oceananigans, :OceananigansCUDAExt)

@kernel function _copy_on_grid!(a, grid, b)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = b[i, j, k] + zero(grid)
end

# Two kernels with the same body: one launched as configured, one with a per-kernel launch configuration
# (another workgroup and a register cap).
@kernel function _stencil_default!(a, grid, b)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = (b[i+1, j, k] - b[i-1, j, k]) * b[i, j+1, k] + b[i, j, k-1] / (1 + b[i, j, k+1]^2)
end

@kernel function _stencil_configured!(a, grid, b)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = (b[i+1, j, k] - b[i-1, j, k]) * b[i, j+1, k] + b[i, j, k-1] / (1 + b[i, j, k+1]^2)
end

Utils.kernel_launch_workgroup(::typeof(gpu_kernel_function(_stencil_configured!)), ::Val{W}) where W = ntuple(d -> d == 1 ? 4 : 2, length(W))
Utils.kernel_launch_maxregs(::typeof(gpu_kernel_function(_stencil_configured!))) = 32

function stencil(kernel!, grid, workspec = :xyz)
    a = CenterField(grid)
    b = CenterField(grid)
    set!(b, (x, y, z) -> sin(3x) + cos(5y) * z)
    fill_halo_regions!(b)
    launch!(architecture(grid), grid, workspec, kernel!, a, grid, b)
    return Array(interior(a))
end

# A vertical tridiagonal solve of a diagonally dominant system on `grid`, cached launch on or off.
function tridiagonal_solution(grid, cached)
    Ext.CACHED_KERNEL_LAUNCH[] = cached
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    lower = on_architecture(arch, [sin(i + 2j + 3k) for i in 1:Nx, j in 1:Ny, k in 1:Nz])
    upper = on_architecture(arch, [cos(2i + j + 5k) for i in 1:Nx, j in 1:Ny, k in 1:Nz])
    diagonal = on_architecture(arch, [4 + sin(i * j + k) for i in 1:Nx, j in 1:Ny, k in 1:Nz])
    rhs = on_architecture(arch, [cos(i - j) + k / Nz for i in 1:Nx, j in 1:Ny, k in 1:Nz])
    solver = BatchedTridiagonalSolver(grid; lower_diagonal = lower, diagonal, upper_diagonal = upper)
    ϕ = CenterField(grid)
    solve!(ϕ, solver, rhs)
    return Array(interior(ϕ))
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
        # Per-kernel launch configuration: another workgroup and a register cap change the launch, not the result.
        Ext.CACHED_KERNEL_LAUNCH[] = true
        fill!(Ext.cached_launch_counts, 0)
        for grid in (regular, stretched, latlon)
            default = stencil(_stencil_default!, grid)
            configured = stencil(_stencil_configured!, grid)
            @test reinterpret(UInt64, configured) == reinterpret(UInt64, default)
        end
        # … also over an offset range (the ndrange keeps its type, the workgroup changes)
        for grid in (regular, stretched, latlon)
            range = KernelParameters(2:7, 2:5, 2:3)
            default = stencil(_stencil_default!, grid, range)
            configured = stencil(_stencil_configured!, grid, range)
            @test reinterpret(UInt64, configured) == reinterpret(UInt64, default)
            @test any(!iszero, configured) && iszero(configured[1, :, :]) && iszero(configured[:, :, 4])
        end
        @test Ext.cached_launch_counts[2] == 0
        @test gpu_kernel_function(_stencil_default!) === gpu__stencil_default!
        keys_of(f) = [key for key in keys(Ext.cached_launch_kernels) if key[2] === f]
        @test all(key -> key[4] == 8 && key[6] == 32, keys_of(gpu__stencil_configured!))     # maxthreads 4 × 2, maxregs 32
        @test length(keys_of(gpu__stencil_configured!)) >= 2                                 # :xyz and the offset range
        @test all(key -> key[4] == 256 && key[6] == 0, keys_of(gpu__stencil_default!))
        @test all(kernel -> CUDA.registers(kernel) <= 32, [Ext.cached_launch_kernels[key] for key in keys_of(gpu__stencil_configured!)])

        # The batched tridiagonal solve's default workgroup entry (32 × 8): cached launch (entry applied) against
        # the standard launch (16 × 16), several blocks in each direction.
        wide = RectilinearGrid(arch; size = (70, 40, 12), halo = (3, 3, 3), x = (0, 1), y = (0, 1), z = (0, 1),
                               topology = (Bounded, Bounded, Bounded))
        tridiagonal = gpu_kernel_function(solve_batched_tridiagonal_system_kernel!)
        @test Utils.kernel_launch_workgroup(tridiagonal, Val((16, 16))) === (32, 8)
        @test Utils.kernel_launch_workgroup(tridiagonal, Val((1, 256))) === (1, 256)
        fill!(Ext.cached_launch_counts, 0)
        entry_applied = tridiagonal_solution(wide, true)
        @test Ext.cached_launch_counts[2] == 0
        standard = tridiagonal_solution(wide, false)
        @test reinterpret(UInt64, entry_applied) == reinterpret(UInt64, standard)
        @test all(isfinite, entry_applied) && any(!iszero, entry_applied)
        @test !isempty(keys_of(tridiagonal))
        @test all(key -> occursin("StaticSize{(32, 8)}", string(key[3])), keys_of(tridiagonal))

        # A launch with an argument that is not in its device form takes the standard launch, and is recorded.
        Ext.CACHED_KERNEL_LAUNCH[] = true
        Ext.RECORD_STANDARD_LAUNCHES[] = true
        empty!(Ext.standard_launches)
        a, b = CenterField(regular), CenterField(regular)
        set!(b, (x, y, z) -> x - y + z)
        loop! = _copy_on_grid!(CUDA.CUDABackend(), (16, 16), size(regular))
        launch_configured_kernel!(loop!, a.data, regular, b.data)
        @test Array(interior(a)) == Array(interior(b))
        @test Ext.standard_launches == Dict((gpu_kernel_function(_copy_on_grid!), [2, 3, 4]) => 1)
        dynamic_loop! = _copy_on_grid!(CUDA.CUDABackend(), (16, 16))
        @test_throws Exception launch_configured_kernel!(dynamic_loop!, convert_to_device(arch, (a.data, regular, b.data))...)
        @test Ext.standard_launches[(gpu_kernel_function(_copy_on_grid!), :dynamic)] == 1
        Ext.RECORD_STANDARD_LAUNCHES[] = false
    finally
        Ext.CACHED_KERNEL_LAUNCH[], Ext.CHECK_CACHED_KERNEL_LAUNCH[] = cached_before, check_before
    end
end
