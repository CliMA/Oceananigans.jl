# Static-shape specialisations of the cached launch (OceananigansCUDAExt: a kernel that opts in with
# `Utils.kernel_launch_static_shapes` runs a kernel compiled for argument types that carry the dimensions and offsets
# of its `OffsetArray` arguments; same body, same memory, same results). Run on a CUDA GPU:
#   julia --project=<env> test/test_static_shapes.jl
using Test
using CUDA
using Oceananigans
using Oceananigans.Architectures: convert_to_device, architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: Δxᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.Utils: Utils, launch!, KernelParameters, gpu_kernel_function
using KernelAbstractions: @kernel, @index

const Ext = Base.get_extension(Oceananigans, :OceananigansCUDAExt)
const OffsetArray = Ext.OffsetArray

# One body that reads fields of four shapes (cell centres, the three face locations on a bounded grid, a
# two-dimensional field) and grid metrics, in two kernels: one launched as ever, one that opts in.
@inline stencil_value(i, j, k, grid, f) =
    @inbounds (f.u[i+1, j, k] - f.u[i, j, k]) * f.v[i, j+1, k] + f.w[i, j, k+1] * f.c[i, j, k-1] +
              f.η[i, j, 1] * Δxᶜᶜᶜ(i, j, k, grid) + f.c[i-1, j+1, k] / (1 + Δzᶜᶜᶜ(i, j, k, grid))

@kernel function _stencil_ordinary!(a, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = stencil_value(i, j, k, grid, f)
end

@kernel function _stencil_static!(a, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = stencil_value(i, j, k, grid, f)
end

Utils.kernel_launch_static_shapes(::typeof(gpu_kernel_function(_stencil_static!)), ::Type) = true

# A generic kernel that opts in for one of the functions it is given.
@kernel function _apply!(a, grid, operation, f)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = operation(i, j, k, grid, f)
end

@inline other_value(i, j, k, grid, f) = stencil_value(i, j, k, grid, f)

Utils.kernel_launch_static_shapes(::typeof(gpu_kernel_function(_apply!)),
                                  ::Type{<:Tuple{Any, Any, Any, typeof(stencil_value), Vararg}}) = true

# A kernel launched on output arrays of several shapes under one argument type.
@kernel function _shifted_copy!(a, grid, b)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = b[i, j, k] + b[i, j, k+1]
end

Utils.kernel_launch_static_shapes(::typeof(gpu_kernel_function(_shifted_copy!)), ::Type) = true

# Two fields of one type parameter, and a parameter that also appears inside a composite field type.
struct TwoOfOneType{A}
    first :: A
    second :: A
end

struct ParameterInComposite{A}
    pair :: Tuple{A, A}
    single :: A
end

function test_fields(grid)
    fields = (u = XFaceField(grid), v = YFaceField(grid), w = ZFaceField(grid), c = CenterField(grid),
              η = Field{Center, Center, Nothing}(grid))
    set!(fields.u, (x, y, z) -> sin(7x) + z)
    set!(fields.v, (x, y, z) -> cos(5y) - x * z)
    set!(fields.w, (x, y, z) -> x * y + sin(3z))
    set!(fields.c, (x, y, z) -> 1 + cos(x + 2y + 3z))
    set!(fields.η, (x, y) -> 0.5 + x - y)
    foreach(fill_halo_regions!, fields)
    return fields
end

function launched(kernel!, grid, fields, workspec, arguments...)
    a = CenterField(grid)
    launch!(architecture(grid), grid, workspec, kernel!, a, grid, arguments..., fields)
    return Array(interior(a))
end

same_bits(a, b) = reinterpret(UInt64, a) == reinterpret(UInt64, b)

@testset "Static-shape specialisations of the cached launch" begin
    arch = GPU()
    stretched = RectilinearGrid(arch; size = (8, 6, 4), halo = (3, 3, 3), x = (0, 1), y = (0, 1), z = [0, 0.1, 0.3, 0.6, 1],
                                topology = (Bounded, Bounded, Bounded))
    periodic = RectilinearGrid(arch; size = (8, 6, 4), halo = (3, 3, 3), x = (0, 1), y = (0, 1), z = [0, 0.1, 0.3, 0.6, 1],
                               topology = (Periodic, Periodic, Bounded))
    latlon = LatitudeLongitudeGrid(arch; size = (8, 6, 4), halo = (3, 3, 3), longitude = (0, 10), latitude = (0, 10), z = (0, 1))

    @test Utils.kernel_launch_static_shapes(gpu__stencil_ordinary!, Tuple{Int}) === false      # the fallback
    @test Utils.kernel_launch_static_shapes(gpu__stencil_static!, Tuple{Int}) === true

    @testset "The static-shape type of an argument" begin
        @test Ext.static_linear_index((5, 4, 3), (-1, -2, 0), (0, -1, 1)) == 0
        @test Ext.static_linear_index((5, 4, 3), (-1, -2, 0), (2, 1, 3)) == LinearIndices((5, 4, 3))[3, 3, 3] - 1
        @test Ext.static_linear_index((7,), (-3,), (2,)) == 4

        fields = test_fields(stretched)
        device = convert_to_device(arch, map(f -> f.data, fields))
        T = typeof(device)
        @test Ext.argument_shapes(1.0) === () && Ext.argument_shapes((1, Val(2), nothing)) === ()
        shapes = Ext.argument_shapes(device)
        @test shapes === (parent(device.u).dims, device.u.offsets, parent(device.v).dims, device.v.offsets,
                          parent(device.w).dims, device.w.offsets, parent(device.c).dims, device.c.offsets,
                          parent(device.η).dims, device.η.offsets)

        S = Ext.static_shape_type(T, device)
        @test S !== T && Ext.same_layout(S, T) && isbitstype(S)
        static = Ext.reinterpret_arguments(S, device)
        for name in keys(fields)
            d, s = device[name], static[name]
            @test s isa OffsetArray{Float64, 3, <:Ext.StaticShapeArray}
            @test typeof(parent(s)).parameters[4] === size(parent(d)) && typeof(parent(s)).parameters[5] === d.offsets
            @test parent(s).ptr === parent(d).ptr && parent(s).dims === parent(d).dims && s.offsets === d.offsets
            @test size(s) == size(d) && axes(s) == axes(d)
        end
        @test typeof(static.u) !== typeof(static.c)                                   # two shapes, two types
        @test Ext.static_shape_type(S, static) === S                                  # nothing left to do
        @test Ext.static_shape_type(Tuple{Float64, Nothing}, (1.0, nothing)) === Tuple{Float64, Nothing}

        # one parameter for two arrays: replaced when they agree, kept when they do not
        @test Ext.static_shape_type(typeof(TwoOfOneType(device.c, device.c)), TwoOfOneType(device.c, device.c)) ===
              TwoOfOneType{typeof(static.c)}
        mixed = TwoOfOneType{typeof(device.c)}(device.c, device.u)
        @test Ext.static_shape_type(typeof(mixed), mixed) === typeof(mixed)
        # … and kept when the parameter also appears inside a composite field type
        composite = ParameterInComposite((device.c, device.c), device.c)
        @test Ext.static_shape_type(typeof(composite), composite) === typeof(composite)
        # a grid: its metric arrays, where each has its own parameter
        device_grid = convert_to_device(arch, stretched)
        G = Ext.static_shape_type(typeof(device_grid), device_grid)
        @test Ext.same_layout(G, typeof(device_grid))
        @test Ext.reinterpret_arguments(G, device_grid).Nx == stretched.Nx
    end

    @testset "The register budget of a block count" begin
        registers = Ext.occupancy.attribute(CUDA.device(), Ext.occupancy.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
        for blocks in 2:8
            budget = Ext.block_register_budget(256, blocks)
            @test budget % 4 == 0 && budget * 256 * blocks <= registers < (budget + 4) * 256 * blocks
        end
    end

    cached_before, check_before, static_before = Ext.CACHED_KERNEL_LAUNCH[], Ext.CHECK_CACHED_KERNEL_LAUNCH[], Ext.STATIC_SHAPES[]
    try
        Ext.CACHED_KERNEL_LAUNCH[] = true
        Ext.CHECK_CACHED_KERNEL_LAUNCH[] = true     # every launch re-derives the static type and re-converts the arguments
        Ext.STATIC_SHAPES[] = true

        @testset "Launches on $(nameof(typeof(grid))) $(Oceananigans.Grids.topology(grid))" for grid in (stretched, periodic, latlon)
            fields = test_fields(grid)
            for workspec in (:xyz, KernelParameters(2:7, 2:5, 2:3))
                fill!(Ext.static_shape_counts, 0)
                ordinary = launched(_stencil_ordinary!, grid, fields, workspec)
                @test Ext.static_shape_counts[1] == 0                                  # not opted in
                static = launched(_stencil_static!, grid, fields, workspec)
                @test Ext.static_shape_counts[1] == 1 && Ext.static_shape_counts[2] == 0
                @test any(!iszero, static) && same_bits(static, ordinary)
                again = launched(_stencil_static!, grid, fields, workspec)             # the entry's shapes, no compilation
                @test Ext.static_shape_counts[1] == 2 && same_bits(again, ordinary)

                # a generic kernel opts in by the function it is given
                fill!(Ext.static_shape_counts, 0)
                by_function = launched(_apply!, grid, fields, workspec, stencil_value)
                @test Ext.static_shape_counts[1] == 1 && same_bits(by_function, ordinary)
                not_opted = launched(_apply!, grid, fields, workspec, other_value)
                @test Ext.static_shape_counts[1] == 1 && same_bits(not_opted, ordinary)
            end
            @test isempty(Ext.static_shape_refusals)
        end

        @testset "One argument type, arrays of several shapes" begin
            grid = stretched
            b = ZFaceField(grid); set!(b, (x, y, z) -> x - 2y + z^2); fill_halo_regions!(b)
            outputs = (CenterField(grid), XFaceField(grid), YFaceField(grid))        # one device type, three shapes
            @test allequal(map(a -> typeof(convert_to_device(arch, a.data)), outputs))
            results = Dict{Bool, Any}()
            for static in (false, true)
                Ext.STATIC_SHAPES[] = static
                fill!(Ext.static_shape_counts, 0)
                results[static] = map(outputs) do a
                    fill!(a, 0)
                    for round in 1:2                                                 # alternate between the shapes twice
                        launch!(arch, grid, :xyz, _shifted_copy!, a, grid, b)
                    end
                    Array(interior(a))
                end
                @test Ext.static_shape_counts[1] == (static ? 6 : 0)
            end
            @test all(same_bits(results[true][n], results[false][n]) && any(!iszero, results[true][n]) for n in 1:3)
            entries = [entry for (key, entry) in Ext.static_shape_entries if key[2] === gpu__shifted_copy!]
            @test length(entries) == 1 && length(only(entries).others) == 2       # three shapes: the current one and two others
            @test all(other -> last(other) isa Ext.StaticShapeLaunch, only(entries).others)
        end

        @testset "Switched off" begin
            Ext.STATIC_SHAPES[] = false
            fields = test_fields(latlon)
            fill!(Ext.static_shape_counts, 0)
            off = launched(_stencil_static!, latlon, fields, :xyz)
            @test Ext.static_shape_counts[1] == 0
            @test same_bits(off, launched(_stencil_ordinary!, latlon, fields, :xyz))
        end
    finally
        Ext.CACHED_KERNEL_LAUNCH[], Ext.CHECK_CACHED_KERNEL_LAUNCH[], Ext.STATIC_SHAPES[] = cached_before, check_before, static_before
    end
end
