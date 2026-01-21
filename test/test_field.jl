include("dependencies_for_runtests.jl")

using Statistics

using Oceananigans.Fields: CenterField, ReducedField, has_velocities
using Oceananigans.Fields: VelocityFields, TracerFields, interpolate, interpolate!
using Oceananigans.Fields: reduced_location
using Oceananigans.Fields: FractionalIndices, interpolator, instantiate
using Oceananigans.Fields: convert_to_0_360, convert_to_λ₀_λ₀_plus360
using Oceananigans.Fields: ZeroField, OneField, ConstantField, prognostic_state, restore_prognostic_state!
using Oceananigans.Grids: ξnode, ηnode, rnode
using Oceananigans.Grids: total_length
using Oceananigans.Grids: λnode
using Oceananigans.Grids: RectilinearGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom

using Random
using GPUArraysCore: @allowscalar

"""
    correct_field_size(grid, FieldType, Tx, Ty, Tz)

Test that the field initialized by the FieldType constructor on `grid`
has size `(Tx, Ty, Tz)`.
"""
correct_field_size(grid, loc, Tx, Ty, Tz) = size(parent(Field(instantiate(loc), grid))) == (Tx, Ty, Tz)

function run_similar_field_tests(f)
    g = similar(f)
    @test typeof(f) == typeof(g)
    @test f.grid == g.grid
    @test location(f) === location(g)
    @test !(f.data === g.data)
    return nothing
end

"""
     correct_field_value_was_set(N, L, ftf, val)

Test that the field initialized by the field type function `ftf` on the grid g
can be correctly filled with the value `val` using the `set!(f::AbstractField, v)`
function.
"""
function correct_field_value_was_set(grid, FieldType, val::Number)
    arch = architecture(grid)
    f = FieldType(grid)
    set!(f, val)
    return all(interior(f) .≈ val * on_architecture(arch, ones(size(f))))
end

function run_field_reduction_tests(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    c = CenterField(grid)
    η = Field{Center, Center, Nothing}(grid)

    f(x, y, z) = 1 + exp(x) * sin(y) * tanh(z)

    ϕs = [u, v, w, c]
    [set!(ϕ, f) for ϕ in ϕs]

    z_top = znodes(grid, Face())[end]
    set!(η, (x, y) -> f(x, y, z_top))
    push!(ϕs, η)
    ϕs = Tuple(ϕs)

    u_vals = f.(nodes(u, reshape=true)...)
    v_vals = f.(nodes(v, reshape=true)...)
    w_vals = f.(nodes(w, reshape=true)...)
    c_vals = f.(nodes(c, reshape=true)...)
    η_vals = f.(nodes(η, reshape=true)...)

    # Convert to CuArray if needed.
    arch = architecture(grid)
    u_vals = on_architecture(arch, u_vals)
    v_vals = on_architecture(arch, v_vals)
    w_vals = on_architecture(arch, w_vals)
    c_vals = on_architecture(arch, c_vals)
    η_vals = on_architecture(arch, η_vals)

    ϕs_vals = (u_vals, v_vals, w_vals, c_vals, η_vals)

    dims_to_test = (1, 2, 3, (1, 2), (1, 3), (2, 3), (1, 2, 3))

    for (ϕ, ϕ_vals) in zip(ϕs, ϕs_vals)
        ε = eps(eltype(grid)) * 10 * maximum(maximum.(ϕs_vals))
        @info "      Testing field reductions with tolerance $ε..."

        @test @allowscalar all(isapprox.(ϕ, ϕ_vals, atol=ε)) # if this isn't true, reduction tests can't pass

        # Important to make sure no scalar operations occur on GPU!
        GPUArraysCore.allowscalar(false)

        @test minimum(ϕ) ≈ minimum(ϕ_vals) atol=ε
        @test maximum(ϕ) ≈ maximum(ϕ_vals) atol=ε
        @test mean(ϕ) ≈ mean(ϕ_vals) atol=2ε
        @test minimum(∛, ϕ) ≈ minimum(∛, ϕ_vals) atol=ε
        @test maximum(abs, ϕ) ≈ maximum(abs, ϕ_vals) atol=ε
        @test mean(abs2, ϕ) ≈ mean(abs2, ϕ) atol=ε

        @test extrema(ϕ) == (minimum(ϕ), maximum(ϕ))
        @test extrema(∛, ϕ) == (minimum(∛, ϕ), maximum(∛, ϕ))

        for dims in dims_to_test
            @test all(isapprox(minimum(ϕ, dims=dims), minimum(ϕ_vals, dims=dims), atol=4ε))
            @test all(isapprox(maximum(ϕ, dims=dims), maximum(ϕ_vals, dims=dims), atol=4ε))
            @test all(isapprox(mean(ϕ, dims=dims), mean(ϕ_vals, dims=dims), atol=4ε))

            @test all(isapprox(minimum(sin, ϕ, dims=dims), minimum(sin, ϕ_vals, dims=dims), atol=4ε))
            @test all(isapprox(maximum(cos, ϕ, dims=dims), maximum(cos, ϕ_vals, dims=dims), atol=4ε))
            @test all(isapprox(mean(cosh, ϕ, dims=dims), mean(cosh, ϕ_vals, dims=dims), atol=5ε))
        end
    end

    return nothing
end

# Choose a trilinear function so trilinear interpolation can return values that
# are exactly correct.
@inline func(x, y, z) = convert(typeof(x), exp(-1) + 3x - y/7 + z + 2x*y - 3x*z + 4y*z - 5x*y*z)

function run_field_interpolation_tests(grid)
    arch = architecture(grid)
    velocities = VelocityFields(grid)
    tracers = TracerFields((:c,), grid)

    (u, v, w), c = velocities, tracers.c

    # Maximum expected rounding error is the unit in last place of the maximum value
    # of func over the domain of the grid.

    # TODO: remove this allowscalar when `nodes` returns broadcastable object on GPU
    xf, yf, zf = nodes(grid, (Face(), Face(), Face()), reshape=true)
    f_max = @allowscalar maximum(func.(xf, yf, zf))
    ε_max = eps(f_max)
    tolerance = 10 * ε_max

    set!(u, func)
    set!(v, func)
    set!(w, func)
    set!(c, func)

    # Check that interpolating to the field's own grid points returns
    # the same value as the field itself.
    for f in (u, v, w, c)
        loc = Tuple(L() for L in location(f))

        result = true
        @allowscalar begin
            for k in size(f, 3), j in size(f, 2), i in size(f, 1)
                x, y, z = Oceananigans.node(i, j, k, f)
                ℑf = interpolate((x, y, z), f, loc, f.grid)
                true_value = interior(f, i, j, k)[]

                # If at last one of the points is not approximately equal to the true value, set result to false and break
                if !isapprox(ℑf, true_value, atol=tolerance)
                    result = false
                    break
                end
            end
        end
        @test result
    end

    # Check that interpolating between grid points works as expected.

    xs = Array(reshape([0.3, 0.55, 0.73], (3, 1, 1)))
    ys = Array(reshape([-π/6, 0, 1+1e-7], (1, 3, 1)))
    zs = Array(reshape([-1.3, 1.23, 2.1], (1, 1, 3)))

    X = [(xs[i], ys[j], zs[k]) for i=1:3, j=1:3, k=1:3]
    X = on_architecture(arch, X)

    xs = on_architecture(arch, xs)
    ys = on_architecture(arch, ys)
    zs = on_architecture(arch, zs)

    @allowscalar begin
        for f in (u, v, w, c)
            loc = Tuple(L() for L in location(f))
            result = true
            for k in size(f, 3), j in size(f, 2), i in size(f, 1)
                xi, yi, zi = Oceananigans.node(i, j, k, f)
                ℑf = interpolate((xi, yi, zi), f, loc, f.grid)
                true_value = func(xi, yi, zi)

                # If at last one of the points is not approximately equal to the true value, set result to false and break
                if !isapprox(ℑf, true_value, atol=tolerance)
                    result = false
                    break
                end
            end
            @test result

            # for the next test we first call fill_halo_regions! on the
            # original field `f`
            # note, that interpolate! will call fill_halo_regions! on
            # the interpolated field after the interpolation
            fill_halo_regions!(f)

            f_copy = deepcopy(f)
            fill!(f_copy, 0)
            interpolate!(f_copy, f)

            @test all(interior(f_copy) .≈ interior(f))
        end
    end

    @info "    Testing the convert functions"
    for n in 1:30
        @test convert_to_0_360(- 10.e0^(-n)) > 359
        @test convert_to_0_360(- 10.f0^(-n)) > 359
        @test convert_to_0_360(10.e0^(-n))   < 1
        @test convert_to_0_360(10.f0^(-n))   < 1
    end

    # Generating a random longitude left bound between -1000 and 1000
    λs₀ = rand(1000) .* 2000 .- 1000

    # Generating a random interpolation longitude
    λsᵢ = rand(1000) .* 2000 .- 1000

    for λ₀ in λs₀, λᵢ in λsᵢ
        @test λ₀ ≤ convert_to_λ₀_λ₀_plus360(λᵢ, λ₀) ≤ λ₀ + 360
    end

    # Check interpolation on Windowed fields
    wf = ZFaceField(grid; indices=(:, :, grid.Nz+1))
    If = Field{Center, Center, Nothing}(grid)
    set!(If, (x, y)-> x * y)
    interpolate!(wf, If)

    @allowscalar begin
        @test all(interior(wf) .≈ interior(If))
    end

    # interpolation between fields on latitudelongitude grids with different longitudes
    grid1 = LatitudeLongitudeGrid(size=(10, 1, 1), longitude=(    0,       360), latitude=(-90, 90), z=(0, 1))
    grid2 = LatitudeLongitudeGrid(size=(10, 1, 1), longitude=( -180,       180), latitude=(-90, 90), z=(0, 1))
    grid3 = LatitudeLongitudeGrid(size=(10, 1, 1), longitude=(-1080, -1080+360), latitude=(-90, 90), z=(0, 1))
    grid4 = LatitudeLongitudeGrid(size=(10, 1, 1), longitude=(  180,       540), latitude=(-90, 90), z=(0, 1))

    f1 = CenterField(grid1)
    f2 = CenterField(grid2)
    f3 = CenterField(grid3)
    f4 = CenterField(grid4)

    set!(f1, (λ, y, z) -> λ)
    fill_halo_regions!(f1)
    interpolate!(f2, f1)
    interpolate!(f3, f1)
    interpolate!(f4, f1)

    @test all(interior(f2) .≈ map(convert_to_0_360, λnodes(grid2, Center())))
    @test all(interior(f3) .≈ map(convert_to_0_360, λnodes(grid3, Center())))
    @test all(interior(f4) .≈ map(convert_to_0_360, λnodes(grid4, Center())))

    # now interpolate back
    fill_halo_regions!(f2)
    fill_halo_regions!(f3)
    fill_halo_regions!(f4)

    interpolate!(f1, f2)
    @test all(interior(f1) .≈ λnodes(grid1, Center()))

    interpolate!(f1, f3)
    @test all(interior(f1) .≈ λnodes(grid1, Center()))

    interpolate!(f1, f4)
    @test all(interior(f1) .≈ λnodes(grid1, Center()))

    return nothing
end

function nodes_of_field_views_are_consistent(grid)
    # Test with different field types
    test_fields = [CenterField(grid), XFaceField(grid), YFaceField(grid), ZFaceField(grid)]

    for field in test_fields
        loc = instantiated_location(field)

        # Test various view patterns
        test_indices = [
            (2:6, :, :),           # x slice
            (:, 2:4, :),           # y slice
            (:, :, 2:3),           # z slice
            (3:5, 2:4, :),         # xy slice
            (2:6, :, 2:3),         # xz slice
            (:, 2:4, 2:3),         # yz slice
            (3:5, 2:4, 2:3),       # xyz slice
        ]

        for test_idx in test_indices
            # Create field view with these indices
            field_view = view(field, test_idx...)

            # Get nodes from the view
            view_nodes = nodes(field_view)

            # Get nodes from the original field with the same indices
            # This is what should be equivalent to the view_nodes
            full_nodes = nodes(field.grid, loc...; indices=test_idx)

            # Test that they are equal
            @test view_nodes == full_nodes

            # Also test that the view's indices match what we expect
            @test indices(field_view) == test_idx

            # Test that view nodes have sizes consistent with the view indices
            for (i, coord_nodes) in enumerate(view_nodes)
                if coord_nodes !== nothing && full_nodes[i] !== nothing
                    @test coord_nodes == full_nodes[i]
                end
            end
        end
    end

    return nothing
end


#####
#####
#####

@testset "Fields" begin
    @info "Testing Fields..."

    @testset "Field initialization" begin
        @info "  Testing Field initialization..."

        N = (4, 6, 8)
        L = (2π, 3π, 5π)
        H = (1, 1, 1)

        for arch in archs, FT in float_types
            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Periodic))
            @test correct_field_size(grid, (Center, Center, Center), N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Face,   Center, Center), N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center, Face,   Center), N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center, Center, Face),   N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])

            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Bounded))
            @test correct_field_size(grid, (Center, Center, Center), N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Face, Center, Center),   N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center, Face, Center),   N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center, Center, Face),   N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3] + 1)

            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Periodic, Bounded, Bounded))
            @test correct_field_size(grid, (Center, Center, Center), N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Face, Center, Center),   N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center, Face, Center),   N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center, Center, Face),   N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])

            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Bounded, Bounded, Bounded))
            @test correct_field_size(grid, (Center, Center, Center), N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Face, Center, Center),   N[1] + 1 + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center, Face, Center),   N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center, Center, Face),   N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])

            # Reduced fields
            @test correct_field_size(grid, (Nothing, Center,  Center),  1,               N[2] + 2 * H[2],     N[3] + 2 * H[3])
            @test correct_field_size(grid, (Nothing, Center,  Center),  1,               N[2] + 2 * H[2],     N[3] + 2 * H[3])
            @test correct_field_size(grid, (Nothing, Face,    Center),  1,               N[2] + 2 * H[2] + 1, N[3] + 2 * H[3])
            @test correct_field_size(grid, (Nothing, Face,    Face),    1,               N[2] + 2 * H[2] + 1, N[3] + 2 * H[3] + 1)
            @test correct_field_size(grid, (Center,  Nothing, Center),  N[1] + 2 * H[1], 1,                   N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center,  Nothing, Center),  N[1] + 2 * H[1], 1,                   N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center,  Center,  Nothing), N[1] + 2 * H[1], N[2] + 2 * H[2],     1)
            @test correct_field_size(grid, (Nothing, Nothing, Center),  1,               1,                   N[3] + 2 * H[3])
            @test correct_field_size(grid, (Center,  Nothing, Nothing), N[1] + 2 * H[1], 1,                   1)
            @test correct_field_size(grid, (Nothing, Nothing, Nothing), 1,               1,                   1)

            # "View" fields
            for f in [CenterField(grid), XFaceField(grid), YFaceField(grid), ZFaceField(grid)]

                test_indices = [(:, :, :), (1:2, 3:4, 5:6), (1, 1:6, :)]
                test_field_sizes  = [size(f), (2, 2, 2), (1, 6, size(f, 3))]
                test_parent_sizes = [size(parent(f)), (2, 2, 2), (1, 6, size(parent(f), 3))]

                for (t, indices) in enumerate(test_indices)
                    field_sz = test_field_sizes[t]
                    parent_sz = test_parent_sizes[t]
                    f_view = view(f, indices...)
                    f_sliced = Field(f; indices)
                    @test size(f_view) == field_sz
                    @test size(parent(f_view)) == parent_sz
                end
            end

            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Periodic))
            for side in (:east, :west, :north, :south, :top, :bottom)
                for wrong_bc in (ValueBoundaryCondition(0),
                                 FluxBoundaryCondition(0),
                                 GradientBoundaryCondition(0))

                    wrong_kw = Dict(side => wrong_bc)
                    wrong_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()); wrong_kw...)
                    @test_throws ArgumentError CenterField(grid, boundary_conditions=wrong_bcs)
                end
            end

            grid = RectilinearGrid(arch, FT, size=N[2:3], extent=L[2:3], halo=H[2:3], topology=(Flat, Periodic, Periodic))
            for side in (:east, :west)
                for wrong_bc in (ValueBoundaryCondition(0),
                                 FluxBoundaryCondition(0),
                                 GradientBoundaryCondition(0))

                    wrong_kw = Dict(side => wrong_bc)
                    wrong_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()); wrong_kw...)
                    @test_throws ArgumentError CenterField(grid, boundary_conditions=wrong_bcs)
                end
            end

            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Periodic, Bounded, Bounded))
            for side in (:east, :west, :north, :south)
                for wrong_bc in (ValueBoundaryCondition(0),
                                 FluxBoundaryCondition(0),
                                 GradientBoundaryCondition(0))

                    wrong_kw = Dict(side => wrong_bc)
                    wrong_bcs = FieldBoundaryConditions(grid, (Center(), Face(), Face()); wrong_kw...)

                    @test_throws ArgumentError Field{Center, Face, Face}(grid, boundary_conditions=wrong_bcs)
                end
            end

            if arch isa GPU
                wrong_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()),
                                                    top=FluxBoundaryCondition(zeros(FT, N[1], N[2])))
                @test_throws ArgumentError CenterField(grid, boundary_conditions=wrong_bcs)
            end
        end
    end

    @testset "Setting fields" begin
        @info "  Testing field setting..."

        FieldTypes = (CenterField, XFaceField, YFaceField, ZFaceField)

        N = (4, 6, 8)
        L = (2π, 3π, 5π)
        H = (1, 1, 1)

        int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4)]
        uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10)]
        float_vals = Any[0.0, -0.0, 6e-34, 1.0f10]
        rational_vals = Any[1//11, -23//7]
        other_vals = Any[π]
        vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

        for arch in archs, FT in float_types
            ArrayType = array_type(arch)
            grid = RectilinearGrid(arch, FT, size=N, extent=L, topology=(Periodic, Periodic, Bounded))

            for FieldType in FieldTypes, val in vals
                @test correct_field_value_was_set(grid, FieldType, val)
            end

            for loc in ((Center, Center, Center),
                        (Face, Center, Center),
                        (Center, Face, Center),
                        (Center, Center, Face),
                        (Nothing, Center, Center),
                        (Center, Nothing, Center),
                        (Center, Center, Nothing),
                        (Nothing, Nothing, Center),
                        (Nothing, Nothing, Nothing))

                field = Field(instantiate(loc), grid)
                sz = size(field)
                A = rand(FT, sz...)
                set!(field, A)
                @test @allowscalar field.data[1, 1, 1] == A[1, 1, 1]
            end

            Nx = 8
            topo = (Bounded, Bounded, Bounded)
            grid = RectilinearGrid(arch, FT, topology=topo, size=(Nx, Nx, Nx), x=(-1, 1), y=(0, 2π), z=(-1, 1))

            @info "  Testing field construction with `field` function..."

            array_data = ones(FT, Nx, Nx, Nx)
            f = field((Center, Center, Center), array_data, grid)
            @test @allowscalar all(isone, interior(f))

            # With an OffsetArray or a Field, we point to the same data
            offset_data = Oceananigans.Grids.new_data(FT, grid, (Center(), Center(), Center()))
            fill!(offset_data, 1)
            f = field((Center, Center, Center), offset_data, grid)
            @test @allowscalar all(isone, f.data)
            @test f.data === offset_data

            field_data = CenterField(grid)
            set!(field_data, 1)
            f = field((Center, Center, Center), field_data, grid)
            @test @allowscalar all(isone, interior(f))
            @test f === field_data

            number_data = FT(1)
            f = field((Center, Center, Center), number_data, grid)
            @test f.constant == 1

            function_data = (x, y, z) -> 1
            f = field((Center, Center, Center), function_data, grid)
            @test @allowscalar all(isone, interior(f))

            @info "  Testing Field constructors..."

            u = XFaceField(grid)
            v = YFaceField(grid)
            w = ZFaceField(grid)
            c = CenterField(grid)

            f(x, y, z) = exp(x) * sin(y) * tanh(z)

            ϕs = (u, v, w, c)
            [set!(ϕ, f) for ϕ in ϕs]

            xu, yu, zu = nodes(u)
            xv, yv, zv = nodes(v)
            xw, yw, zw = nodes(w)
            xc, yc, zc = nodes(c)

            @test @allowscalar u[1, 2, 3] ≈ f(xu[1], yu[2], zu[3])
            @test @allowscalar v[1, 2, 3] ≈ f(xv[1], yv[2], zv[3])
            @test @allowscalar w[1, 2, 3] ≈ f(xw[1], yw[2], zw[3])
            @test @allowscalar c[1, 2, 3] ≈ f(xc[1], yc[2], zc[3])

            # Test for Field-to-Field setting on same architecture, and cross architecture.
            # The behavior depends on halo size: if the halos of two fields are the same, we can
            # (easily) copy halo data over.
            # Otherwise, we take the easy way out (for now) and only copy interior data.
            big_halo = (3, 3, 3)
            small_halo = (1, 1, 1)
            domain = (; x=(0, 1), y=(0, 1), z=(0, 1))
            sz = (3, 3, 3)

            grid = RectilinearGrid(arch, FT; halo=big_halo, size=sz, domain...)
            a = CenterField(grid)
            b = CenterField(grid)
            parent(a) .= 1
            set!(b, a)
            @test parent(b) == parent(a)

            grid_with_smaller_halo = RectilinearGrid(arch, FT; halo=small_halo, size=sz, domain...)
            c = CenterField(grid_with_smaller_halo)
            set!(c, a)
            @test interior(c) == interior(a)

            # Cross-architecture setting should have similar behavior
            if arch isa GPU
                cpu_grid = RectilinearGrid(CPU(), FT; halo=big_halo, size=sz, domain...)
                d = CenterField(cpu_grid)
                set!(d, a)
                @test parent(d) == Array(parent(a))
                @test d == a
                @test a == d

                cpu_grid_with_smaller_halo = RectilinearGrid(CPU(), FT; halo=small_halo, size=sz, domain...)
                e = CenterField(cpu_grid_with_smaller_halo)
                set!(e, a)
                @test e == a
                @test a == e
                @test Array(interior(e)) == Array(interior((a)))
            end
        end
    end

    @testset "isapprox on Fields" begin
        for arch in archs, FT in float_types
            # Make sure this doesn't require scalar indexing
            GPUArraysCore.allowscalar(false)

            rect_grid = RectilinearGrid(arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

            H = 100.0
            W = 1000.0
            mountain(x, y) = H * exp(-(x^2 + y^2) / 2W^2)
            imm_grid = ImmersedBoundaryGrid(rect_grid, GridFittedBottom(mountain))

            for grid in (rect_grid, imm_grid)
                @info "  Testing isapprox on fields [$(typeof(arch)), $FT, $(nameof(typeof(grid)))]..."
                u = CenterField(grid)
                v = CenterField(grid)
                set!(u, 1)
                set!(v, 1)
                Oceananigans.ImmersedBoundaries.mask_immersed_field!(u, 1)
                Oceananigans.ImmersedBoundaries.mask_immersed_field!(v, 2)
                # Make sure the two fields are the same
                @test isapprox(u, v)
                @test isapprox(u, v; rtol=0, atol=0)

                set!(v, FT(1.1))
                @test !isapprox(u, v)
                @test isapprox(u, v; rtol=0.1)
                @test !isapprox(u, v; atol=2.0)
                # norm(u) = √512, norm(v) = 1.1 * √512, difference is 0.1 * √512 ∼ 2.26274,
                # we use a slightly larger tolerance to make the check successful.
                @test isapprox(u, v; atol=2.26275)
            end
        end
    end

    @testset "Field reductions" begin
        for arch in archs, FT in float_types
            @info "  Testing field reductions [$(typeof(arch)), $FT]..."
            N = 8
            topo = (Bounded, Bounded, Bounded)
            size = (N, N, N)
            y = (0, 2π)
            z = (-1, 1)

            x = (-1, 1)
            regular_grid = RectilinearGrid(arch, FT; topology=topo, size, x, y, z)

            x = range(-1, stop=1, length=N+1)
            variably_spaced_grid = RectilinearGrid(arch, FT; topology=topo, size, x, y, z)

            for (name, grid) in [(:regular_grid => regular_grid),
                                 (:variably_spaced_grid => variably_spaced_grid)]
                @info "    Testing field reductions on $name..."
                run_field_reduction_tests(grid)
            end
        end

        for arch in archs, FT in float_types
            @info "    Test reductions on WindowedFields [$(typeof(arch)), $FT]..."

            grid = RectilinearGrid(arch, FT, size=(2, 3, 4), x=(0, 1), y=(0, 1), z=(0, 1))
            c = CenterField(grid)
            Random.seed!(42)
            set!(c, rand(size(c)...))

            windowed_c = view(c, :, 2:3, 1:2)

            for fun in (sum, maximum, minimum)
                @test fun(c) ≈ fun(interior(c))
                @test fun(windowed_c) ≈ fun(interior(windowed_c))
            end

            @test mean(c) ≈ @allowscalar mean(interior(c))
            @test mean(windowed_c) ≈ @allowscalar mean(interior(windowed_c))
        end
    end

    @testset "Unit interpolation" begin
        for arch in archs
            hu = (-1, 1)
            hs = range(-1, 1, length=21)
            zu = (-100, 0)
            zs = range(-100, 0, length=33)

            for latitude in (hu, hs), longitude in (hu, hs), z in (zu, zs), loc in (Center(), Face())
                @info "    Testing interpolation for $(latitude) latitude and longitude, $(z) z on $(typeof(loc))s..."
                grid = LatitudeLongitudeGrid(arch; size = (20, 20, 32), longitude, latitude, z, halo = (5, 5, 5))

                # Test random positions,
                # set seed for reproducibility
                Random.seed!(1234)
                Xs = [(2rand()-1, 2rand()-1, -100rand()) for p in 1:20]

                for X in Xs
                    (x, y, z)  = X
                    fi = @allowscalar FractionalIndices(X, grid, loc, loc, loc)

                    i⁻, i⁺, _ = interpolator(fi.i)
                    j⁻, j⁺, _ = interpolator(fi.j)
                    k⁻, k⁺, _ = interpolator(fi.k)

                    x⁻ = @allowscalar ξnode(i⁻, j⁻, k⁻, grid, loc, loc, loc)
                    y⁻ = @allowscalar ηnode(i⁻, j⁻, k⁻, grid, loc, loc, loc)
                    z⁻ = @allowscalar rnode(i⁻, j⁻, k⁻, grid, loc, loc, loc)

                    x⁺ = @allowscalar ξnode(i⁺, j⁺, k⁺, grid, loc, loc, loc)
                    y⁺ = @allowscalar ηnode(i⁺, j⁺, k⁺, grid, loc, loc, loc)
                    z⁺ = @allowscalar rnode(i⁺, j⁺, k⁺, grid, loc, loc, loc)

                    @test x⁻ ≤ x ≤ x⁺
                    @test y⁻ ≤ y ≤ y⁺
                    @test z⁻ ≤ z ≤ z⁺
                end
            end
        end
    end

    @testset "Field interpolation" begin
        for arch in archs, FT in float_types
            @info "  Testing field interpolation [$(typeof(arch)), $FT]..."
            reg_grid = RectilinearGrid(arch, FT, size=(4, 5, 7), x=(0, 1), y=(-π, π), z=(-5.3, 2.7), halo=(1, 1, 1))

            # Choose points z points to be rounded values of `reg_grid` z nodes so that interpolation matches tolerance
            stretched_grid = RectilinearGrid(arch,
                                             size = (4, 5, 7),
                                             halo = (1, 1, 1),
                                             x = [0.0, 0.26, 0.49, 0.78, 1.0],
                                             y = [-3.1, -1.9, -0.6, 0.6, 1.9, 3.1],
                                             z = [-5.3, -4.2, -3.0, -1.9, -0.7, 0.4, 1.6, 2.7])

            grids = [reg_grid, stretched_grid]

            for grid in grids
                run_field_interpolation_tests(grid)
            end

            x = y = z = (0, 1)
            grid = RectilinearGrid(arch, FT; size=(2, 2, 2), x, y, z)

            # Test 2D interpolation on xy-field
            # Note: Cell centers are at 0.25 and 0.75, so test points must be
            # within the interpolation domain [0.25, 0.75] in each direction
            xy_field = Field{Center, Center, Nothing}(grid)
            set!(xy_field, (x, y) -> x + y)

            node = convert.(FT, (0.4, 0.5))
            @test interpolate(node, xy_field) ≈ node[1] + node[2]
            node = convert.(FT, (0.5, 0.4))
            @test interpolate(node, xy_field) ≈ node[1] + node[2]

            # Test 2D interpolation on xz-field
            xz_field = Field{Center, Nothing, Center}(grid)
            set!(xz_field, (x, z) -> x + z)
            node = convert.(FT, (0.4, 0.5))
            @test interpolate(node, xz_field) ≈ node[1] + node[2]
            node = convert.(FT, (0.5, 0.4))
            @test interpolate(node, xz_field) ≈ node[1] + node[2]

            # Test 2D interpolation on yz-field
            yz_field = Field{Nothing, Center, Center}(grid)
            set!(yz_field, (y, z) -> y + z)
            node = convert.(FT, (0.5, 0.4))
            @test interpolate(node, yz_field) ≈ node[1] + node[2]
            node = convert.(FT, (0.4, 0.5))
            @test interpolate(node, yz_field) ≈ node[1] + node[2]

            # Test 1D interpolation on z-field
            z_field = Field{Nothing, Nothing, Center}(grid)
            set!(z_field, z -> z)
            @test interpolate(FT(0.4), z_field) ≈ FT(0.4)
        end
    end

    @testset "Field utils" begin
        @info "  Testing field utils..."

        @test has_velocities(()) == false
        @test has_velocities((:u,)) == false
        @test has_velocities((:u, :v)) == false
        @test has_velocities((:u, :v, :w)) == true

        @info "    Testing similar(f) for f::Union(Field, ReducedField)..."

        grid = RectilinearGrid(CPU(), size=(1, 1, 1), extent=(1, 1, 1))

        for X in (Center, Face), Y in (Center, Face), Z in (Center, Face)
            for arch in archs
                f = Field{X, Y, Z}(grid)
                run_similar_field_tests(f)

                for dims in (3, (1, 2), (1, 2, 3))
                    loc = reduced_location((X(), Y(), Z()); dims)
                    f = Field(loc, grid)
                    run_similar_field_tests(f)
                end
            end
        end
    end

    @testset "Views of field views" begin
        @info "  Testing views of field views..."

        Nx, Ny, Nz = 1, 1, 7

        FieldTypes = (CenterField, XFaceField, YFaceField, ZFaceField)
        ZTopologies = (Periodic, Bounded)

        for arch in archs, FT in float_types, FieldType in FieldTypes, ZTopology in ZTopologies
            grid = RectilinearGrid(arch, FT, size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z=(0, 1), topology = (Periodic, Periodic, ZTopology))
            Hx, Hy, Hz = halo_size(grid)

            c = FieldType(grid)
            set!(c, (x, y, z) -> rand())

            k_top = total_length(location(c, 3)(), topology(c, 3)(), size(grid, 3))

            # First test that the regular view is correct
            cv = view(c, :, :, 1+1:k_top-1)
            @test size(cv) == (Nx, Ny, k_top-2)
            @test size(parent(cv)) == (Nx+2Hx, Ny+2Hy, k_top-2)
            @allowscalar @test all(cv[i, j, k] == c[i, j, k] for k in 1+1:k_top-1, j in 1:Ny, i in 1:Nx)

            # Now test the views of views
            cvv = view(cv, :, :, 1+2:k_top-2)
            @test size(cvv) == (Nx, Ny, k_top-4)
            @test size(parent(cvv)) == (Nx+2Hx, Ny+2Hy, k_top-4)
            @allowscalar @test all(cvv[i, j, k] == cv[i, j, k] for k in 1+2:k_top-2, j in 1:Ny, i in 1:Nx)

            cvvv = view(cvv, :, :, 1+3:k_top-3)
            @test size(cvvv) == (1, 1, k_top-6)
            @test size(parent(cvvv)) == (Nx+2Hx, Ny+2Hy, k_top-6)
            @allowscalar @test all(cvvv[i, j, k] == cvv[i, j, k] for k in 1+3:k_top-3, j in 1:Ny, i in 1:Nx)

            @test_throws ArgumentError view(cv, :, :, 1)
            @test_throws ArgumentError view(cv, :, :, k_top)
            @test_throws ArgumentError view(cvv, :, :, 1:1+1)
            @test_throws ArgumentError view(cvv, :, :, k_top-1:k_top)
            @test_throws ArgumentError view(cvvv, :, :, 1:1+2)
            @test_throws ArgumentError view(cvvv, :, :, k_top-2:k_top)

            @test_throws BoundsError cv[:, :, 1]
            @test_throws BoundsError cv[:, :, k_top]
            @test_throws BoundsError cvv[:, :, 1:1+1]
            @test_throws BoundsError cvv[:, :, k_top-1:k_top]
            @test_throws BoundsError cvvv[:, :, 1:1+2]
            @test_throws BoundsError cvvv[:, :, k_top-2:k_top]
        end
    end

    @testset "Field nodes and view consistency" begin
        @info "  Testing that nodes() returns indices consistent with view()..."
        for arch in archs, FT in float_types
            # Test RectilinearGrid
            rectilinear_grid = RectilinearGrid(arch, FT, size=(8, 6, 4), extent=(2, 3, 1))
            nodes_of_field_views_are_consistent(rectilinear_grid)

            # Test LatitudeLongitudeGrid
            latlon_grid = LatitudeLongitudeGrid(arch, FT, size=(8, 6, 4), longitude = (-180, 180), latitude = (-85, 85), z = (-100, 0))
            nodes_of_field_views_are_consistent(latlon_grid)

            # Test OrthogonalSphericalShellGrid (TripolarGrid)
            tripolar_grid = TripolarGrid(arch, FT, size=(8, 6, 4))
            nodes_of_field_views_are_consistent(tripolar_grid)

            # Test Flat topology behavior for RectilinearGrid
            flat_rlgrid = RectilinearGrid(arch, FT, size=(), extent=(), topology=(Flat, Flat, Flat))
            c_flat = CenterField(flat_rlgrid)
            @test nodes(c_flat) == (nothing, nothing, nothing)

            # Test Flat topology behavior for LatitudeLongitudeGrid
            flat_llgrid = LatitudeLongitudeGrid(arch, FT, size=(), topology=(Flat, Flat, Flat))
            c_flat = CenterField(flat_llgrid)
            @test nodes(c_flat) == (nothing, nothing, nothing)
        end
    end

    @testset "Constant field prognostic state" begin
        @info "  Testing prognostic_state for constant fields..."
        zf = ZeroField()
        of = OneField()
        cf = ConstantField(42)

        @test prognostic_state(zf) === nothing
        @test prognostic_state(of) === nothing
        @test prognostic_state(cf) === nothing

        @test restore_prognostic_state!(zf, nothing) === zf
        @test restore_prognostic_state!(of, :some_state) === of
        @test restore_prognostic_state!(cf, nothing) === cf
    end
end
