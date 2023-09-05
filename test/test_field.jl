include("dependencies_for_runtests.jl")

using Statistics

using Oceananigans.Fields: ReducedField, has_velocities
using Oceananigans.Fields: VelocityFields, TracerFields, interpolate
using Oceananigans.Fields: reduced_location

"""
    correct_field_size(grid, FieldType, Tx, Ty, Tz)

Test that the field initialized by the FieldType constructor on `grid`
has size `(Tx, Ty, Tz)`.
"""
correct_field_size(grid, loc, Tx, Ty, Tz) = size(parent(Field(loc, grid))) == (Tx, Ty, Tz)

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
    return all(interior(f) .≈ val * arch_array(arch, ones(size(f))))
end

function run_field_reduction_tests(FT, arch)
    N = 8
    topo = (Bounded, Bounded, Bounded)
    grid = RectilinearGrid(arch, FT, topology=topo, size=(N, N, N), x=(-1, 1), y=(0, 2π), z=(-1, 1))

    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    c = CenterField(grid)

    f(x, y, z) = 1 + exp(x) * sin(y) * tanh(z)

    ϕs = (u, v, w, c)
    [set!(ϕ, f) for ϕ in ϕs]

    u_vals = f.(nodes(u, reshape=true)...)
    v_vals = f.(nodes(v, reshape=true)...)
    w_vals = f.(nodes(w, reshape=true)...)
    c_vals = f.(nodes(c, reshape=true)...)

    # Convert to CuArray if needed.
    u_vals = arch_array(arch, u_vals)
    v_vals = arch_array(arch, v_vals)
    w_vals = arch_array(arch, w_vals)
    c_vals = arch_array(arch, c_vals)

    ϕs_vals = (u_vals, v_vals, w_vals, c_vals)

    dims_to_test = (1, 2, 3, (1, 2), (1, 3), (2, 3), (1, 2, 3))

    for (ϕ, ϕ_vals) in zip(ϕs, ϕs_vals)

        ε = eps(eltype(ϕ_vals)) * 10 * maximum(maximum.(ϕs_vals))
        @info "    Testing field reductions with tolerance $ε..."

        @test CUDA.@allowscalar all(isapprox.(ϕ, ϕ_vals, atol=ε)) # if this isn't true, reduction tests can't pass

        # Important to make sure no CUDA scalar operations occur!
        CUDA.allowscalar(false)

        @test minimum(ϕ) ≈ minimum(ϕ_vals) atol=ε
        @test maximum(ϕ) ≈ maximum(ϕ_vals) atol=ε
        @test mean(ϕ) ≈ mean(ϕ_vals) atol=2ε
        @test minimum(∛, ϕ) ≈ minimum(∛, ϕ_vals) atol=ε
        @test maximum(abs, ϕ) ≈ maximum(abs, ϕ_vals) atol=ε
        @test mean(abs2, ϕ) ≈ mean(abs2, ϕ) atol=ε

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

function run_field_interpolation_tests(grid)
    velocities = VelocityFields(grid)
    tracers = TracerFields((:c,), grid)

    (u, v, w), c = velocities, tracers.c

    # Choose a trilinear function so trilinear interpolation can return values that
    # are exactly correct.
    f(x, y, z) = convert(typeof(x), exp(-1) + 3x - y/7 + z + 2x*y - 3x*z + 4y*z - 5x*y*z)

    # Maximum expected rounding error is the unit in last place of the maximum value
    # of f over the domain of the grid.

    # TODO: remove this allowscalar when `nodes` returns broadcastable object on GPU
    xf, yf, zf = nodes(grid, (Face(), Face(), Face()), reshape=true)
    f_max = CUDA.@allowscalar maximum(f.(xf, yf, zf))
    ε_max = eps(f_max)
    tolerance = 10 * ε_max

    set!(u, f)
    set!(v, f)
    set!(w, f)
    set!(c, f)

    # Check that interpolating to the field's own grid points returns
    # the same value as the field itself.

    CUDA.@allowscalar begin
        ℑu = interpolate.(Ref(u), nodes(u, reshape=true)...)
        ℑv = interpolate.(Ref(v), nodes(v, reshape=true)...)
        ℑw = interpolate.(Ref(w), nodes(w, reshape=true)...)
        ℑc = interpolate.(Ref(c), nodes(c, reshape=true)...)

        @test all(isapprox.(ℑu, Array(interior(u)), atol=tolerance))
        @test all(isapprox.(ℑv, Array(interior(v)), atol=tolerance))
        @test all(isapprox.(ℑw, Array(interior(w)), atol=tolerance))
        @test all(isapprox.(ℑc, Array(interior(c)), atol=tolerance))
    end

    # Check that interpolating between grid points works as expected.

    xs = reshape([0.3, 0.55, 0.73], (3, 1, 1))
    ys = reshape([-π/6, 0, 1+1e-7], (1, 3, 1))
    zs = reshape([-1.3, 1.23, 2.1], (1, 1, 3))

    CUDA.@allowscalar begin
        ℑu = interpolate.(Ref(u), xs, ys, zs)
        ℑv = interpolate.(Ref(v), xs, ys, zs)
        ℑw = interpolate.(Ref(w), xs, ys, zs)
        ℑc = interpolate.(Ref(c), xs, ys, zs)

        F = f.(xs, ys, zs)

        @test all(isapprox.(ℑu, F, atol=tolerance))
        @test all(isapprox.(ℑv, F, atol=tolerance))
        @test all(isapprox.(ℑw, F, atol=tolerance))
        @test all(isapprox.(ℑc, F, atol=tolerance))
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
                    wrong_bcs = FieldBoundaryConditions(grid, (Center, Center, Center); wrong_kw...)
                    @test_throws ArgumentError CenterField(grid, boundary_conditions=wrong_bcs)
                end
            end

            grid = RectilinearGrid(arch, FT, size=N[2:3], extent=L[2:3], halo=H[2:3], topology=(Flat, Periodic, Periodic))
            for side in (:east, :west)
                for wrong_bc in (ValueBoundaryCondition(0), 
                                 FluxBoundaryCondition(0),
                                 GradientBoundaryCondition(0))

                    wrong_kw = Dict(side => wrong_bc)
                    wrong_bcs = FieldBoundaryConditions(grid, (Center, Center, Center); wrong_kw...)
                    @test_throws ArgumentError CenterField(grid, boundary_conditions=wrong_bcs)
                end
            end

            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Periodic, Bounded, Bounded))
            for side in (:east, :west, :north, :south)
                for wrong_bc in (ValueBoundaryCondition(0), 
                                 FluxBoundaryCondition(0),
                                 GradientBoundaryCondition(0))

                    wrong_kw = Dict(side => wrong_bc)
                    wrong_bcs = FieldBoundaryConditions(grid, (Center, Face, Face); wrong_kw...)

                    @test_throws ArgumentError Field{Center, Face, Face}(grid, boundary_conditions=wrong_bcs)
                end
            end

            if arch isa GPU
                wrong_bcs = FieldBoundaryConditions(grid, (Center, Center, Center),
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

                field = Field(loc, grid)
                sz = size(field)
                A = rand(FT, sz...)
                set!(field, A)
                @test CUDA.@allowscalar field.data[1, 1, 1] == A[1, 1, 1]
            end

            Nx = 8
            topo = (Bounded, Bounded, Bounded)
            grid = RectilinearGrid(arch, FT, topology=topo, size=(Nx, Nx, Nx), x=(-1, 1), y=(0, 2π), z=(-1, 1))

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

            @test CUDA.@allowscalar u[1, 2, 3] ≈ f(xu[1], yu[2], zu[3])
            @test CUDA.@allowscalar v[1, 2, 3] ≈ f(xv[1], yv[2], zv[3])
            @test CUDA.@allowscalar w[1, 2, 3] ≈ f(xw[1], yw[2], zw[3])
            @test CUDA.@allowscalar c[1, 2, 3] ≈ f(xc[1], yc[2], zc[3])

            # Test for Field-to-Field setting on same architecture, and cross architecture.
            # The behavior depends on halo size: if the halos of two fields are the same, we can
            # (easily) copy halo data over.
            # Otherwise, we take the easy way out (for now) and only copy interior data.
            big_halo = (3, 3, 3)
            small_halo = (1, 1, 1)
            domain = (; x=(0, 1), y=(0, 1), z=(0, 1))
            sz = (1, 1, 1)

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

                cpu_grid_with_smaller_halo = RectilinearGrid(CPU(), FT; halo=small_halo, size=sz, domain...)
                e = CenterField(cpu_grid_with_smaller_halo)
                set!(e, a)
                @test Array(interior(e)) == Array(interior((a)))
            end
        end
    end

    @testset "Field reductions" begin
        @info "  Testing field reductions..."

        for arch in archs, FT in float_types
            run_field_reduction_tests(FT, arch)
        end
    end

    @testset "Field interpolation" begin
        @info "  Testing field interpolation..."

        for arch in archs, FT in float_types
            reg_grid = RectilinearGrid(arch, FT, size=(4, 5, 7), x=(0, 1), y=(-π, π), z=(-5.3, 2.7), halo=(1, 1, 1))
            # Chosen these z points to be rounded values of `reg_grid` z nodes so that interpolation matches tolerance

            stretched_grid = RectilinearGrid(arch, size=(4, 5, 7),
                                             x = [0.0, 0.26, 0.49, 0.78, 1.0],
                                             y = [-3.1, -1.9, -0.6, 0.6, 1.9, 3.1],
                                             z = [-5.3, -4.2, -3.0, -1.9, -0.7, 0.4, 1.6, 2.7], halo=(1, 1, 1))
    
            grids = [reg_grid, stretched_grid]

            for grid in grids
                run_field_interpolation_tests(grid)
            end
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
                    loc = reduced_location((X, Y, Z); dims)
                    f = Field(loc, grid)
                    run_similar_field_tests(f)
                end
            end
        end
    end
end
