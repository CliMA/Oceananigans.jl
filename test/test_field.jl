include("dependencies_for_runtests.jl")

using Statistics

using Oceananigans.Fields: FieldSlicer
using Oceananigans.Fields: regrid!, ReducedField, has_velocities
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

        @test all(isapprox.(ϕ, ϕ_vals, atol=ε)) # if this isn't true, reduction tests can't pass

        # Important to make sure no CUDA scalar operations occur!
        CUDA.allowscalar(false)

        @test minimum(ϕ) ≈ minimum(ϕ_vals) atol=ε
        @test maximum(ϕ) ≈ maximum(ϕ_vals) atol=ε
        @test mean(ϕ) ≈ mean(ϕ_vals) atol=2ε
        @test minimum(∛, ϕ) ≈ minimum(∛, ϕ_vals) atol=ε
        @test maximum(abs, ϕ) ≈ maximum(abs, ϕ_vals) atol=ε
        @test mean(abs2, ϕ) ≈ mean(abs2, ϕ) atol=ε

        for dims in dims_to_test
            @test all(isapprox(minimum(ϕ, dims=dims),
                               minimum(ϕ_vals, dims=dims), atol=4ε))

            @test all(isapprox(maximum(ϕ, dims=dims),
                               maximum(ϕ_vals, dims=dims), atol=4ε))

            @test all(isapprox(mean(ϕ, dims=dims),
                               mean(ϕ_vals, dims=dims), atol=4ε))

            @test all(isapprox(minimum(sin, ϕ, dims=dims),
                               minimum(sin, ϕ_vals, dims=dims), atol=4ε))

            @test all(isapprox(maximum(cos, ϕ, dims=dims),
                               maximum(cos, ϕ_vals, dims=dims), atol=4ε))

            @test all(isapprox(mean(cosh, ϕ, dims=dims),
                               mean(cosh, ϕ_vals, dims=dims), atol=5ε))
        end

        CUDA.allowscalar(true)
    end

    return nothing
end

function run_field_interpolation_tests(FT, arch)

    grid = RectilinearGrid(arch, size=(4, 5, 7), x=(0, 1), y=(-π, π), z=(-5.3, 2.7))

    velocities = VelocityFields(grid)
    tracers = TracerFields((:c,), grid)

    (u, v, w), c = velocities, tracers.c

    # Choose a trilinear function so trilinear interpolation can return values that
    # are exactly correct.
    f(x, y, z) = exp(-1) + 3x - y/7 + z + 2x*y - 3x*z + 4y*z - 5x*y*z

    # Maximum expected rounding error is the unit in last place of the maximum value
    # of f over the domain of the grid.
    ε_max = f.(nodes((Face, Face, Face), grid, reshape=true)...) |> maximum |> eps

    set!(u, f)
    set!(v, f)
    set!(w, f)
    set!(c, f)

    # Check that interpolating to the field's own grid points returns
    # the same value as the field itself.

    ℑu = interpolate.(Ref(u), nodes(u, reshape=true)...)
    ℑv = interpolate.(Ref(v), nodes(v, reshape=true)...)
    ℑw = interpolate.(Ref(w), nodes(w, reshape=true)...)
    ℑc = interpolate.(Ref(c), nodes(c, reshape=true)...)

    @test all(isapprox.(ℑu, Array(interior(u)), atol=ε_max))
    @test all(isapprox.(ℑv, Array(interior(v)), atol=ε_max))
    @test all(isapprox.(ℑw, Array(interior(w)), atol=ε_max))
    @test all(isapprox.(ℑc, Array(interior(c)), atol=ε_max))

    # Check that interpolating between grid points works as expected.

    xs = reshape([0.3, 0.55, 0.73], (3, 1, 1))
    ys = reshape([-π/6, 0, 1+1e-7], (1, 3, 1))
    zs = reshape([-1.3, 1.23, 2.1], (1, 1, 3))

    ℑu = interpolate.(Ref(u), xs, ys, zs)
    ℑv = interpolate.(Ref(v), xs, ys, zs)
    ℑw = interpolate.(Ref(w), xs, ys, zs)
    ℑc = interpolate.(Ref(c), xs, ys, zs)

    F = f.(xs, ys, zs)

    @test all(isapprox.(ℑu, F, atol=ε_max))
    @test all(isapprox.(ℑv, F, atol=ε_max))
    @test all(isapprox.(ℑw, F, atol=ε_max))
    @test all(isapprox.(ℑc, F, atol=ε_max))

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

        CUDA.allowscalar(true)

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
                @test field.data[1, 1, 1] == A[1, 1, 1]
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

            @test u[1, 2, 3] ≈ f(xu[1], yu[2], zu[3])
            @test v[1, 2, 3] ≈ f(xv[1], yv[2], zv[3])
            @test w[1, 2, 3] ≈ f(xw[1], yw[2], zw[3])
            @test c[1, 2, 3] ≈ f(xc[1], yc[2], zc[3])
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
            run_field_interpolation_tests(FT, arch)
        end
    end

    @testset "Field utils" begin
        @info "  Testing field utils..."

        @test has_velocities(()) == false
        @test has_velocities((:u,)) == false
        @test has_velocities((:u, :v)) == false
        @test has_velocities((:u, :v, :w)) == true

        @test FieldSlicer() isa FieldSlicer

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

    @testset "Regridding" begin
        @info "  Testing field regridding..."

        Lz = 1.1
        ℓz = 0.5
        topology = (Flat, Flat, Bounded)
        
        for arch in archs
            fine_regular_grid                = RectilinearGrid(arch, size=(4, 6, 2), x=(0, 1), y=(0, 2), z=(0, Lz), topology=(Periodic, Periodic, Bounded))
            fine_stretched_grid              = RectilinearGrid(arch, size=(4, 6, 2), x=(0, 1), y=(0, 2), z = [0, ℓz, Lz], topology=(Periodic, Periodic, Bounded))
            coarse_column_regular_grid       = RectilinearGrid(arch, size=1, z=(0, Lz), topology=topology)
            fine_column_regular_grid         = RectilinearGrid(arch, size=2, z=(0, Lz), topology=topology)
            fine_column_stretched_grid       = RectilinearGrid(arch, size=2, z = [0, ℓz, Lz], topology=topology)
            very_fine_column_stretched_grid  = RectilinearGrid(arch, size=3, z = [0, 0.2, 0.6, Lz], topology=topology)
            super_fine_column_stretched_grid = RectilinearGrid(arch, size=4, z = [0, 0.1, 0.3, 0.65, Lz], topology=topology)
            super_fine_column_regular_grid   = RectilinearGrid(arch, size=5, z=(0, Lz), topology=topology)
            
            fine_stretched_c              = CenterField(fine_stretched_grid)
            coarse_column_regular_c       = CenterField(coarse_column_regular_grid)
            fine_column_regular_c         = CenterField(fine_column_regular_grid)
            fine_column_stretched_c       = CenterField(fine_column_stretched_grid)
            very_fine_column_stretched_c  = CenterField(very_fine_column_stretched_grid)
            super_fine_column_stretched_c = CenterField(super_fine_column_stretched_grid)
            super_fine_column_regular_c   = CenterField(super_fine_column_regular_grid)
            super_fine_from_reduction_regular_c = CenterField(super_fine_column_regular_grid)

            # we initialize an array on the `fine_column_stretched_grid`, regrid it to the rest
            # grids, and check whether we get the anticipated results
            c₁ = 1
            c₂ = 3
            CUDA.@allowscalar begin
                fine_column_stretched_c[1, 1, 1] = c₁
                fine_column_stretched_c[1, 1, 2] = c₂
            end

            fine_stretched_c[:, :, 1] .= c₁
            fine_stretched_c[:, :, 2] .= c₂

            # Coarse-graining
            regrid!(coarse_column_regular_c, fine_column_stretched_c)

            CUDA.@allowscalar begin
                @test coarse_column_regular_c[1, 1, 1] ≈ ℓz/Lz * c₁ + (1 - ℓz/Lz) * c₂
            end

            regrid!(fine_column_regular_c, fine_column_stretched_c)

            CUDA.@allowscalar begin
                @test fine_column_regular_c[1, 1, 1] ≈ ℓz/(Lz/2) * c₁ + (1 - ℓz/(Lz/2)) * c₂
                @test fine_column_regular_c[1, 1, 2] ≈ c₂
            end            

            # Fine-graining
            regrid!(very_fine_column_stretched_c, fine_column_stretched_c)

            CUDA.@allowscalar begin
                @test very_fine_column_stretched_c[1, 1, 1] ≈ c₁
                @test very_fine_column_stretched_c[1, 1, 2] ≈ (ℓz - 0.2)/0.4 * c₁ + (0.6 - ℓz)/0.4 * c₂
                @test very_fine_column_stretched_c[1, 1, 3] ≈ c₂
            end
            
            regrid!(super_fine_column_stretched_c, fine_column_stretched_c)

            CUDA.@allowscalar begin
                @test super_fine_column_stretched_c[1, 1, 1] ≈ c₁
                @test super_fine_column_stretched_c[1, 1, 2] ≈ c₁
                @test super_fine_column_stretched_c[1, 1, 3] ≈ (ℓz - 0.3)/0.35 * c₁ + (0.65 - ℓz)/0.35 * c₂
                @test super_fine_column_stretched_c[1, 1, 4] ≈ c₂
            end
            
            regrid!(super_fine_column_regular_c, fine_column_stretched_c)
            
            CUDA.@allowscalar begin
                @test super_fine_column_regular_c[1, 1, 1] ≈ c₁
                @test super_fine_column_regular_c[1, 1, 2] ≈ c₁
                @test super_fine_column_regular_c[1, 1, 3] ≈ (3 - ℓz/(Lz/5)) * c₂ + (-2 + ℓz/(Lz/5)) * c₁
                @test super_fine_column_regular_c[1, 1, 4] ≈ c₂
                @test super_fine_column_regular_c[1, 1, 5] ≈ c₂
            end

            # Fine-graining from reduction
            fine_stretched_c_mean_xy = Field(Reduction(mean!, fine_stretched_c, dims=(1, 2)))
            compute!(fine_stretched_c_mean_xy)

            regrid!(super_fine_from_reduction_regular_c, fine_stretched_c_mean_xy)
            
            CUDA.@allowscalar begin
                @test super_fine_from_reduction_regular_c[1, 1, 1] ≈ c₁
                @test super_fine_from_reduction_regular_c[1, 1, 2] ≈ c₁
                @test super_fine_from_reduction_regular_c[1, 1, 3] ≈ (3 - ℓz/(Lz/5)) * c₂ + (-2 + ℓz/(Lz/5)) * c₁
                @test super_fine_from_reduction_regular_c[1, 1, 4] ≈ c₂
                @test super_fine_from_reduction_regular_c[1, 1, 5] ≈ c₂
            end
        end
    end
end
