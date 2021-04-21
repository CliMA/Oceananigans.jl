using Oceananigans.Fields: cpudata, FieldSlicer

"""
    correct_field_size(arch, grid, FieldType, Tx, Ty, Tz)

Test that the field initialized by the FieldType constructor on `arch` and `grid`
has size `(Tx, Ty, Tz)`.
"""
correct_field_size(a, g, FieldType, Tx, Ty, Tz) = size(parent(FieldType(a, g))) == (Tx, Ty, Tz)

function run_similar_field_tests(f)
    g = similar(f)
    @test typeof(f) == typeof(g)
    @test f.grid == g.grid
    return nothing
end

"""
     correct_field_value_was_set(N, L, ftf, val)

Test that the field initialized by the field type function `ftf` on the grid g
can be correctly filled with the value `val` using the `set!(f::AbstractField, v)`
function.
"""
function correct_field_value_was_set(arch, grid, FieldType, val::Number)
    f = FieldType(arch, grid)
    set!(f, val)
    CUDA.@allowscalar return interior(f) ≈ val * ones(size(f))
end

function run_field_reduction_tests(FT, arch)
    N = 8
    topo = (Bounded, Bounded, Bounded)
    grid = RegularRectilinearGrid(FT, topology=topo, size=(N, N, N), x=(-1, 1), y=(0, 2π), z=(-1, 1))

    u = XFaceField(arch, grid)
    v = YFaceField(arch, grid)
    w = ZFaceField(arch, grid)
    c = CenterField(arch, grid)

    f(x, y, z) = exp(x) * sin(y) * tanh(z)

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

    dims_to_test = (1, 2, 3, (1, 2), (1, 3), (2, 3))

    # Important to make sure no CUDA scalar operations occur!
    CUDA.@disallowscalar begin
        for (ϕ, ϕ_vals) in zip(ϕs, ϕs_vals)
            @test minimum(ϕ) == minimum(ϕ_vals)
            @test maximum(ϕ) == maximum(ϕ_vals)
            @test mean(ϕ) == mean(ϕ_vals)
            @test minimum(∛, ϕ) == minimum(∛, ϕ_vals)
            @test maximum(abs, ϕ) == maximum(abs, ϕ_vals)
            @test mean(abs2, ϕ) == mean(abs2, ϕ)

            for dims in dims_to_test
                @test minimum(ϕ, dims=dims) == minimum(ϕ_vals, dims=dims)
                @test maximum(ϕ, dims=dims) == maximum(ϕ_vals, dims=dims)
                @test mean(ϕ, dims=dims) == mean(ϕ_vals, dims=dims)

                @test minimum(sin, ϕ, dims=dims) == minimum(sin, ϕ_vals, dims=dims)
                @test maximum(cos, ϕ, dims=dims) == maximum(cos, ϕ_vals, dims=dims)
                @test mean(cosh, ϕ, dims=dims) == mean(cosh, ϕ, dims=dims)
            end
        end
    end

    return nothing
end

function run_field_interpolation_tests(arch, FT)

    grid = RegularRectilinearGrid(size=(4, 5, 7), x=(0, 1), y=(-π, π), z=(-5.3, 2.7))

    velocities = VelocityFields(arch, grid)
    tracers = TracerFields((:c,), arch, grid)

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

    @test all(isapprox.(ℑu, interior(u), atol=ε_max))
    @test all(isapprox.(ℑv, interior(v), atol=ε_max))
    @test all(isapprox.(ℑw, interior(w), atol=ε_max))
    @test all(isapprox.(ℑc, interior(c), atol=ε_max))

    # Check that interpolating between grid points works as expected.

    xs = reshape([0.3, 0.55, 0.73], (3, 1, 1))
    ys = reshape([-π/6, 0, 1+1e-7], (1, 3, 1))
    zs = reshape([-1.3, 1.23, 2.1], (1, 1, 3))

    F = f.(xs, ys, zs)

    ℑu = interpolate.(Ref(u), xs, ys, zs)
    ℑv = interpolate.(Ref(v), xs, ys, zs)
    ℑw = interpolate.(Ref(w), xs, ys, zs)
    ℑc = interpolate.(Ref(c), xs, ys, zs)

    @test all(isapprox.(ℑu, F, atol=ε_max))
    @test all(isapprox.(ℑv, F, atol=ε_max))
    @test all(isapprox.(ℑw, F, atol=ε_max))
    @test all(isapprox.(ℑc, F, atol=ε_max))

    return nothing
end

@testset "Fields" begin
    @info "Testing Fields..."

    @testset "Field initialization" begin
        @info "  Testing Field initialization..."

        N = (4, 6, 8)
        L = (2π, 3π, 5π)
        H = (1, 1, 1)

        for arch in archs, FT in float_types
            grid = RegularRectilinearGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Periodic))
            @test correct_field_size(arch, grid, CenterField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])

            grid = RegularRectilinearGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Bounded))
            @test correct_field_size(arch, grid, CenterField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3] + 1)

            grid = RegularRectilinearGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Bounded, Bounded))
            @test correct_field_size(arch, grid, CenterField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField,  N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])

            grid = RegularRectilinearGrid(FT, size=N, extent=L, halo=H, topology=(Bounded, Bounded, Bounded))
            @test correct_field_size(arch, grid, CenterField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField,  N[1] + 1 + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField,  N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])
        end
    end

    @testset "Setting fields" begin
        @info "  Testing field setting..."

        FieldTypes = (CenterField, XFaceField, YFaceField, ZFaceField)

        N = (4, 6, 8)
        L = (2π, 3π, 5π)
        H = (1, 1, 1)

        int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
        uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
        float_vals = Any[0.0, -0.0, 6e-34, 1.0f10]
        rational_vals = Any[1//11, -23//7]
        other_vals = Any[π]
        vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

        for arch in archs, FT in float_types
            ArrayType = array_type(arch)
            grid = RegularRectilinearGrid(FT, size=N, extent=L, topology=(Periodic, Periodic, Bounded))

            for FieldType in FieldTypes, val in vals
                @test correct_field_value_was_set(arch, grid, FieldType, val)
            end

            for FieldType in FieldTypes
                field = FieldType(arch, grid)
                sz = size(field)
                A = rand(FT, sz...) |> ArrayType
                set!(field, A)
                @test field.data[2, 4, 6] == A[2, 4, 6]
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
            run_field_interpolation_tests(arch, FT)
        end
    end

    @testset "Field utils" begin
        @info "  Testing field utils..."

        @test Fields.has_velocities(()) == false
        @test Fields.has_velocities((:u,)) == false
        @test Fields.has_velocities((:u, :v)) == false
        @test Fields.has_velocities((:u, :v, :w)) == true

        grid = RegularRectilinearGrid(size=(4, 6, 8), extent=(1, 1, 1))
        ϕ = CenterField(CPU(), grid)
        @test cpudata(ϕ).parent isa Array

        if CUDA.has_cuda()
            ϕ = CenterField(GPU(), grid)
            @test cpudata(ϕ).parent isa Array
        end

        @test FieldSlicer() isa FieldSlicer

        @info "    Testing similar(f) for f::Union(Field, ReducedField)..."

        grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

        for X in (Center, Face), Y in (Center, Face), Z in (Center, Face)
            for arch in archs
                f = Field(X, Y, Z, arch, grid)
                run_similar_field_tests(f)

                for dims in (3, (1, 2), (1, 2, 3))
                    f = ReducedField(X, Y, Z, arch, grid, dims=dims)
                    run_similar_field_tests(f)
                end
            end
        end
    end
end
