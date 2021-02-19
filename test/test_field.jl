using Oceananigans.Fields: cpudata

"""
    correct_field_size(arch, grid, FieldType, Tx, Ty, Tz)

Test that the field initialized by the FieldType constructor on `arch` and `grid`
has size `(Tx, Ty, Tz)`.
"""
correct_field_size(a, g, FieldType, Tx, Ty, Tz) = size(parent(FieldType(a, g))) == (Tx, Ty, Tz)

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

function run_field_interpolation_tests(arch, FT)

    grid = RegularRectilinearOrthogonalGrid(size=(4, 5, 7), x=(0, 1), y=(-π, π), z=(-5.3, 2.7))

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

    N = (4, 6, 8)
    L = (2π, 3π, 5π)
    H = (1, 1, 1)

    @testset "Field initialization" begin
        @info "  Testing Field initialization..."
        for arch in archs, FT in float_types
            grid = RegularRectilinearOrthogonalGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Periodic))
            @test correct_field_size(arch, grid, CenterField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])

            grid = RegularRectilinearOrthogonalGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Bounded))
            @test correct_field_size(arch, grid, CenterField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3] + 1)

            grid = RegularRectilinearOrthogonalGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Bounded, Bounded))
            @test correct_field_size(arch, grid, CenterField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField,  N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])

            grid = RegularRectilinearOrthogonalGrid(FT, size=N, extent=L, halo=H, topology=(Bounded, Bounded, Bounded))
            @test correct_field_size(arch, grid, CenterField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField,  N[1] + 1 + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField,  N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])
        end
    end

    FieldTypes = (CenterField, XFaceField, YFaceField, ZFaceField)

    int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
    uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
    float_vals = Any[0.0, -0.0, 6e-34, 1.0f10]
    rational_vals = Any[1//11, -23//7]
    other_vals = Any[π]
    vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

    @testset "Setting fields" begin
        @info "  Testing field setting..."

        for arch in archs, FT in float_types
	        ArrayType = array_type(arch)
            grid = RegularRectilinearOrthogonalGrid(FT, size=N, extent=L, topology=(Periodic, Periodic, Bounded))

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

    @testset "Field utils" begin
        @info "  Testing field utils..."

        @test Fields.has_velocities(()) == false
        @test Fields.has_velocities((:u,)) == false
        @test Fields.has_velocities((:u, :v)) == false
        @test Fields.has_velocities((:u, :v, :w)) == true

		grid = RegularRectilinearOrthogonalGrid(size=(4, 6, 8), extent=(1, 1, 1))
		ϕ = CenterField(CPU(), grid)
		@test cpudata(ϕ).parent isa Array

		@hascuda begin
			ϕ = CenterField(GPU(), grid)
			@test cpudata(ϕ).parent isa Array
		end
    end

    @testset "Field interpolation" begin
        @info "  Testing field interpolation..."

        for arch in archs, FT in float_types
            run_field_interpolation_tests(arch, FT)
        end
    end
end
