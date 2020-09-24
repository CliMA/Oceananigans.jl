using Oceananigans.Fields: cpudata

"""
    correct_field_size(arch, grid, FieldType, Tx, Ty, Tz)

Test that the field initialized by the FieldType constructor on `arch` and `grid`
has size `(Tx, Ty, Tz)`.
"""
correct_field_size(a, g, FieldType, Tx, Ty, Tz) = size(parent(FieldType(a, g))) == (Tx, Ty, Tz)

"""
    correct_reduced_field_size(loc, arch, grid, dims, Tx, Ty, Tz)

Test that the ReducedField at `loc`ation on `arch`itecture and `grid`
and reduced along `dims` has size `(Tx, Ty, Tz)`.
"""
correct_reduced_field_size(loc, arch, grid, dims, Tx, Ty, Tz) =
    size(parent(ReducedField(loc, arch, grid; dims=dims))) == (Tx, Ty, Tz)

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

function correct_reduced_field_value_was_set(arch, grid, loc, dims, val::Number)
    f = ReducedField(loc, arch, grid; dims=dims)
    set!(f, val)
    CUDA.@allowscalar return interior(f) ≈ val * ones(size(f))
end


@testset "Fields" begin
    @info "Testing fields..."

    N = (4, 6, 8)
    L = (2π, 3π, 5π)
    H = (1, 1, 1)

    @testset "Field initialization" begin
        @info "  Testing field initialization..."
        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Periodic))
            @test correct_field_size(arch, grid, CellField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])

            grid = RegularCartesianGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Bounded))
            @test correct_field_size(arch, grid, CellField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3] + 1)

            grid = RegularCartesianGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Bounded, Bounded))
            @test correct_field_size(arch, grid, CellField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField, N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])

            grid = RegularCartesianGrid(FT, size=N, extent=L, halo=H, topology=(Bounded, Bounded, Bounded))
            @test correct_field_size(arch, grid, CellField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField, N[1] + 1 + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField, N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])

            @test correct_reduced_field_size((Cell, Cell, Cell), arch, grid, 1,         1,               N[2] + 2 * H[2],     N[3] + 2 * H[3])
            @test correct_reduced_field_size((Face, Cell, Cell), arch, grid, 1,         1,               N[2] + 2 * H[2],     N[3] + 2 * H[3])
            @test correct_reduced_field_size((Cell, Face, Cell), arch, grid, 1,         1,               N[2] + 2 * H[2] + 1, N[3] + 2 * H[3])
            @test correct_reduced_field_size((Cell, Face, Face), arch, grid, 1,         1,               N[2] + 2 * H[2] + 1, N[3] + 2 * H[3] + 1)
            @test correct_reduced_field_size((Cell, Cell, Cell), arch, grid, 2,         N[1] + 2 * H[1], 1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Cell, Cell, Cell), arch, grid, 2,         N[1] + 2 * H[1], 1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Cell, Cell, Cell), arch, grid, 3,         N[1] + 2 * H[1], N[2] + 2 * H[2],     1)
            @test correct_reduced_field_size((Cell, Cell, Cell), arch, grid, (1, 2),    1,               1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Cell, Cell, Cell), arch, grid, (2, 3),    N[1] + 2 * H[1], 1,                   1)
            @test correct_reduced_field_size((Cell, Cell, Cell), arch, grid, (1, 2, 3), 1,               1,                   1)
        end
    end

    FieldTypes = (CellField, XFaceField, YFaceField, ZFaceField)
    reduced_dims = (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))

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
            grid = RegularCartesianGrid(FT, size=N, extent=L, topology=(Periodic, Periodic, Bounded))

            for FieldType in FieldTypes, val in vals
                @test correct_field_value_was_set(arch, grid, FieldType, val)
            end

            for dims in reduced_dims, val in vals
                @test correct_reduced_field_value_was_set(arch, grid, (Cell, Cell, Cell), dims, val)
            end

            for FieldType in FieldTypes
                field = FieldType(arch, grid)
                sz = size(field)
                A = rand(FT, sz...) |> ArrayType
                set!(field, A)
                @test field.data[2, 4, 6] == A[2, 4, 6]
            end

            for dims in reduced_dims
                reduced_field = ReducedField((Cell, Cell, Cell), arch, grid, dims=dims)
                sz = size(reduced_field)
                A = rand(FT, sz...) |> ArrayType
                set!(reduced_field, A)
                @test reduced_field.data[1, 1, 1] == A[1, 1, 1]
            end
        end
    end

    @testset "Conditional computation" begin
        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(size=(2, 2, 2), extent=(1, 1, 1)) 
            c = CellField(FT, arch, grid)

            for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))
                C = AveragedField(c, dims=dims)

                # Test conditional computation
                set!(c, 1)
                compute!(C, 1.0)
                @test all(C.data .== 1)

                set!(c, 2)
                compute!(C, 1.0)
                @test all(C.data .== 1)

                compute!(C, 2.0)
                @test all(C.data .== 2)
            end
        end
    end

    @testset "Field utils" begin
        @info "  Testing field utils..."

        @test Fields.has_velocities(()) == false
        @test Fields.has_velocities((:u,)) == false
        @test Fields.has_velocities((:u, :v)) == false
        @test Fields.has_velocities((:u, :v, :w)) == true

		grid = RegularCartesianGrid(size=(4, 6, 8), extent=(1, 1, 1))
		ϕ = CellField(CPU(), grid)
		@test cpudata(ϕ).parent isa Array

		@hascuda begin
			ϕ = CellField(GPU(), grid)
			@test cpudata(ϕ).parent isa Array
		end
    end
end
