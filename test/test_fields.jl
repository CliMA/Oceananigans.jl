"""
    test_init_field(N, L, ftf)

Test that the field initialized by the field type function `ftf` on the grid g
has the correct size.
"""
correct_field_size(a, g, fieldtype, Tx, Ty, Tz) = size(parent(fieldtype(a, g)))  == (Tx, Ty, Tz)
    
"""
    test_set_field(N, L, ftf, val)

Test that the field initialized by the field type function `ftf` on the grid g
can be correctly filled with the value `val` using the `set!(f::AbstractField, v)`
function.
"""
function correct_field_value_was_set(arch, grid, fieldtype, val::Number)
    f = fieldtype(arch, grid)
    set!(f, val)
    return interior(f) ≈ val * ones(size(f))
end

@testset "Fields" begin
    @info "Testing fields..."

    N = (4, 6, 8)
    L = (2π, 3π, 5π)
    H = (1, 1, 1)

    fieldtypes = (CellField, XFaceField, YFaceField, ZFaceField)

    @testset "Field initialization" begin
        @info "  Testing field initialization..."
        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT, size=N, length=L, halo=H, topology=(Periodic, Periodic, Periodic))
            @test correct_field_size(arch, grid, CellField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])

            grid = RegularCartesianGrid(FT, size=N, length=L, halo=H, topology=(Periodic, Periodic, Bounded))
            @test correct_field_size(arch, grid, CellField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3] + 1)

            grid = RegularCartesianGrid(FT, size=N, length=L, halo=H, topology=(Periodic, Bounded, Bounded))
            @test correct_field_size(arch, grid, CellField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField, N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])

            grid = RegularCartesianGrid(FT, size=N, length=L, halo=H, topology=(Bounded, Bounded, Bounded))
            @test correct_field_size(arch, grid, CellField,  N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, XFaceField, N[1] + 1 + 2 * H[1], N[2] + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, YFaceField, N[1] + 2 * H[1], N[2] + 1 + 2 * H[2], N[3] + 2 * H[3])
            @test correct_field_size(arch, grid, ZFaceField, N[1] + 2 * H[1], N[2] + 2 * H[2], N[3] + 1 + 2 * H[3])
        end
    end

    int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
    uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
    float_vals = Any[0.0, -0.0, 6e-34, 1.0f10]
    rational_vals = Any[1//11, -23//7]
    other_vals = Any[π]
    vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

    @testset "Setting fields" begin
        @info "  Testing field setting..."

        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT, size=N, length=L, topology=(Periodic, Periodic, Bounded))

            for fieldtype in fieldtypes, val in vals
                @test correct_field_value_was_set(arch, grid, fieldtype, val)
            end

            for fieldtype in fieldtypes
                field = fieldtype(arch, grid)
                A = rand(FT, N...)
                arch isa GPU && (A = CuArray(A))
                set!(field, A)
                @test field.data[2, 4, 6] == A[2, 4, 6]
            end
        end
    end
end
