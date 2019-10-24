"""
    test_init_field(N, L, ftf)

Test that the field initialized by the field type function `ftf` on the grid g
has the correct size.
"""
function correct_field_size(arch, grid, fieldtype)
    f = fieldtype(arch, grid)
    return size(f) == size(grid)
end

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
    println("Testing fields...")

    N = (4, 6, 8)
    L = (2π, 3π, 5π)

    fieldtypes = (CellField, FaceFieldX, FaceFieldY, FaceFieldZ)

    @testset "Field initialization" begin
        println("  Testing field initialization...")
        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT; size=N, length=L)

            for fieldtype in fieldtypes
                @test correct_field_size(arch, grid, fieldtype)
            end
        end
    end

    int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
    uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
    float_vals = Any[0.0, -0.0, 6e-34, 1.0f10]
    rational_vals = Any[1//11, -23//7]
    other_vals = Any[π]
    vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

    @testset "Setting fields" begin
        println("  Testing field setting...")

        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT; size=N, length=L)

            for fieldtype in fieldtypes, val in vals
                @test correct_field_value_was_set(arch, grid, fieldtype, val)
            end
        end
    end
end
