"""
    test_init_field(N, L, ftf)

Test that the field initialized by the field type function `ftf` on the grid g
has the correct size.
"""
function correct_field_size(arch::Architecture, g::Grid, field_type)
    f = field_type(arch, g)
    size(f) == size(g)
end

"""
    test_set_field(N, L, ftf, val)

Test that the field initialized by the field type function `ftf` on the grid g
can be correctly filled with the value `val` using the `set!(f::Field, v)`
function.
"""
function correct_field_value_was_set(arch::Architecture, g::Grid, field_type, val::Number)
    f = field_type(arch, g)
    set!(f, val)
    data(f) ≈ val * ones(size(f))
end

function correct_field_addition(arch::Architecture, g::Grid, field_type, val1::Number, val2::Number)
    f1 = field_type(arch, g)
    f2 = field_type(arch, g)

    set!(f1, val1)
    set!(f2, val2)
    f3 = f1 + f2

    val3 = convert(mm.float_type, val1) + convert(mm.float_type, val2)
    f_ans = val3 * ones(size(f1))
    f3.data ≈ f_ans
end

@testset "Fields" begin
    println("Testing fields...")

    N = (4, 6, 8)
    L = (2π, 3π, 5π)

    field_types = [CellField, FaceFieldX, FaceFieldY, FaceFieldZ, EdgeField]

    @testset "Field initialization" begin
        println("  Testing field initialization...")
        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT, N, L)

            for field_type in field_types
                @test correct_field_size(arch, grid, field_type)
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
            grid = RegularCartesianGrid(FT, N, L)

            for field_type in field_types, val in vals
                @test correct_field_value_was_set(arch, grid, field_type, val)
            end
        end
    end

    # @testset "Field operations" begin
    #     for arch in archs, FT in float_types
    #         grid = RegularCartesianGrid(FT, N, L)
    #
    #         for field_type in field_types, val1 in vals, val2 in vals
    #             @test correct_field_addition(arch, grid, field_type, val1, val2)
    #         end
    #     end
    # end
end
