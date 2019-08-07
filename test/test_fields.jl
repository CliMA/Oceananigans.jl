"""
    test_init_field(N, L, ftf)

Test that the field initialized by the field type function `ftf` on the grid g
has the correct size.
"""
function correct_field_size(arch::Architecture, g::Grid, fieldtype)
    f = fieldtype(arch, g)
    return size(f) == size(g)
end

"""
    test_set_field(N, L, ftf, val)

Test that the field initialized by the field type function `ftf` on the grid g
can be correctly filled with the value `val` using the `set!(f::Field, v)`
function.
"""
function correct_field_value_was_set(arch::Architecture, g::Grid, fieldtype, val::Number)
    f = fieldtype(arch, g)
    set!(f, val)
    return data(f) ≈ val * ones(size(f))
end

function correct_field_addition(arch::Architecture, g::Grid, fieldtype, val1::Number, val2::Number)
    f1 = fieldtype(arch, g)
    f2 = fieldtype(arch, g)

    set!(f1, val1)
    set!(f2, val2)
    f3 = f1 + f2

    val3 = convert(mm.float_type, val1) + convert(mm.float_type, val2)
    f_ans = val3 * ones(size(f1))
    return f3.data ≈ f_ans
end

function set_velocity_tracer_fields(arch, grid, fieldname, value, answer)

    model = Model(arch=arch, float_type=eltype(grid), 
                  N=size(grid), L=(grid.Lx, grid.Ly, grid.Lz))

    kwarg = Dict(fieldname=>value)
    set!(model; kwarg...)

    if fieldname ∈ propertynames(model.velocities)
        ϕ = getproperty(model.velocities, fieldname)
    else
        ϕ = getproperty(model.tracers, fieldname)
    end

    return data(ϕ) ≈ answer
end
 
@testset "Fields" begin
    println("Testing fields...")

    N = (4, 6, 8)
    L = (2π, 3π, 5π)

    fieldtypes = (CellField, FaceFieldX, FaceFieldY, FaceFieldZ)

    @testset "Field initialization" begin
        println("  Testing field initialization...")
        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT, N, L)

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
            grid = RegularCartesianGrid(FT, N, L)

            for fieldtype in fieldtypes, val in vals
                @test correct_field_value_was_set(arch, grid, fieldtype, val)
            end
        end

        for FT in float_types
            grid = RegularCartesianGrid(FT, N, L)
            xF = reshape(grid.xF[1:end-1], N[1], 1, 1)
            yC = reshape(grid.yC, 1, N[2], 1)
            zC = reshape(grid.zC, 1, 1, N[3])

            u₀(x, y, z) = x * y^2 * z^3
            u_answer = @. xF * yC^2 * zC^3

            T₀ = rand(size(grid)...)
            T_answer = deepcopy(T₀)

            for arch in archs
                @test set_velocity_tracer_fields(arch, grid, :u, u₀, u_answer)
                @test set_velocity_tracer_fields(arch, grid, :T, T₀, T_answer)
            end
        end
    end
end
