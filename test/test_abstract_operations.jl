function test_simple_binary_operation(op, a, b, num1, num2)
    a_b = op(a, b)
    data(a) .= num1
    data(b) .= num2
    return a_b[2, 1, 2] == op(num1, num2)
end

function test_three_field_addition(a, b, c, num1, num2)
    a_b_c = a + b + c
    data(a) .= num1
    data(b) .= num2
    data(c) .= num2

    return a_b_c[2, 1, 2] == num1 + num2 + num2
end

function test_x_derivative(a)
    dx_a = ∂x(a)

    for k in 1:3
        data(a)[:, 1, k] .= [1, 2, 3]
        data(a)[:, 2, k] .= [1, 2, 3]
        data(a)[:, 3, k] .= [1, 2, 3]
    end

    return dx_a[2, 2, 2] == 1 
end

function test_y_derivative(a)
    dy_a = ∂y(a)

    for k in 1:3
        data(a)[1, :, k] .= [1, 3, 5]
        data(a)[2, :, k] .= [1, 3, 5]
        data(a)[3, :, k] .= [1, 3, 5]
    end

    return dy_a[2, 2, 2] == 2 
end

function test_z_derivative(a)
    dz_a = ∂z(a)

    for k in 1:3
        data(a)[1, k, :] .= [1, 4, 7]
        data(a)[2, k, :] .= [1, 4, 7]
        data(a)[3, k, :] .= [1, 4, 7]
    end

    return dz_a[2, 2, 2] == -3 
end

function test_x_derivative_cell(FT, arch)
    grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))
    a = Field(Cell, Cell, Cell, arch, grid)
    dx_a = ∂x(a)

    for k in 1:3
        data(a)[:, 1, k] .= [1, 4, 4]
        data(a)[:, 2, k] .= [1, 4, 4]
        data(a)[:, 3, k] .= [1, 4, 4]
    end

    return dx_a[2, 2, 2] == 3 
end

function test_binary_operation_derivative_mix(FT, arch)
    grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))
    a = Field(Cell, Cell, Cell, arch, grid)
    b = Field(Cell, Cell, Cell, arch, grid)

    a∇b = b * ∂x(a)

    data(b) .= 2

    for k in 1:3
        data(a)[:, 1, k] .= [1, 4, 4]
        data(a)[:, 2, k] .= [1, 4, 4]
        data(a)[:, 3, k] .= [1, 4, 4]
    end

    return a∇b[2, 2, 2] == 6
end

@testset "Abstract operations" begin
    println("Testing abstract operations...")

    @testset "Simple binary operations" begin
        println("  Testing simple binary operations...")
        for FT in float_types
            num1 = FT(π)
            num2 = FT(42)
            grid = RegularCartesianGrid(FT, (3, 1, 3), (3, 1, 3))

            for arch in archs
                u, v, w = Oceananigans.VelocityFields(arch, grid)
                T, S = Oceananigans.TracerFields(arch, grid, (:T, :S))

                for op in (+, *, -, /)
                    @test test_simple_binary_operation(op, u, v, num1, num2)
                    @test test_simple_binary_operation(op, u, w, num1, num2)
                    @test test_simple_binary_operation(op, u, T, num1, num2)
                    @test test_simple_binary_operation(op, T, S, num1, num2)
                end
                @test test_three_field_addition(u, v, w, num1, num2)
            end
        end
    end

    @testset "Derivatives" begin
        println("  Testing derivatives...")
        for FT in float_types
            grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))

            for arch in archs
                u, v, w = Oceananigans.VelocityFields(arch, grid)
                T, S = Oceananigans.TracerFields(arch, grid, (:T, :S))
                for a in (u, v, w, T)
                    @test test_x_derivative(a)
                    @test test_y_derivative(a)
                    @test test_z_derivative(a)
                end
                @test test_x_derivative_cell(FT, arch)
            end
        end
    end

    @testset "Combined binary operations and derivatives" begin
        println("  Testing combined binary operations and derivatives...")
        for FT in float_types
            for arch in archs
                @test test_binary_operation_derivative_mix(FT, arch)
            end
        end
    end
end
