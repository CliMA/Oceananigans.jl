function simple_binary_operation(op, a, b, num1, num2)
    a_b = op(a, b)
    interior(a) .= num1
    interior(b) .= num2
    return a_b[2, 2, 2] == op(num1, num2)
end

function three_field_addition(a, b, c, num1, num2)
    a_b_c = a + b + c
    interior(a) .= num1
    interior(b) .= num2
    interior(c) .= num2
    return a_b_c[2, 2, 2] == num1 + num2 + num2
end

function x_derivative(a)
    dx_a = ∂x(a)

    for k in 1:3
        interior(a)[:, 1, k] .= [1, 2, 3]
        interior(a)[:, 2, k] .= [1, 2, 3]
        interior(a)[:, 3, k] .= [1, 2, 3]
    end

    return dx_a[2, 2, 2] == 1 
end

function y_derivative(a)
    dy_a = ∂y(a)

    for k in 1:3
        interior(a)[1, :, k] .= [1, 3, 5]
        interior(a)[2, :, k] .= [1, 3, 5]
        interior(a)[3, :, k] .= [1, 3, 5]
    end

    return dy_a[2, 2, 2] == 2 
end

function z_derivative(a)
    dz_a = ∂z(a)

    for k in 1:3
        interior(a)[1, k, :] .= [1, 4, 7]
        interior(a)[2, k, :] .= [1, 4, 7]
        interior(a)[3, k, :] .= [1, 4, 7]
    end

    return dz_a[2, 2, 2] == -3 
end

function x_derivative_cell(FT, arch)
    grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))
    a = Field(Cell, Cell, Cell, arch, grid)
    dx_a = ∂x(a)

    for k in 1:3
        interior(a)[:, 1, k] .= [1, 4, 4]
        interior(a)[:, 2, k] .= [1, 4, 4]
        interior(a)[:, 3, k] .= [1, 4, 4]
    end

    return dx_a[2, 2, 2] == 3 
end

function binary_operation_derivative_mix(FT, arch)
    grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))
    a = Field(Cell, Cell, Cell, arch, grid)
    b = Field(Cell, Cell, Cell, arch, grid)

    a∇b = @at (Face, Cell, Cell) b * ∂x(a)

    interior(b) .= 2

    for k in 1:3
        interior(a)[:, 1, k] .= [1, 4, 4]
        interior(a)[:, 2, k] .= [1, 4, 4]
        interior(a)[:, 3, k] .= [1, 4, 4]
    end

    return a∇b[2, 2, 2] == 6
end

function horizontal_average_of_plus(model)
    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    ST = HorizontalAverage(S + T, model)
    computed_profile = ST(model)
    correct_profile = @. sin(π*model.grid.zC) + 42 * model.grid.zC

    return all(computed_profile[:][2:end-1] .≈ correct_profile)
end

function horizontal_average_of_minus(model)
    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    ST = HorizontalAverage(S - T, model)
    computed_profile = ST(model)

    correct_profile = @. sin(π*model.grid.zC) - 42 * model.grid.zC

    return all(computed_profile[:][2:end-1] .≈ correct_profile)
end

function horizontal_average_of_times(model)
    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    ST = HorizontalAverage(S * T, model)
    computed_profile = ST(model)
    correct_profile = @. sin(π*model.grid.zC) * 42 * model.grid.zC

    return all(computed_profile[:][2:end-1] .≈ correct_profile)
end

function multiplication_and_derivative_ccf(model)
    w₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; w=w₀, T=T₀)

    w = model.velocities.w
    T = model.tracers.T

    wT = HorizontalAverage(w * ∂z(T), model)
    computed_profile = wT(model)
    correct_profile = @. sin(π*model.grid.zF) * 42

    return all(computed_profile[:][2:end-1] .≈ correct_profile[1:end-1])
end

const C = Cell
const F = Face

function multiplication_and_derivative_ccc(model)
    w₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; w=w₀, T=T₀)

    w = model.velocities.w
    T = model.tracers.T

    wT_ccc = @at (C, C, C) w * ∂z(T)
    wT_ccc_avg = HorizontalAverage(wT_ccc, model)
    computed_profile_ccc = wT_ccc_avg(model)

    sinusoid = sin.(π*model.grid.zF)
    interped_sin = [(sinusoid[k] + sinusoid[k+1]) / 2 for k in 1:model.grid.Nz]
    correct_profile = interped_sin .* 42

    return all(computed_profile_ccc[:][3:end-2] .≈ correct_profile[2:end-1])
end

@testset "Abstract operations" begin
    println("Testing abstract operations...")

    @testset "Simple binary operations" begin
        println("  Testing simple binary operations...")
        for FT in float_types
            num1 = FT(π)
            num2 = FT(42)
            grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))

            for arch in archs
                u, v, w = Oceananigans.VelocityFields(arch, grid)
                T, S = Oceananigans.TracerFields(arch, grid, (:T, :S))

                for op in (+, *, -, /)
                    @test simple_binary_operation(op, u, v, num1, num2)
                    @test simple_binary_operation(op, u, w, num1, num2)
                    @test simple_binary_operation(op, u, T, num1, num2)
                    @test simple_binary_operation(op, T, S, num1, num2)
                end
                @test three_field_addition(u, v, w, num1, num2)
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
                    @test x_derivative(a)
                    @test y_derivative(a)
                    @test z_derivative(a)
                end
                @test x_derivative_cell(FT, arch)
            end
        end
    end

    @testset "Combined binary operations and derivatives" begin
        println("  Testing combined binary operations and derivatives...")
        for FT in float_types
            for arch in archs
                @test binary_operation_derivative_mix(FT, arch)

                model = BasicModel(N=(16, 16, 16), L=(1, 1, 1), architecture=arch, float_type=FT)

                @test horizontal_average_of_plus(model)
                @test horizontal_average_of_minus(model)
                @test horizontal_average_of_times(model)

                @test multiplication_and_derivative_ccf(model)
                @test multiplication_and_derivative_ccc(model)
            end
        end
    end
end
