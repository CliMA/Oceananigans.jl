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

function times_x_derivative(a, b, location, i, j, k, answer)
    a∇b = @at location b * ∂x(a)
    return a∇b[i, j, k] == answer
end

function compute_derivative(model, ∂)
    #set!(model; S=π)
    T, S = model.tracers
    S.data.parent .= π

    computation = Computation(∂(S), model.pressures.pHY′)
    compute!(computation)
    result = Array(interior(computation.result))

    return all(result .≈ zero(eltype(model.grid)))
end

function compute_plus(model)
    set!(model; S=π, T=42)
    T, S = model.tracers

    computation = Computation(S + T, model.pressures.pHY′)
    compute!(computation)
    result = Array(interior(computation.result))

    return all(result .≈ eltype(model.grid)(π+42))
end

function compute_minus(model)
    set!(model; S=π, T=42)
    T, S = model.tracers

    computation = Computation(S - T, model.pressures.pHY′)
    compute!(computation)
    result = Array(interior(computation.result))

    return all(result .≈ eltype(model.grid)(π-42))
end

function compute_times(model)
    set!(model; S=π, T=42)
    T, S = model.tracers

    computation = Computation(S * T, model.pressures.pHY′)
    compute!(computation)
    result = Array(interior(computation.result))

    return all(result .≈ eltype(model.grid)(π*42))
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

    for FT in float_types
        arch = CPU()
        grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))
        u, v, w = Oceananigans.VelocityFields(arch, grid)
        c = Field(Cell, Cell, Cell, arch, grid)

        @testset "Unary operations and derivatives [$FT]" begin
            for ψ in (u, v, w, c)
                for op_symbol in Oceananigans.AbstractOperations.unary_operators
                    op = eval(op_symbol)
                    @test typeof(op(ψ)[2, 2, 2]) <: Number
                end

                for d_symbol in Oceananigans.AbstractOperations.derivative_operators
                    d = eval(d_symbol)
                    @test typeof(d(ψ)[2, 2, 2]) <: Number
                end
            end
        end

        @testset "Binary operations [$FT]" begin
            generic_function(x, y, z) = x + y + z
            for (ψ, ϕ) in ((u, v), (u, w), (v, w), (u, c), (generic_function, c), (u, generic_function))
                for op_symbol in Oceananigans.AbstractOperations.binary_operators
                    op = eval(op_symbol)
                    @test typeof(op(ψ, ϕ)[2, 2, 2]) <: Number
                end
            end
        end

        @testset "Polynary operations [$FT]" begin
            for (ψ, ϕ, σ) in ((u, v, w), (u, v, c))
                for op_symbol in Oceananigans.AbstractOperations.polynary_operators
                    op = eval(op_symbol)
                    @test typeof(op(ψ, ϕ, σ)[2, 2, 2]) <: Number
                end
            end
        end
    end

    @testset "Fidelity of simple binary operations" begin
        arch = CPU()
        println("  Testing simple binary operations...")
        for FT in float_types
            num1 = FT(π)
            num2 = FT(42)
            grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))

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

    @testset "Derivatives" begin
        arch = CPU()
        println("  Testing derivatives...")
        for FT in float_types
            grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))

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

    @testset "Combined binary operations and derivatives" begin
        println("  Testing combined binary operations and derivatives...")
        arch = CPU()
        Nx = 3 # Δx=1, xC = 0.5, 1.5, 2.5
        for FT in float_types
            grid = RegularCartesianGrid(FT, (Nx, Nx, Nx), (Nx, Nx, Nx))
            a, b = Tuple(Field(Cell, Cell, Cell, arch, grid) for i in 1:2)
    
            set!(b, 2)
            set!(a, (x, y, z) -> x < 2 ? 3x : 6)
    
            #                            0   0.5   1   1.5   2   2.5   3
            # x -▶                  ∘ ~~~|--- * ---|--- * ---|--- * ---|~~~ ∘
            #        i Face:    0        1         2        3          4
            #        i Cell:        0         1         2         3         4

            #              a = [    0,       1.5,      4.5,       6,        0    ]
            #              b = [    0,        2,        2,        2,        0    ]
            #          ∂x(a) = [        1.5,       3,       1.5,      -6         ] 

            # x -▶                  ∘ ~~~|--- * ---|--- * ---|--- * ---|~~~ ∘
            #        i Face:    0        1         2         3         4
            #        i Cell:        0         1         2         3         4

            # ccc: b * ∂x(a) = [             4.5,      4.5      -4.5,            ] 
            # fcc: b * ∂x(a) = [         3,        6,        3,       -6         ] 


            @test times_x_derivative(a, b, (C, C, C), 1, 2, 2, 4.5)
            @test times_x_derivative(a, b, (F, C, C), 1, 2, 2, 3)

            @test times_x_derivative(a, b, (C, C, C), 2, 2, 2, 4.5)
            @test times_x_derivative(a, b, (F, C, C), 2, 2, 2, 6)

            @test times_x_derivative(a, b, (C, C, C), 3, 2, 2, -4.5)
            @test times_x_derivative(a, b, (F, C, C), 3, 2, 2, 3)
        end
    end

    for arch in archs
        @testset "Computations [$(typeof(arch))]" begin
            println("  Testing combined binary operations and derivatives...")
            for FT in float_types
                println("    Testing computation of abstract operations [$(typeof(arch))]...")
                model = BasicModel(N=(16, 16, 16), L=(1, 1, 1), architecture=arch, float_type=FT)

                println("      Testing compute!...")
                @test compute_derivative(model, ∂x)
                @test compute_derivative(model, ∂y)
                @test compute_derivative(model, ∂z)

                @test compute_plus(model)
                @test compute_minus(model)
                @test compute_times(model)

                println("      Testing horizontal averges...")
                @test horizontal_average_of_plus(model)
                @test horizontal_average_of_minus(model)
                @test horizontal_average_of_times(model)

                @test multiplication_and_derivative_ccf(model)
                @test multiplication_and_derivative_ccc(model)
            end
        end
    end
end
