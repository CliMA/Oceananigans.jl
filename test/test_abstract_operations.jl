using Oceananigans.AbstractOperations: UnaryOperation, Derivative, BinaryOperation, MultiaryOperation

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

    return dz_a[2, 2, 2] == 3
end

function x_derivative_cell(FT, arch)
    grid = RegularCartesianGrid(FT, size=(3, 3, 3), extent=(3, 3, 3))
    a = Field(Cell, Cell, Cell, arch, grid, nothing)
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
    T, S = model.tracers
    S.data.parent .= π

    @compute ∂S = ComputedField(∂(S))
    result = Array(interior(∂S))

    return all(result .≈ zero(eltype(model.grid)))
end

function compute_unary(unary, model)
    set!(model; S=π)
    T, S = model.tracers

    @compute uS = ComputedField(unary(S), data=model.pressures.pHY′.data)
    result = Array(interior(uS))

    return all(result .≈ unary(eltype(model.grid)(π)))
end

function compute_plus(model)
    set!(model; S=π, T=42)
    T, S = model.tracers

    @compute ST = ComputedField(S + T, data=model.pressures.pHY′.data)

    result = Array(interior(ST))

    return all(result .≈ eltype(model.grid)(π+42))
end

function compute_many_plus(model)
    set!(model; u=2, S=π, T=42)
    T, S = model.tracers
    u, v, w = model.velocities

    @compute uTS = ComputedField(@at((Cell, Cell, Cell), u + T + S))
    result = Array(interior(uTS))

    return all(result .≈ eltype(model.grid)(2+π+42))
end

function compute_minus(model)
    set!(model; S=π, T=42)
    T, S = model.tracers

    @compute ST = ComputedField(S - T, data=model.pressures.pHY′.data)
    result = Array(interior(ST))

    return all(result .≈ eltype(model.grid)(π-42))
end

function compute_times(model)
    set!(model; S=π, T=42)
    T, S = model.tracers

    @compute ST = ComputedField(S * T, data=model.pressures.pHY′.data)
    result = Array(interior(ST))

    return all(result .≈ eltype(model.grid)(π*42))
end

function compute_kinetic_energy(model)
    u, v, w = model.velocities
    set!(u, 1)
    set!(v, 2)
    set!(w, 3)

    kinetic_energy_operation = @at (Cell, Cell, Cell) (u^2 + v^2 + w^2) / 2
    @compute kinetic_energy = ComputedField(kinetic_energy_operation, data=model.pressures.pHY′.data)
    result = Array(interior(kinetic_energy))

    return all(result .≈ 7)
end

function horizontal_average_of_plus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = AveragedField(S + T, dims=(1, 2))

    zC = znodes(Cell, model.grid)
    correct_profile = @. sin(π * zC) + 42 * zC

    return all(ST[1, 1, 1:Nz] .≈ correct_profile)
end

function zonal_average_of_plus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π*z) * sin(π*y)
    T₀(x, y, z) = 42*z + y^2
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = AveragedField(S + T, dims=1)

    yC = ynodes(Cell, model.grid, reshape=true)
    zC = znodes(Cell, model.grid, reshape=true)
    correct_slice = @. sin(π * zC) * sin(π * yC) + 42*zC + yC^2

    return all(ST[1, 1:Ny, 1:Nz] .≈ correct_slice[1, :, :])
end

function volume_average_of_times(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = 1 + sin(2π*x)
    T₀(x, y, z) = y
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = AveragedField(S * T, dims=(1, 2, 3))
    return all(ST[1, 1, 1] .≈ 0.5)
end

function horizontal_average_of_minus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = AveragedField(S - T, dims=(1, 2))

    zC = znodes(Cell, model.grid)
    correct_profile = @. sin(π * zC) - 42 * zC

    return all(ST[1, 1, 1:Nz] .≈ correct_profile)
end

function horizontal_average_of_times(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = AveragedField(S * T, dims=(1, 2))

    zC = znodes(Cell, model.grid)
    correct_profile = @. sin(π * zC) * 42 * zC

    return all(ST[1, 1, 1:Nz] .≈ correct_profile)
end

function multiplication_and_derivative_ccf(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    w₀(x, y, z) = sin(π * z)
    T₀(x, y, z) = 42 * z
    set!(model; w=w₀, T=T₀)

    w = model.velocities.w
    T = model.tracers.T

    @compute wT = AveragedField(w * ∂z(T), dims=(1, 2))

    zF = znodes(Face, model.grid)
    correct_profile = @. 42 * sin(π * zF)

    # Omit both halos and boundary points
    return all(wT[1, 1, 2:Nz] .≈ correct_profile[2:Nz])
end

const C = Cell
const F = Face

function multiplication_and_derivative_ccc(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    w₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; w=w₀, T=T₀)

    w = model.velocities.w
    T = model.tracers.T

    wT_ccc = @at (C, C, C) w * ∂z(T)
    @compute wT_ccc_avg = AveragedField(wT_ccc, dims=(1, 2))

    zF = znodes(Face, model.grid)
    sinusoid = sin.(π * zF)
    interped_sin = [(sinusoid[k] + sinusoid[k+1]) / 2 for k in 1:model.grid.Nz]
    correct_profile = interped_sin .* 42

    # Omit boundary-adjacent points from comparison
    return all(wT_ccc_avg[1, 1, 2:Nz-1] .≈ correct_profile[2:Nz-1])
end

function computation_including_boundaries(FT, arch)
    topo = (Bounded, Bounded, Bounded)
    grid = RegularCartesianGrid(FT, topology=topo, size=(13, 17, 19), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, float_type=FT, grid=grid)

    u, v, w = model.velocities
    @. u.data = 1 + rand()
    @. v.data = 2 + rand()
    @. w.data = 3 + rand()

    op = @at (Face, Face, Face) u * v * w
    @compute uvw = ComputedField(op)

    return all(interior(uvw) .!= 0)
end

@testset "Abstract operations" begin
    @info "Testing abstract operations..."

    for FT in float_types
        arch = CPU()
        grid = RegularCartesianGrid(FT, size=(3, 3, 3), extent=(3, 3, 3))
        u, v, w = VelocityFields(arch, grid)
        c = Field(Cell, Cell, Cell, arch, grid, nothing)

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

        @testset "Multiary operations [$FT]" begin
            generic_function(x, y, z) = x + y + z
            for (ψ, ϕ, σ) in ((u, v, w), (u, v, c), (u, v, generic_function))
                for op_symbol in Oceananigans.AbstractOperations.multiary_operators
                    op = eval(op_symbol)
                    @test typeof(op((Cell, Cell, Cell), ψ, ϕ, σ)[2, 2, 2]) <: Number
                end
            end
        end
    end

    @testset "Fidelity of simple binary operations" begin
        arch = CPU()
        @info "  Testing simple binary operations..."
        for FT in float_types
            num1 = FT(π)
            num2 = FT(42)
            grid = RegularCartesianGrid(FT, size=(3, 3, 3), extent=(3, 3, 3))

            u, v, w = VelocityFields(arch, grid)
            T, S = TracerFields(arch, grid, (:T, :S))

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
        @info "  Testing derivatives..."
        for FT in float_types
            grid = RegularCartesianGrid(FT, size=(3, 3, 3), extent=(3, 3, 3),
                                        topology=(Periodic, Periodic, Periodic))

            u, v, w = VelocityFields(arch, grid)
            T, S = TracerFields(arch, grid, (:T, :S))
            for a in (u, v, w, T)
                @test x_derivative(a)
                @test y_derivative(a)
                @test z_derivative(a)
            end
            @test x_derivative_cell(FT, arch)
        end
    end

    @testset "Combined binary operations and derivatives" begin
        @info "  Testing combined binary operations and derivatives..."
        arch = CPU()
        Nx = 3 # Δx=1, xC = 0.5, 1.5, 2.5
        for FT in float_types
            grid = RegularCartesianGrid(FT, size=(Nx, Nx, Nx), extent=(Nx, Nx, Nx))
            a, b = (Field(Cell, Cell, Cell, arch, grid, nothing) for i in 1:2)

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
        @testset "AbstractOperations and ComputedFields [$(typeof(arch))]" begin
            @info "  Testing combined binary operations and derivatives..."
            for FT in float_types
                model = IncompressibleModel(
                    architecture = arch,
                      float_type = FT,
                            grid = RegularCartesianGrid(FT, size=(4, 4, 4), extent=(1, 1, 1),
                                                        topology=(Periodic, Periodic, Bounded))
                )

                @testset "Construction of abstract operations [$FT, $(typeof(arch))]" begin
                    @info "    Testing construction of abstract operations [$FT, $(typeof(arch))]..."

                    u, v, w = model.velocities
                    T, S = model.tracers

                    for ϕ in (u, v, w, T, S)
                        for op in (sin, cos, sqrt, exp, tanh)
                            @test op(ϕ) isa UnaryOperation
                        end

                        for ∂ in (∂x, ∂y, ∂z)
                            @test ∂(ϕ) isa Derivative
                        end

                        @test u ^ 2 isa BinaryOperation
                        @test u * 2 isa BinaryOperation
                        @test u + 2 isa BinaryOperation
                        @test u - 2 isa BinaryOperation
                        @test u / 2 isa BinaryOperation

                        for ψ in (u, v, w, T, S)
                            @test ψ ^ ϕ isa BinaryOperation
                            @test ψ * ϕ isa BinaryOperation
                            @test ψ + ϕ isa BinaryOperation
                            @test ψ - ϕ isa BinaryOperation
                            @test ψ / ϕ isa BinaryOperation

                            for χ in (u, v, w, T, S)
                                @test ψ * ϕ * χ isa MultiaryOperation
                                @test ψ + ϕ + χ isa MultiaryOperation
                            end
                        end
                    end
                end

                @info "    Testing computation of abstract operations [$FT, $(typeof(arch))]..."

                @testset "Derivative computations [$FT, $(typeof(arch))]" begin
                    @info "      Testing compute! derivatives..."
                    @test compute_derivative(model, ∂x)
                    @test compute_derivative(model, ∂y)
                    @test compute_derivative(model, ∂z)
                end

                @testset "Unary computations [$FT, $(typeof(arch))]" begin
                    @info "      Testing compute! unary operations..."
                    for unary in Oceananigans.AbstractOperations.unary_operators
                        @test compute_unary(eval(unary), model)
                    end
                end

                @testset "Binary computations [$FT, $(typeof(arch))]" begin
                    @info "      Testing compute! binary operations..."
                    @test compute_plus(model)
                    @test compute_minus(model)
                    @test compute_times(model)
                end

                @testset "Multiary computations [$FT, $(typeof(arch))]" begin
                    @info "      Testing compute! multiary operations..."
                    @test compute_many_plus(model)

                    @info "      Testing compute! kinetic energy..."
                    @test compute_kinetic_energy(model)
                end

                @testset "Horizontal averages of operations [$FT, $(typeof(arch))]" begin
                    @info "      Testing horizontal averges..."
                    @test horizontal_average_of_plus(model)
                    @test horizontal_average_of_minus(model)
                    @test horizontal_average_of_times(model)

                    @test multiplication_and_derivative_ccf(model)
                    @test multiplication_and_derivative_ccc(model)
                end

                @testset "Zonal averages of operations [$FT, $(typeof(arch))]" begin
                    @info "      Testing zonal averges..."
                    @test zonal_average_of_plus(model)
                end

                @testset "Volume averages of operations [$FT, $(typeof(arch))]" begin
                    @info "      Testing volume averges..."
                    @test volume_average_of_times(model)
                end

                @testset "Faces along Bounded dimensions" begin
                    @info "      Testing compute! on faces along bounded dimensions..."
                    @test computation_including_boundaries(FT, arch)
                end
            end
        end
    end
end
