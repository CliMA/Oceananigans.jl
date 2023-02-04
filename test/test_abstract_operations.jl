include("dependencies_for_runtests.jl")

using Oceananigans.Operators: ℑxyᶜᶠᵃ, ℑxyᶠᶜᵃ
using Oceananigans.Fields: ZeroField, ConstantField, compute_at!, indices
using Oceananigans.BuoyancyModels: BuoyancyField

function simple_binary_operation(op, a, b, num1, num2)
    a_b = op(a, b)
    interior(a) .= num1
    interior(b) .= num2
    return CUDA.@allowscalar a_b[2, 2, 2] == op(num1, num2)
end

function three_field_addition(a, b, c, num1, num2)
    a_b_c = a + b + c
    interior(a) .= num1
    interior(b) .= num2
    interior(c) .= num2
    return CUDA.@allowscalar a_b_c[2, 2, 2] == num1 + num2 + num2
end

function x_derivative(a)
    dx_a = ∂x(a)

    arch = architecture(a)
    one_two_three = arch_array(arch, [1, 2, 3])

    for k in 1:3
        interior(a)[:, 1, k] .= one_two_three
        interior(a)[:, 2, k] .= one_two_three
        interior(a)[:, 3, k] .= one_two_three
    end

    return CUDA.@allowscalar dx_a[2, 2, 2] == 1
end

function y_derivative(a)
    dy_a = ∂y(a)

    arch = architecture(a)
    one_three_five = arch_array(arch, [1, 3, 5])

    for k in 1:3
        interior(a)[1, :, k] .= one_three_five
        interior(a)[2, :, k] .= one_three_five
        interior(a)[3, :, k] .= one_three_five
    end

    return CUDA.@allowscalar dy_a[2, 2, 2] == 2
end

function z_derivative(a)
    dz_a = ∂z(a)

    arch = architecture(a)
    one_four_seven = arch_array(arch, [1, 4, 7])

    for k in 1:3
        interior(a)[1, k, :] .= one_four_seven
        interior(a)[2, k, :] .= one_four_seven
        interior(a)[3, k, :] .= one_four_seven
    end

    return CUDA.@allowscalar dz_a[2, 2, 2] == 3
end

function x_derivative_cell(arch)
    grid = RectilinearGrid(arch, size=(3, 3, 3), extent=(3, 3, 3))
    a = Field{Center, Center, Center}(grid)
    dx_a = ∂x(a)

    one_four_four = arch_array(arch, [1, 4, 4])

    for k in 1:3
        interior(a)[:, 1, k] .= one_four_four 
        interior(a)[:, 2, k] .= one_four_four 
        interior(a)[:, 3, k] .= one_four_four 
    end

    return CUDA.@allowscalar dx_a[2, 2, 2] == 3
end

function times_x_derivative(a, b, location, i, j, k, answer)
    a∇b = @at location b * ∂x(a)
    
    return CUDA.@allowscalar a∇b[i, j, k] == answer
end

for arch in archs
    @testset "Abstract operations [$(typeof(arch))]" begin
        @info "Testing abstract operations [$(typeof(arch))]..."

        grid = RectilinearGrid(arch, size=(3, 3, 3), extent=(3, 3, 3))
        u, v, w = VelocityFields(grid)
        c = Field{Center, Center, Center}(grid)

        @testset "Unary operations and derivatives [$(typeof(arch))]" begin
            for ψ in (u, v, w, c)
                for op in (sqrt, sin, cos, exp, tanh)
                    @test CUDA.@allowscalar typeof(op(ψ)[2, 2, 2]) <: Number
                end

                for d_symbol in Oceananigans.AbstractOperations.derivative_operators
                    d = eval(d_symbol)
                    @test CUDA.@allowscalar typeof(d(ψ)[2, 2, 2]) <: Number
                end
            end
        end

        @testset "Binary operations [$(typeof(arch))]" begin
            generic_function(x, y, z) = x + y + z
            for (ψ, ϕ) in ((u, v), (u, w), (v, w), (u, c), (generic_function, c), (u, generic_function))
                for op_symbol in Oceananigans.AbstractOperations.binary_operators
                    op = eval(op_symbol)
                    @test CUDA.@allowscalar typeof(op(ψ, ϕ)[2, 2, 2]) <: Number
                end
            end

            @test ZeroField() + u == u
            @test u + ZeroField() == u
            @test ZeroField() - u == -u
            @test u - ZeroField() == u
            @test ZeroField() * u == ZeroField()
            @test u * ZeroField() == ZeroField()
            @test ZeroField() / u == ZeroField()
            @test u / ZeroField() == ConstantField(Inf)

            @test ZeroField() + 1 == ConstantField(1)
            @test 1 + ZeroField() == ConstantField(1)
            @test ZeroField() - 1 == ConstantField(-1)
            @test 1 - ZeroField() == ConstantField(1)
            @test ZeroField() * 1 == ZeroField()
            @test 1 * ZeroField() == ZeroField()
            @test ZeroField() / 1 == ZeroField()
            @test 1 / ZeroField() == ConstantField(Inf)

            @test ConstantField(1) + u == 1 + u
            @test ConstantField(1) - u == 1 - u
            @test ConstantField(1) * u == 1 * u
            @test u / ConstantField(1) == u / 1

            @test ConstantField(1) + 1 == ConstantField(2)
            @test ConstantField(1) - 1 == ConstantField(0)
            @test ConstantField(1) * 2 == ConstantField(2)
            @test ConstantField(1) / 2 == ConstantField(1/2)
        end

        @testset "Multiary operations [$(typeof(arch))]" begin
            generic_function(x, y, z) = x + y + z
            for (ψ, ϕ, σ) in ((u, v, w), (u, v, c), (u, v, generic_function))
                for op_symbol in Oceananigans.AbstractOperations.multiary_operators
                    op = eval(op_symbol)
                    @test CUDA.@allowscalar typeof(op((Center, Center, Center), ψ, ϕ, σ)[2, 2, 2]) <: Number
                end
            end
        end

        @testset "KernelFunctionOperations [$(typeof(arch))]" begin
            trivial_kernel_function(i, j, k, grid) = 1
            op = KernelFunctionOperation{Center, Center, Center}(trivial_kernel_function, grid)
            @test op isa KernelFunctionOperation

            less_trivial_kernel_function(i, j, k, grid, u, v) = @inbounds u[i, j, k] * ℑxyᶠᶜᵃ(i, j, k, grid, v)
            op = KernelFunctionOperation{Face, Center, Center}(less_trivial_kernel_function, grid, computed_dependencies=(u, v))
            @test op isa KernelFunctionOperation

            still_fairly_trivial_kernel_function(i, j, k, grid, u, v, μ) = @inbounds μ * ℑxyᶜᶠᵃ(i, j, k, grid, u) * v[i, j, k]
            op = KernelFunctionOperation{Center, Face, Center}(still_fairly_trivial_kernel_function, grid,
                                                               computed_dependencies=(u, v), parameters=0.1)
            @test op isa KernelFunctionOperation
        end

        @testset "Fidelity of simple binary operations" begin
            @info "  Testing simple binary operations..."
            num1 = Float64(π)
            num2 = Float64(42)
            grid = RectilinearGrid(arch, size=(3, 3, 3), extent=(3, 3, 3))

            u, v, w = VelocityFields(grid)
            T, S = TracerFields((:T, :S), grid)

            for op in (+, *, -, /)
                @test simple_binary_operation(op, u, v, num1, num2)
                @test simple_binary_operation(op, u, w, num1, num2)
                @test simple_binary_operation(op, u, T, num1, num2)
                @test simple_binary_operation(op, T, S, num1, num2)
            end
            @test three_field_addition(u, v, w, num1, num2)
        end

        @testset "Derivatives" begin
            @info "  Testing derivatives..."
            grid = RectilinearGrid(arch, size=(3, 3, 3), extent=(3, 3, 3),
                                   topology=(Periodic, Periodic, Periodic))

            u, v, w = VelocityFields(grid)
            T, S = TracerFields((:T, :S), grid)
            for a in (u, v, w, T)
                @test x_derivative(a)
                @test y_derivative(a)
                @test z_derivative(a)
            end

            @test x_derivative_cell(arch)
        end

        @testset "Combined binary operations and derivatives" begin
            @info "  Testing combined binary operations and derivatives..."
            Nx = 3 # Δx=1, xC = 0.5, 1.5, 2.5
            grid = RectilinearGrid(arch, size=(Nx, Nx, Nx), extent=(Nx, Nx, Nx))
            a, b = (Field{Center, Center, Center}(grid) for i in 1:2)

            set!(b, 2)
            set!(a, (x, y, z) -> x < 2 ? 3x : 6)

            #                            0   0.5   1   1.5   2   2.5   3
            # x -▶                  ∘ ~~~|--- * ---|--- * ---|--- * ---|~~~ ∘
            #        i Face:    0        1         2        3          4
            #        i Center:        0         1         2         3         4

            #              a = [    0,       1.5,      4.5,       6,        0    ]
            #              b = [    0,        2,        2,        2,        0    ]
            #          ∂x(a) = [        1.5,       3,       1.5,      -6         ]

            # x -▶                  ∘ ~~~|--- * ---|--- * ---|--- * ---|~~~ ∘
            #        i Face:    0        1         2         3         4
            #        i Center:        0         1         2         3         4

            # ccc: b * ∂x(a) = [             4.5,      4.5      -4.5,            ]
            # fcc: b * ∂x(a) = [        1.5,       6,        3,       -6         ]

            C = Center
            F = Face

            @test times_x_derivative(a, b, (C, C, C), 1, 2, 2, 4.5)
            @test times_x_derivative(a, b, (C, C, C), 2, 2, 2, 4.5)
            @test times_x_derivative(a, b, (C, C, C), 3, 2, 2, -4.5)

            @test times_x_derivative(a, b, (F, C, C), 1, 2, 2, 1.5)
            @test times_x_derivative(a, b, (F, C, C), 2, 2, 2, 6)
            @test times_x_derivative(a, b, (F, C, C), 3, 2, 2, 3)
            @test times_x_derivative(a, b, (F, C, C), 4, 2, 2, -6)
        end

        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Bounded))

        buoyancy = SeawaterBuoyancy(gravitational_acceleration = 1,
                                    equation_of_state = LinearEquationOfState(thermal_expansion=1, haline_contraction=1))

        model = NonhydrostaticModel(; grid, buoyancy, tracers = (:T, :S))

        @testset "Construction of abstract operations [$(typeof(arch))]" begin
            @info "    Testing construction of abstract operations [$(typeof(arch))]..."

            u, v, w, T, S = fields(model)

            for ϕ in (u, v, w, T)
                for op in (sin, cos, sqrt, exp, tanh)
                    @test op(ϕ) isa UnaryOperation
                end

                for ∂ in (∂x, ∂y, ∂z)
                    @test ∂(ϕ) isa Derivative
                end

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

                for metric in (AbstractOperations.Δx,
                               AbstractOperations.Δy,
                               AbstractOperations.Δz,
                               AbstractOperations.Ax,
                               AbstractOperations.Ay,
                               AbstractOperations.Az,
                               AbstractOperations.volume)

                    @test location(metric * ϕ) == location(ϕ)
                end
            end

            @test u ^ 2 isa BinaryOperation
            @test u * 2 isa BinaryOperation
            @test u + 2 isa BinaryOperation
            @test u - 2 isa BinaryOperation
            @test u / 2 isa BinaryOperation
        end

        @testset "BinaryOperations with GridMetricOperation [$(typeof(arch))]" begin
            lat_lon_grid = LatitudeLongitudeGrid(arch, size=(1, 1, 1), longitude=(0, 1), latitude=(0, 1), z=(0, 1))
            rectilinear_grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(2, 3, 4))

            for LX in (Center, Face)
                for LY in (Center, Face)
                    for LZ in (Center, Face)
                        loc = (LX, LY, LZ)
                        f = Field(loc, rectilinear_grid)
                        f .= 1
                        
                        CUDA.@allowscalar begin
                            # Δx, Δy, Δz = 2, 3, 4
                            # Ax, Ay, Az = 12, 8, 6
                            # volume = 24
                            op = f * AbstractOperations.Δx;     @test op[1, 1, 1] == 2
                            op = f * AbstractOperations.Δy;     @test op[1, 1, 1] == 3
                            op = f * AbstractOperations.Δz;     @test op[1, 1, 1] == 4
                            op = f * AbstractOperations.Ax;     @test op[1, 1, 1] == 12
                            op = f * AbstractOperations.Ay;     @test op[1, 1, 1] == 8
                            op = f * AbstractOperations.Az;     @test op[1, 1, 1] == 6
                            op = f * AbstractOperations.volume; @test op[1, 1, 1] == 24

                            # Here we are really testing that `op` can be called
                            f = Field(loc, lat_lon_grid)
                            op = f * AbstractOperations.Δx;     @test op[1, 1, 1] == 0
                            op = f * AbstractOperations.Δy;     @test op[1, 1, 1] == 0
                            op = f * AbstractOperations.Δz;     @test op[1, 1, 1] == 0
                            op = f * AbstractOperations.Ax;     @test op[1, 1, 1] == 0
                            op = f * AbstractOperations.Ay;     @test op[1, 1, 1] == 0
                            op = f * AbstractOperations.Az;     @test op[1, 1, 1] == 0
                            op = f * AbstractOperations.volume; @test op[1, 1, 1] == 0
                        end
                    end
                end
            end
        end

        @testset "Indexing of AbstractOperations [$(typeof(arch))]" begin
            
            grid = RectilinearGrid(arch, size=(3, 3, 3), extent=(1, 1, 1))

            test_indices   = [(2:3, :, :), (:, 2:3, :), (:, :, 2:3)]
            face_indices   = [(2:2, :, :), (:, 2:2, :), (:, :, 2:2)]
            center_indices = [(3:3, :, :), (:, 3:3, :), (:, :, 3:3)]

            FaceFields = (XFaceField, YFaceField, ZFaceField)
            
            for (ti, fi, ci, FaceField) in zip(test_indices, face_indices, center_indices, FaceFields)
                a = CenterField(grid)
                b = CenterField(grid, indices = ti)
                @test indices(a * b)  == ti
                @test indices(sin(b)) == ti
                            
                c = CenterField(grid, indices=ti)
                d = FaceField(grid, indices=ti)
                @test indices(c * d) == fi
                @test indices(d * c) == ci
            end

            a = CenterField(grid, indices = test_indices[1])
            b = XFaceField(grid,  indices = test_indices[2])
            c = YFaceField(grid,  indices = test_indices[3])

            d = Field((Face, Face, Center), grid, indices = (:, 2:3, 1:2))

            @test indices(a * b * c) == (2:3, 2:3, 2:3)
            @test indices(b * a * c) == (3:3, 2:3, 2:3)
            @test indices(c * a * b) == (2:3, 3:3, 2:3)
            @test indices(a * b * c * d) == (2:3, 2:2, 2:2)
            @test indices(b * c * d * a) == (3:3, 2:2, 2:2)
            @test indices(c * d * a * b) == (2:3, 3:3, 2:2)
            @test indices(d * a * b * c) == (3:3, 3:3, 2:2)
        end
    end
end
