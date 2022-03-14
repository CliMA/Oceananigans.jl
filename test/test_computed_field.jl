include("dependencies_for_runtests.jl")

using Oceananigans.AbstractOperations: UnaryOperation, Derivative, BinaryOperation, MultiaryOperation
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: ℑxyᶜᶠᵃ, ℑxyᶠᶜᵃ
using Oceananigans.Fields: compute_at!
using Oceananigans.BuoyancyModels: BuoyancyField

function compute_derivative(model, ∂)
    T, S = model.tracers
    S.data.parent .= π
    @compute ∂S = Field(∂(S))
    result = Array(interior(∂S))
    return all(result .≈ zero(eltype(model.grid)))
end

function compute_unary(unary, model)
    set!(model; S=π)
    T, S = model.tracers
    @compute uS = Field(unary(S), data=model.pressures.pHY′.data)
    result = Array(interior(uS))
    return all(result .≈ unary(eltype(model.grid)(π)))
end

function compute_plus(model)
    set!(model; S=π, T=42)
    T, S = model.tracers
    @compute ST = Field(S + T, data=model.pressures.pHY′.data)
    result = Array(interior(ST))
    return all(result .≈ eltype(model.grid)(π+42))
end

function compute_many_plus(model)
    set!(model; u=2, S=π, T=42)
    T, S = model.tracers
    u, v, w = model.velocities
    @compute uTS = Field(@at((Center, Center, Center), u + T + S))
    result = Array(interior(uTS))
    return all(result .≈ eltype(model.grid)(2+π+42))
end

function compute_minus(model)
    set!(model; S=π, T=42)
    T, S = model.tracers
    @compute ST = Field(S - T, data=model.pressures.pHY′.data)
    result = Array(interior(ST))
    return all(result .≈ eltype(model.grid)(π-42))
end

function compute_times(model)
    set!(model; S=π, T=42)
    T, S = model.tracers
    @compute ST = Field(S * T, data=model.pressures.pHY′.data)
    result = Array(interior(ST))
    return all(result .≈ eltype(model.grid)(π*42))
end

function compute_kinetic_energy(model)
    u, v, w = model.velocities
    set!(u, 1)
    set!(v, 2)
    set!(w, 3)

    kinetic_energy_operation = @at (Center, Center, Center) (u^2 + v^2 + w^2) / 2
    @compute kinetic_energy = Field(kinetic_energy_operation, data=model.pressures.pHY′.data)
    result = Array(interior(kinetic_energy))[2:3, 2:3, 2:3]

    return all(result .≈ 7)
end

function horizontal_average_of_plus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π * z)
    T₀(x, y, z) = 42 * z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S + T, dims=(1, 2)))

    @test ST.operand isa Reduction
    @test ST.operand.reduce! === mean!

    zC = znodes(Center, model.grid)
    correct_profile = @. sin(π * zC) + 42 * zC

    result = Array(interior(ST))[:]

    return all(result .≈ correct_profile)
end

function zonal_average_of_plus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π*z) * sin(π*y)
    T₀(x, y, z) = 42*z + y^2
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S + T, dims=1))

    yC = ynodes(Center, model.grid, reshape=true)
    zC = znodes(Center, model.grid, reshape=true)
    correct_slice = @. sin(π * zC) * sin(π * yC) + 42*zC + yC^2

    result = Array(interior(ST))

    return all(result[1, :, :] .≈ correct_slice[1, :, :])
end

function volume_average_of_times(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = 1 + sin(2π*x)
    T₀(x, y, z) = y
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S * T, dims=(1, 2, 3)))

    result = Array(interior(ST))

    return all(result[1, 1, 1] .≈ 0.5)
end

function horizontal_average_of_minus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S - T, dims=(1, 2)))

    zC = znodes(Center, model.grid)
    correct_profile = @. sin(π * zC) - 42 * zC

    result = Array(interior(ST))

    return all(result[1, 1, 1:Nz] .≈ correct_profile)
end

function horizontal_average_of_times(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S * T, dims=(1, 2)))

    zC = znodes(Center, model.grid)
    correct_profile = @. sin(π * zC) * 42 * zC

    result = Array(interior(ST))

    return all(result[1, 1, 1:Nz] .≈ correct_profile)
end

function multiplication_and_derivative_ccf(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    w₀(x, y, z) = sin(π * z)
    T₀(x, y, z) = 42 * z
    set!(model; enforce_incompressibility=false, w=w₀, T=T₀)

    w = model.velocities.w
    T = model.tracers.T

    @compute wT = Field(Average(w * ∂z(T), dims=(1, 2)))

    zF = znodes(Face, model.grid)
    correct_profile = @. 42 * sin(π * zF)

    result = Array(interior(wT))

    # Omit both halos and boundary points
    return all(result[1, 1, 2:Nz] .≈ correct_profile[2:Nz])
end

const C = Center
const F = Face

function multiplication_and_derivative_ccc(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    w₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; enforce_incompressibility=false, w=w₀, T=T₀)

    w = model.velocities.w
    T = model.tracers.T

    wT_ccc = @at (C, C, C) w * ∂z(T)
    @compute wT_ccc_avg = Field(Average(wT_ccc, dims=(1, 2)))

    zF = znodes(Face, model.grid)
    sinusoid = sin.(π * zF)
    interped_sin = [(sinusoid[k] + sinusoid[k+1]) / 2 for k in 1:model.grid.Nz]
    correct_profile = interped_sin .* 42

    result = Array(interior(wT_ccc_avg))

    # Omit boundary-adjacent points from comparison
    return all(result[1, 1, 2:Nz-1] .≈ correct_profile[2:Nz-1])
end

function computation_including_boundaries(arch)
    topo = (Periodic, Bounded, Bounded)
    grid = RectilinearGrid(arch, topology=topo, size=(13, 17, 19), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid)

    u, v, w = model.velocities
    @. u.data = 1 + rand()
    @. v.data = 2 + rand()
    @. w.data = 3 + rand()

    op = @at (Center, Face, Face) u * v * w
    @compute uvw = Field(op)

    return all(interior(uvw) .!= 0)
end

function operations_with_computed_field(model)
    u, v, w = model.velocities
    uv = Field(u * v)
    @compute uvw = Field(uv * w)
    return true
end

function operations_with_averaged_field(model)
    u, v, w = model.velocities
    UV = Field(Average(u * v, dims=(1, 2)))
    wUV = Field(w * UV)
    compute!(wUV)
    return true
end

#=
function pressure_field(model)
    p = PressureField(model)
    u, v, w = model.velocities
    @compute up = Field(u * p)
    return true
end
=#

function computations_with_buoyancy_field(arch, buoyancy)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    tracers = buoyancy isa BuoyancyTracer ? :b : (:T, :S)
    model = NonhydrostaticModel(grid=grid,
                                tracers=tracers, buoyancy=buoyancy)

    b = BuoyancyField(model)
    u, v, w = model.velocities

    compute!(b)

    ub = Field(b * u)
    vb = Field(b * v)
    wb = Field(b * w)

    compute!(ub)
    compute!(vb)
    compute!(wb)

    return true # test that it doesn't error
end

function computations_with_averaged_fields(model)
    u, v, w, T, S = fields(model)

    set!(model, enforce_incompressibility = false, u = (x, y, z) -> z, v = 2, w = 3)

    # Two ways to compute turbulent kinetic energy
    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))

    tke_op = @at (Center, Center, Center) ((u - U)^2  + (v - V)^2 + w^2) / 2
    tke = Field(tke_op)
    compute!(tke)

    return all(interior(tke)[2:3, 2:3, 2:3] .== 9/2)
end

function computations_with_averaged_field_derivative(model)

    set!(model, enforce_incompressibility = false, u = (x, y, z) -> z, v = 2, w = 3)

    u, v, w, T, S = fields(model)

    # Two ways to compute turbulent kinetic energy
    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))

    # This tests a vertical derivative of an AveragedField
    shear_production_op = @at (Center, Center, Center) u * w * ∂z(U)
    shear = Field(shear_production_op)
    compute!(shear)

    set!(model, T = (x, y, z) -> 3 * z)

    return all(interior(shear)[2:3, 2:3, 2:3] .== interior(T)[2:3, 2:3, 2:3])
end

function computations_with_computed_fields(model)
    u, v, w, T, S = fields(model)

    set!(model, enforce_incompressibility = false, u = (x, y, z) -> z, v = 2, w = 3)

    # Two ways to compute turbulent kinetic energy
    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))

    u′ = Field(u - U)
    v′ = Field(v - V)

    tke_op = @at (Center, Center, Center) (u′^2  + v′^2 + w^2) / 2
    tke = Field(tke_op)
    compute!(tke)

    return all(interior(tke)[2:3, 2:3, 2:3] .== 9/2)
end

for arch in archs
    @testset "Computed Fields [$(typeof(arch))]" begin
        @info "  Testing computed Fields [$(typeof(arch))]..."

        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Bounded))

        buoyancy = SeawaterBuoyancy(gravitational_acceleration = 1,
                                    equation_of_state = LinearEquationOfState(thermal_expansion=1, haline_contraction=1))

        model = NonhydrostaticModel(; grid, buoyancy, tracers = (:T, :S))

        @testset "Derivative computations [$(typeof(arch))]" begin
            @info "      Testing compute! derivatives..."
            @test compute_derivative(model, ∂x)
            @test compute_derivative(model, ∂y)
            @test compute_derivative(model, ∂z)
        end

        @testset "Unary computations [$(typeof(arch))]" begin
            @info "      Testing compute! unary operations..."
            for unary in (sqrt, sin, cos, exp, tanh)
                @test compute_unary(unary, model)
            end
        end

        @testset "Binary computations [$(typeof(arch))]" begin
            @info "      Testing compute! binary operations..."
            @test compute_plus(model)
            @test compute_minus(model)
            @test compute_times(model)

            # Basic compilation test for nested BinaryOperations...
            u, v, w = model.velocities
            @test try compute!(Field(u + v - w)); true; catch; false; end
        end

        @testset "Multiary computations [$(typeof(arch))]" begin
            @info "      Testing compute! multiary operations..."
            @test compute_many_plus(model)

            @info "      Testing compute! kinetic energy..."
            @test compute_kinetic_energy(model)
        end

        @testset "Computations with KernelFunctionOperation [$(typeof(arch))]" begin
            @test begin
                @inline trivial_kernel_function(i, j, k, grid) = 1
                op = KernelFunctionOperation{Center, Center, Center}(trivial_kernel_function, grid)
                f = Field(op)
                compute!(f)
                f isa Field && f.operand === op
            end

            @test begin
                @inline trivial_parameterized_kernel_function(i, j, k, grid, μ) = μ
                op = KernelFunctionOperation{Center, Center, Center}(trivial_parameterized_kernel_function, grid, parameters=0.1)
                f = Field(op)
                compute!(f)
                f isa Field && f.operand === op
            end

            @test begin
                u, v, w = model.velocities
                ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, computed_dependencies=(u, v))
                ζ = Field(ζ_op) # identical to `VerticalVorticityField`
                compute!(ζ)
                ζ isa Field && ζ.operand.kernel_function === ζ₃ᶠᶠᶜ
            end
        end

        @testset "Operations with Field and PressureField [$(typeof(arch))]" begin
            @info "      Testing operations with Field..."
            @test operations_with_computed_field(model)

            # @info "      Testing PressureField..."
            # @test pressure_field(model)
        end

        @testset "Horizontal averages of operations [$(typeof(arch))]" begin
            @info "      Testing horizontal averges..."
            @test horizontal_average_of_plus(model)
            @test horizontal_average_of_minus(model)
            @test horizontal_average_of_times(model)

            @test multiplication_and_derivative_ccf(model)
            @test multiplication_and_derivative_ccc(model)
        end

        @testset "Zonal averages of operations [$(typeof(arch))]" begin
            @info "      Testing zonal averges..."
            @test zonal_average_of_plus(model)
        end

        @testset "Volume averages of operations [$(typeof(arch))]" begin
            @info "      Testing volume averges..."
            @test volume_average_of_times(model)
        end

        @testset "Field boundary conditions [$(typeof(arch))]" begin
            @info "      Testing boundary conditions for Field..."

            set!(model; S=π, T=42)
            T, S = model.tracers

            @compute ST = Field(S + T, data=model.pressures.pHY′.data)

            Nx, Ny, Nz = size(model.grid)

            @test all(ST.data[0, 1:Ny, 1:Nz]  .== ST.data[Nx+1, 1:Ny, 1:Nz])
            @test all(ST.data[1:Nx, 0, 1:Nz]  .== ST.data[1:Nx, Ny+1, 1:Nz])
            @test all(ST.data[1:Nx, 1:Ny, 0]  .== ST.data[1:Nx, 1:Ny, 1])
            @test all(ST.data[1:Nx, 1:Ny, Nz] .== ST.data[1:Nx, 1:Ny, Nz+1])

            @compute ST_face = Field(@at (Center, Center, Face) S * T)

            @test all(ST_face.data[1:Nx, 1:Ny, 0] .== 0)
            @test all(ST_face.data[1:Nx, 1:Ny, Nz+2] .== 0)
        end

        @testset "Operations with AveragedField [$(typeof(arch))]" begin
            @info "      Testing operations with AveragedField..."

            T, S = model.tracers
            TS = Field(Average(T * S, dims=(1, 2)))
            @test operations_with_averaged_field(model)
        end

        @testset "Compute! on faces along bounded dimensions" begin
            @info "      Testing compute! on faces along bounded dimensions..."
            @test computation_including_boundaries(arch)
        end

        EquationsOfState = (LinearEquationOfState, SeawaterPolynomials.RoquetEquationOfState,
                            SeawaterPolynomials.TEOS10EquationOfState)

        buoyancies = (BuoyancyTracer(), SeawaterBuoyancy(),
                      (SeawaterBuoyancy(equation_of_state=eos()) for eos in EquationsOfState)...)

        for buoyancy in buoyancies
            @testset "Computations with BuoyancyFields [$(typeof(arch)), $(typeof(buoyancy).name.wrapper)]" begin
                @info "      Testing computations with BuoyancyField " *
                      "[$(typeof(arch)), $(typeof(buoyancy).name.wrapper)]..."

                @test computations_with_buoyancy_field(arch, buoyancy)
            end
        end

        @testset "Computations with AveragedFields [$(typeof(arch))]" begin
            @info "      Testing computations with AveragedField [$(typeof(arch))]..."

            @test computations_with_averaged_field_derivative(model)

            u, v, w = model.velocities

            set!(model, enforce_incompressibility = false, u = (x, y, z) -> z, v = 2, w = 3)

            # Two ways to compute turbulent kinetic energy
            U = Field(Average(u, dims=(1, 2)))
            V = Field(Average(v, dims=(1, 2)))

            # Build up compilation tests incrementally...
            u_prime              = u - U
            u_prime_ccc          = @at (Center, Center, Center) u - U
            u_prime_squared      = (u - U)^2
            u_prime_squared_ccc  = @at (Center, Center, Center) (u - U)^2
            horizontal_twice_tke = (u - U)^2 + (v - V)^2
            horizontal_tke       = ((u - U)^2 + (v - V)^2) / 2
            horizontal_tke_ccc   = @at (Center, Center, Center) ((u - U)^2 + (v - V)^2) / 2
            twice_tke            = (u - U)^2  + (v - V)^2 + w^2
            tke                  = ((u - U)^2  + (v - V)^2 + w^2) / 2
            tke_ccc              = @at (Center, Center, Center) ((u - U)^2  + (v - V)^2 + w^2) / 2

            @test try compute!(Field(u_prime             )); true; catch; false; end
            @test try compute!(Field(u_prime_ccc         )); true; catch; false; end
            @test try compute!(Field(u_prime_squared     )); true; catch; false; end
            @test try compute!(Field(u_prime_squared_ccc )); true; catch; false; end
            @test try compute!(Field(horizontal_twice_tke)); true; catch; false; end
            @test try compute!(Field(horizontal_tke      )); true; catch; false; end
            @test try compute!(Field(twice_tke           )); true; catch; false; end

            @test try compute!(Field(horizontal_tke_ccc  )); true; catch; false; end
            @test try compute!(Field(tke                 )); true; catch; false; end
            @test try compute!(Field(tke_ccc             )); true; catch; false; end

            computed_tke = Field(tke_ccc)
            compute!(computed_tke)
            @test all(interior(computed_tke)[2:3, 2:3, 2:3] .== 9/2)
        end

        @testset "Computations with Fields [$(typeof(arch))]" begin
            @info "      Testing computations with Field [$(typeof(arch))]..."
            @test computations_with_computed_fields(model)
        end

        @testset "Conditional computation of Field and BuoyancyField [$(typeof(arch))]" begin
            @info "      Testing conditional computation of Field and BuoyancyField " *
                  "[$(typeof(arch))]..."

            set!(model, u=2, v=0, w=0, T=3, S=0)
            u, v, w, T, S = fields(model)

            uT = Field(u * T)

            α = model.buoyancy.model.equation_of_state.thermal_expansion
            g = model.buoyancy.model.gravitational_acceleration
            b = BuoyancyField(model)

            compute_at!(uT, 1.0)
            compute_at!(b, 1.0)
            @test all(interior(uT) .== 6)
            @test all(interior(b) .== g * α * 3)

            set!(model, u=2, T=4)
            compute_at!(uT, 1.0)
            compute_at!(b, 1.0)
            @test all(interior(uT) .== 6)
            @test all(interior(b) .== g * α * 3)

            compute_at!(uT, 2.0)
            compute_at!(b, 2.0)
            @test all(interior(uT) .== 8)
            @test all(interior(b) .== g * α * 4)
        end
    end
end
