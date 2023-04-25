include("dependencies_for_runtests.jl")

using Oceananigans.AbstractOperations: UnaryOperation, Derivative, BinaryOperation, MultiaryOperation
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: ℑxyᶜᶠᵃ, ℑxyᶠᶜᵃ
using Oceananigans.Fields: compute_at!
using Oceananigans.BuoyancyModels: BuoyancyField

function compute_derivative(model, ∂)
    T, S = model.tracers
    parent(S) .= π
    @compute ∂S = Field(∂(S))
    result = Array(interior(∂S))
    return all(result .≈ zero(model.grid))
end

function compute_unary(unary, model)
    set!(model; S=π)
    T, S = model.tracers
    @compute uS = Field(unary(S), data=model.pressures.pNHS.data)
    result = Array(interior(uS))
    return all(result .≈ unary(eltype(model.grid)(π)))
end

function compute_plus(model)
    set!(model; S=π, T=42)
    T, S = model.tracers
    @compute ST = Field(S + T, data=model.pressures.pNHS.data)
    result = Array(interior(ST))
    return all(result .≈ eltype(model.grid)(π + 42))
end

function compute_many_plus(model)
    set!(model; u=2, S=π, T=42)
    T, S = model.tracers
    u, v, w = model.velocities
    @compute uTS = Field(@at((Center, Center, Center), u + T + S))
    result = Array(interior(uTS))
    return all(result .≈ eltype(model.grid)(2 + π + 42))
end

function compute_minus(model)
    set!(model; S=π, T=42)
    T, S = model.tracers
    @compute ST = Field(S - T, data=model.pressures.pNHS.data)
    result = Array(interior(ST))
    return all(result .≈ eltype(model.grid)(π - 42))
end

function compute_times(model)
    set!(model; S=π, T=42)
    T, S = model.tracers
    @compute ST = Field(S * T, data=model.pressures.pNHS.data)
    result = Array(interior(ST))
    return all(result .≈ eltype(model.grid)(π * 42))
end

function compute_kinetic_energy(model)
    u, v, w = model.velocities
    set!(u, 1)
    set!(v, 2)
    set!(w, 3)

    kinetic_energy_operation = @at (Center, Center, Center) (u^2 + v^2 + w^2) / 2
    @compute kinetic_energy = Field(kinetic_energy_operation, data=model.pressures.pNHS.data)

    return all(interior(kinetic_energy, 2:3, 2:3, 2:3) .≈ 7)
end

function horizontal_average_of_plus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π * z)
    T₀(x, y, z) = 42 * z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S + T, dims=(1, 2)))

    @test ST.operand isa Reduction

    zC = znodes(model.grid, Center())
    correct_profile = @. sin(π * zC) + 42 * zC
    computed_profile = Array(interior(ST, 1, 1, :))

    return all(computed_profile .≈ correct_profile)
end

function zonal_average_of_plus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π * z) * sin(π * y)
    T₀(x, y, z) = 42 * z + y^2
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S + T, dims=1))

    _, yC, zC = nodes(model.grid, Center(), Center(), Center(); reshape=true)

    correct_slice = @. sin(π * zC) * sin(π * yC) + 42 * zC + yC^2
    computed_slice = Array(interior(ST, 1, :, :))

    return all(computed_slice .≈ view(correct_slice, 1, :, :))
end

function volume_average_of_times(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = 1 + sin(2π * x)
    T₀(x, y, z) = y
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S * T, dims=(1, 2, 3)))
    result = CUDA.@allowscalar ST[1, 1, 1]

    return result ≈ 0.5
end

function horizontal_average_of_minus(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π * z)
    T₀(x, y, z) = 42 * z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S - T, dims=(1, 2)))

    zC = znodes(model.grid, Center())
    correct_profile = @. sin(π * zC) - 42 * zC
    computed_profile = Array(interior(ST, 1, 1, 1:Nz))

    return all(computed_profile .≈ correct_profile)
end

function horizontal_average_of_times(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    S₀(x, y, z) = sin(π*z)
    T₀(x, y, z) = 42*z
    set!(model; S=S₀, T=T₀)
    T, S = model.tracers

    @compute ST = Field(Average(S * T, dims=(1, 2)))

    zC = znodes(model.grid, Center())
    correct_profile = @. sin(π * zC) * 42 * zC
    computed_profile = Array(interior(ST, 1, 1, 1:Nz))

    return all(computed_profile .≈ correct_profile)
end

function multiplication_and_derivative_ccf(model)
    Ny, Nz = model.grid.Ny, model.grid.Nz

    w₀(x, y, z) = sin(π * z)
    T₀(x, y, z) = 42 * z
    set!(model; enforce_incompressibility=false, w=w₀, T=T₀)

    w = model.velocities.w
    T = model.tracers.T

    @compute wT = Field(Average(w * ∂z(T), dims=(1, 2)))

    zF = znodes(model.grid, Face())
    correct_profile = @. 42 * sin(π * zF)
    computed_profile = Array(interior(wT, 1, 1, 1:Nz))

    # Omit boundaries
    return all(computed_profile[2:Nz] .≈ correct_profile[2:Nz])
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

    zF = znodes(model.grid, Face())
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
    model = NonhydrostaticModel(; grid)

    u, v, w = model.velocities
    parent(u) .= 1 + rand()
    parent(v) .= 2 + rand()
    parent(w) .= 3 + rand()

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

    return all(interior(tke, 2:3, 2:3, 2:3) .== 9/2)
end

function computations_with_averaged_field_derivative(model)

    set!(model, enforce_incompressibility = false, u = (x, y, z) -> z, v = 2, w = 3)

    u, v, w, T, S = fields(model)

    # Two ways to compute turbulent kinetic energy
    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))

    # This tests a vertical derivative of an Averaged Field
    shear_production_op = @at (Center, Center, Center) u * w * ∂z(U)
    shear = Field(shear_production_op)
    compute!(shear)

    set!(model, T = (x, y, z) -> 3 * z)

    return all(interior(shear, 2:3, 2:3, 2:3) .== interior(T, 2:3, 2:3, 2:3))
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

    return all(interior(tke, 2:3, 2:3, 2:3) .== 9/2)
end

for arch in archs
    A = typeof(arch)
    @testset "Computed Fields [$A]" begin
        @info "  Testing computed Fields [$A]..."

        gravitational_acceleration = 1
        equation_of_state = LinearEquationOfState(thermal_expansion=1, haline_contraction=1)
        buoyancy = SeawaterBuoyancy(; gravitational_acceleration, equation_of_state)

        underlying_grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
        bottom(x, y) = -2 # below the grid!
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

        for grid in (underlying_grid, immersed_grid)
            G = typeof(grid).name.wrapper
            model = NonhydrostaticModel(; grid, buoyancy, tracers = (:T, :S))

            @testset "Instantiating and computing computed fields [$A, $G]" begin
                @info "  Testing computed Field instantiation and computation [$A, $G]..."
                c = CenterField(grid)
                c² = compute!(Field(c^2))
                @test c² isa Field

                # Test indices
                indices = [(:, :, :), (1, :, :), (:, :, grid.Nz), (2:4, 3, 5)]
                sizes   = [(4, 4, 4), (1, 4, 4), (4, 4, 1),       (3, 1, 1)]
                for (ii, sz) in zip(indices, sizes)
                    c² = compute!(Field(c^2; indices=ii))
                    @test size(interior(c²)) === sz
                end
            end

            @testset "Derivative computations [$A, $G]" begin
                @info "      Testing correctness of compute! derivatives..."
                @test compute_derivative(model, ∂x)
                @test compute_derivative(model, ∂y)
                @test compute_derivative(model, ∂z)
            end

            @testset "Unary computations [$A, $G]" begin
                @info "      Testing correctness of compute! unary operations..."
                for unary in (sqrt, sin, cos, exp, tanh)
                    @test compute_unary(unary, model)
                end
            end

            @testset "Binary computations [$A, $G]" begin
                @info "      Testing correctness of compute! binary operations..."
                @test compute_plus(model)
                @test compute_minus(model)
                @test compute_times(model)

                # Basic compilation test for nested BinaryOperations...
                u, v, w = model.velocities
                @test try compute!(Field(u + v - w)); true; catch; false; end
            end

            @testset "Multiary computations [$A, $G]" begin
                @info "      Testing correctness of compute! multiary operations..."
                @test compute_many_plus(model)

                @info "      Testing correctness of compute! kinetic energy..."
                @test compute_kinetic_energy(model)
            end

            @testset "Computations with KernelFunctionOperation [$A, $G]" begin
                @test begin
                    @inline trivial_kernel_function(i, j, k, grid) = 1
                    op = KernelFunctionOperation{Center, Center, Center}(trivial_kernel_function, grid)
                    f = Field(op)
                    compute!(f)
                    f isa Field && f.operand === op
                end

                @test begin
                    @inline trivial_parameterized_kernel_function(i, j, k, grid, μ) = μ
                    op = KernelFunctionOperation{Center, Center, Center}(trivial_parameterized_kernel_function, grid, 0.1)
                    f = Field(op)
                    compute!(f)
                    f isa Field && f.operand === op
                end

                ϵ(x, y, z) = 2rand() - 1
                set!(model, u=ϵ, v=ϵ)
                u, v, w = model.velocities
                ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)

                ζ = Field(ζ_op) # identical to `VerticalVorticityField`
                compute!(ζ)
                @test ζ isa Field && ζ.operand.kernel_function === ζ₃ᶠᶠᶜ

                ζxy = Field(ζ_op, indices=(:, :, 1))
                compute!(ζxy)
                @test all(interior(ζxy, :, :, 1) .== interior(ζ, :, :, 1))

                ζxz = Field(ζ_op, indices=(:, 1, :))
                compute!(ζxz)
                @test all(interior(ζxz, :, 1, :) .== interior(ζ, :, 1, :))

                ζyz = Field(ζ_op, indices=(1, :, :))
                compute!(ζyz)
                @test all(interior(ζyz, 1, :, :) .== interior(ζ, 1, :, :))
            end

            @testset "Operations with computed Fields [$A, $G]" begin
                @info "      Testing operations with computed Fields..."
                @test operations_with_computed_field(model)
            end

            @testset "Horizontal averages of operations [$A, $G]" begin
                @info "      Testing horizontal averages..."
                @test horizontal_average_of_plus(model)
                @test horizontal_average_of_minus(model)
                @test horizontal_average_of_times(model)

                @test multiplication_and_derivative_ccf(model)
                @test multiplication_and_derivative_ccc(model)
            end

            @testset "Zonal averages of operations [$A, $G]" begin
                @info "      Testing zonal averages..."
                @test zonal_average_of_plus(model)
            end

            @testset "Volume averages of operations [$A, $G]" begin
                @info "      Testing volume averages..."
                @test volume_average_of_times(model)
            end

            @testset "Field boundary conditions [$A, $G]" begin
                @info "      Testing boundary conditions for Field..."

                set!(model; S=π, T=42)
                T, S = model.tracers

                @compute ST = Field(S + T, data=model.pressures.pNHS.data)

                Nx, Ny, Nz = size(model.grid)
                Hx, Hy, Hz = halo_size(model.grid)

                # Periodic xy
                ii = 1+Hx:Nx+Hx
                jj = 1+Hy:Ny+Hy
                kk = 1+Hz:Nz+Hz
                @test all(view(parent(ST), Hx, jj, kk) .== view(parent(ST), Nx+1+Hx, jj, kk))
                @test all(view(parent(ST), ii, Hy, kk) .== view(parent(ST), ii, Ny+1+Hy, kk))
                
                # Bounded z
                @test all(view(parent(ST), ii, jj, Hz)    .== view(parent(ST), ii, jj, 1+Hz))
                @test all(view(parent(ST), ii, jj, Nz+Hz) .== view(parent(ST), ii, jj, Nz+1+Hz))

                @compute ST_face = Field(@at (Center, Center, Face) S * T)

                # These are initially 0 and remain 0
                @test all(view(parent(ST_face), ii, jj, Hz) .== 0)
                @test all(view(parent(ST_face), ii, jj, Nz+2+Hz) .== 0)
            end

            @testset "Operations with Averaged Field [$A, $G]" begin
                @info "      Testing operations with Averaged Field..."

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
                @testset "Computations with BuoyancyFields [$A, $G, $(typeof(buoyancy).name.wrapper)]" begin
                    @info "      Testing computations with BuoyancyField " *
                          "[$A, $G, $(typeof(buoyancy).name.wrapper)]..."

                    @test computations_with_buoyancy_field(arch, buoyancy)
                end
            end

            @testset "Computations with Averaged Fields [$A, $G]" begin
                @info "      Testing computations with Averaged Field [$A, $G]..."

                @test computations_with_averaged_field_derivative(model)

                u, v, w = model.velocities

                set!(model, enforce_incompressibility = false, u = (x, y, z) -> z, v = 2, w = 3)

                # A few ways to compute turbulent kinetic energy
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

                computed_tke = Field(tke_ccc)
                @test try compute!(computed_tke); true; catch; false; end
                @test all(interior(computed_tke, 2:3, 2:3, 2:3) .== 9/2)

                tke_window = Field(tke_ccc, indices=(2:3, 2:3, 2:3))
                if (grid isa ImmersedBoundaryGrid) & (arch==GPU())
                    @test_broken try compute!(tke_window); true; catch; false; end
                    @test_broken all(interior(tke_window) .== 9/2)
                else
                    @test try compute!(tke_window); true; catch; false; end
                    @test all(interior(tke_window) .== 9/2)
                end

                # Computations along slices
                tke_xy = Field(tke_ccc, indices=(:, :, 2))
                @test try compute!(tke_xy); true; catch; false; end
                @test all(interior(tke_xy, 2:3, 2:3, 1) .== 9/2)

                tke_xz = Field(tke_ccc, indices=(2:3, 2, 2:3))
                tke_yz = Field(tke_ccc, indices=(2, 2:3, 2:3))
                tke_x = Field(tke_ccc, indices=(2:3, 2, 2))

                if (grid isa ImmersedBoundaryGrid) & (arch==GPU())
                    @test_broken try compute!(tke_xz); true; catch; false; end
                    @test_broken all(interior(tke_xz) .== 9/2)

                    @test_broken try compute!(tke_yz); true; catch; false; end
                    @test_broken all(interior(tke_yz) .== 9/2)

                    @test_broken try compute!(tke_x); true; catch; false; end
                    @test_broken all(interior(tke_x) .== 9/2)
                else
                    @test try compute!(tke_xz); true; catch; false; end
                    @test all(interior(tke_xz) .== 9/2)

                    @test try compute!(tke_yz); true; catch; false; end
                    @test all(interior(tke_yz) .== 9/2)

                    @test try compute!(tke_x); true; catch; false; end
                    @test all(interior(tke_x) .== 9/2)
                end
            end

            @testset "Computations with Fields [$A, $G]" begin
                @info "      Testing computations with Field [$A, $G]..."
                @test computations_with_computed_fields(model)
            end

            @testset "Conditional computation of Field and BuoyancyField [$A, $G]" begin
                @info "      Testing conditional computation of Field and BuoyancyField " *
                      "[$A, $G]..."

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
end

