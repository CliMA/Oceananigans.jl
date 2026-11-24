include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Operators: uniform_reconstruction_coefficients, ℑxᶠᵃᵃ, ∂xᵣᶠᶜᶜ, ∂xᵣᶜᶜᶜ, ∂zᶜᶜᶠ
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ, with_tracers,
                                       required_halo_size_x, required_halo_size_y, required_halo_size_z,
                                       cell_diffusion_timescale

# Divergence of the diffusive flux of c, ∇ ⋅ q = - ∇ ⋅ (κ ∇c)
function tracer_flux_divergence(c, closure)
    grid = c.grid
    closure = with_tracers((:c,), closure)
    fill_halo_regions!(c)
    op = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, grid, closure, nothing, Val(:c), c,
                                                         Clock(grid), (; c), nothing)
    return Field(op)
end

# Divergence of the viscous stress ∂ⱼτᵢⱼ
function viscous_stress_divergence(u, v, w, closure)
    grid = u.grid
    fill_halo_regions!((u, v, w))
    fields = (; u, v, w)
    clock = Clock(grid)
    ops = (KernelFunctionOperation{Face,   Center, Center}(∂ⱼ_τ₁ⱼ, grid, closure, nothing, clock, fields, nothing),
           KernelFunctionOperation{Center, Face,   Center}(∂ⱼ_τ₂ⱼ, grid, closure, nothing, clock, fields, nothing),
           KernelFunctionOperation{Center, Center, Face  }(∂ⱼ_τ₃ⱼ, grid, closure, nothing, clock, fields, nothing))
    return Tuple(Field(op) for op in ops)
end

max_error(a, b) = maximum(abs, Field(a - b))
mean_absolute_error(a, b) = sum(abs, Field(a - b)) / length(interior(b))
convergence_rate(errors, Ns) = log(errors[end-1] / errors[end]) / log(Ns[end] / Ns[end-1])

const Ns = (16, 32, 64)
const L = 2π

# Periodic: ∇²c for c = sin(x) cos(2y) sin(z)
function periodic_tracer_laplacian_errors(arch, order)
    κ = 0.7
    closure = ScalarDiffusivity(κ=κ, stencil_order=order)
    c₀(x, y, z) = sin(x) * cos(2y) * sin(z)
    errors = map(Ns) do N
        grid = RectilinearGrid(arch; size=(N, N, N), extent=(L, L, L), halo=(3, 3, 3), topology=(Periodic, Periodic, Periodic))
        c = CenterField(grid)
        set!(c, c₀)
        max_error(tracer_flux_divergence(c, closure), 6κ * c) # - κ ∇²c = 6κ c
    end
    return errors
end

# A velocity field that is divergence-free on the C-grid (as after a pressure projection):
#   u =  sin(x) cos(y),  v = - cos(x) sin(y) + sin(y) cos(z),  w = - cos(y) sin(z)
# so that ∇²uᵢ = - 2 uᵢ and ∂ⱼτᵢⱼ = - ν ∇²uᵢ = 2ν uᵢ.
function periodic_viscous_errors(arch, order)
    ν = 0.3
    closure = ScalarDiffusivity(ν=ν, stencil_order=order)
    errors = map(Ns) do N
        grid = RectilinearGrid(arch; size=(N, N, N), extent=(L, L, L), halo=(3, 3, 3), topology=(Periodic, Periodic, Periodic))
        u, v, w = XFaceField(grid), YFaceField(grid), ZFaceField(grid)
        set!(u, (x, y, z) -> sin(x) * cos(y))
        set!(v, (x, y, z) -> - cos(x) * sin(y) + sin(y) * cos(z))
        set!(w, (x, y, z) -> - cos(y) * sin(z))
        τ = viscous_stress_divergence(u, v, w, closure)
        maximum(max_error(τᵢ, 2ν * uᵢ) for (τᵢ, uᵢ) in zip(τ, (u, v, w)))
    end
    return errors
end

# Bounded (no-flux walls) in x and z: c = cos(x) cos(z) on [0, π]² satisfies ∂c/∂n = 0 on the walls.
function bounded_tracer_laplacian_errors(arch, order)
    κ = 0.7
    closure = ScalarDiffusivity(κ=κ, stencil_order=order)
    errors = map(Ns) do N
        grid = RectilinearGrid(arch; size=(N, 4, N), x=(0, π), y=(0, 1), z=(0, π), halo=(3, 3, 3),
                               topology=(Bounded, Periodic, Bounded))
        c = CenterField(grid)
        set!(c, (x, y, z) -> cos(x) * cos(z))
        result = tracer_flux_divergence(c, closure)
        exact = Field(2κ * c)
        # Max error, and L1 error (mean absolute error)
        (max_error(result, exact), mean_absolute_error(result, exact))
    end
    return first.(errors), last.(errors)
end

# Computed separately: a difference inside one kernel contracts to an FMA on GPUs, which returns the product's rounding error
derivative_at_order(∂, location, grid, order, φ) = Array(interior(Field(KernelFunctionOperation{location...}(∂, grid, Val(order), φ))))
order_difference(∂, location, grid, high, low, φ) = derivative_at_order(∂, location, grid, high, φ) .- derivative_at_order(∂, location, grid, low, φ)

@testset "High-order ScalarDiffusivity" begin
    @testset "First derivative reconstruction coefficients" begin
        for (buffer, exact) in ((1, (1,)), (2, (5//4, -1//12)), (3, (49//36, -5//36, 1//90)))
            coefficients = uniform_reconstruction_coefficients(BigFloat, Val(:first_derivative), buffer)
            @test all(coefficients[buffer + n] ≈ exact[n] for n in 1:buffer)
            @test all(coefficients[buffer + n] ≈ -coefficients[buffer - n + 1] for n in 1:buffer)
        end
    end

    @testset "Construction" begin
        closure = ScalarDiffusivity(ν=1, κ=2, stencil_order=4)
        @test required_halo_size_x(closure) == 2
        @test required_halo_size_y(closure) == 2
        @test required_halo_size_z(closure) == 2
        @test required_halo_size_x(ScalarDiffusivity(ν=1, κ=2)) == 1
        @test required_halo_size_z(ScalarDiffusivity(ν=1, κ=2, stencil_order=6)) == 3
        @test summary(closure) == "ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0, κ=2.0, stencil_order=4)"
        @test summary(with_tracers((:c,), ScalarDiffusivity(κ=2, stencil_order=6))) ==
              "ScalarDiffusivity{ExplicitTimeDiscretization}(ν=nothing, κ=(c=2.0,), stencil_order=6)"
        @test summary(VerticalScalarDiffusivity(ν=1, κ=2, stencil_order=4)) == "VerticalScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0, κ=2.0, stencil_order=4)"

        @test_throws ArgumentError ScalarDiffusivity(ν=1, stencil_order=3)
        @test_throws ArgumentError ScalarDiffusivity(ν=1, stencil_order=0)
        @test_throws ArgumentError HorizontalScalarDiffusivity(ν=1, stencil_order=4)
        @test_throws ArgumentError VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1, stencil_order=4)

        # The largest eigenvalue of the Laplacian is 4/3 and 68/45 that of the second-order one at orders 4 and 6
        grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), halo=(3, 3, 3))
        model = NonhydrostaticModel(grid; closure, tracers=:c)
        @test cell_diffusion_timescale(model) ≈ 3/4 * (1/4)^2 / 2
        model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1, κ=2, stencil_order=6), tracers=:c)
        @test cell_diffusion_timescale(model) ≈ 45/68 * (1/4)^2 / 2
    end

    for arch in archs
        @testset "Periodic convergence [$(typeof(arch))]" begin
            for order in (2, 4, 6)
                @test isapprox(convergence_rate(periodic_tracer_laplacian_errors(arch, order), Ns), order, atol=0.2)
                @test isapprox(convergence_rate(periodic_viscous_errors(arch, order), Ns), order, atol=0.2)
            end
        end

        @testset "Order decreases near boundaries [$(typeof(arch))]" begin
            N = 16
            grid = RectilinearGrid(arch; size=(N, 4, N), x=(0, 1), y=(0, 1), z=(0, 1), halo=(3, 3, 3), topology=(Bounded, Periodic, Bounded))
            c = CenterField(grid)
            set!(c, (x, y, z) -> exp(x) * cos(3z))
            fill_halo_regions!(c)

            # The stencil on x-faces reaches a halo at faces 1:2 and N:N+1 for order 4, and one face further for order 6
            d = order_difference(∂xᵣᶠᶜᶜ, (Face, Center, Center), grid, 4, 2, c)
            @test maximum(abs, d[1:2, :, :]) == 0
            @test maximum(abs, d[N:N+1, :, :]) == 0
            @test minimum(abs, d[3:N-1, :, :]) > 0

            d = order_difference(∂xᵣᶠᶜᶜ, (Face, Center, Center), grid, 6, 4, c)
            @test maximum(abs, d[1:3, :, :]) == 0
            @test maximum(abs, d[N-1:N+1, :, :]) == 0
            @test minimum(abs, d[4:N-2, :, :]) > 0

            # The stencil at centers reads faces, and the boundary faces belong to the domain
            u = XFaceField(grid)
            set!(u, (x, y, z) -> sin(π * x) * cos(3z))
            fill_halo_regions!(u)
            d = order_difference(∂xᵣᶜᶜᶜ, (Center, Center, Center), grid, 4, 2, u)
            @test maximum(abs, d[1, :, :]) == 0
            @test maximum(abs, d[N, :, :]) == 0
            @test minimum(abs, d[2:N-1, :, :]) > 0

            ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(0.5))
            c = CenterField(ibg)
            set!(c, (x, y, z) -> exp(x) * cos(3z))
            fill_halo_regions!(c)
            d = order_difference(∂zᶜᶜᶠ, (Center, Center, Face), ibg, 4, 2, c)
            bottom = N ÷ 2 + 1
            @test maximum(abs, d[:, :, bottom:bottom+1]) == 0
            @test minimum(abs, d[:, :, bottom+2:N-1]) > 0
        end

        @testset "Open sides are not crossed [$(typeof(arch))]" begin
            N, H = 16, 3
            grid = RectilinearGrid(arch; size=(N, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1), halo=(H, 3, 3), topology=(Open, Periodic, Bounded))
            c = CenterField(grid)
            fill!(parent(c), NaN)
            set!(c, (x, y, z) -> exp(x))
            view(parent(c), H, :, :) .= 1
            view(parent(c), H + N + 1, :, :) .= 1

            for order in (4, 6)
                derivative = Field(KernelFunctionOperation{Face, Center, Center}(∂xᵣᶠᶜᶜ, grid, Val(order), c))
                @test !any(isnan, Array(interior(derivative)))
            end
        end

        @testset "Second order where the spacing is not regular [$(typeof(arch))]" begin
            f(x, y, z) = exp(x / 10) * cos(y / 10 + z / 10)

            function maximum_order_difference(∂, location, grid)
                c = CenterField(grid)
                set!(c, f)
                fill_halo_regions!(c)
                return maximum(abs, order_difference(∂, location, grid, 4, 2, c))
            end

            stretched = RectilinearGrid(arch; size=(8, 8, 8), x=(0, 8), y=(0, 8), z=(0:8) .^ 1.2, halo=(3, 3, 3))
            latlon = LatitudeLongitudeGrid(arch; size=(8, 8, 8), longitude=(0, 8), latitude=(0, 8), z=(0, 8), halo=(3, 3, 3))
            regular = RectilinearGrid(arch; size=(8, 8, 8), extent=(8, 8, 8), halo=(3, 3, 3))
            partial = ImmersedBoundaryGrid(regular, PartialCellBottom(-7.5))

            @test maximum_order_difference(∂zᶜᶜᶠ, (Center, Center, Face), stretched) == 0
            @test maximum_order_difference(∂xᵣᶠᶜᶜ, (Face, Center, Center), stretched) > 0
            @test maximum_order_difference(∂xᵣᶠᶜᶜ, (Face, Center, Center), latlon) == 0
            @test maximum_order_difference(∂zᶜᶜᶠ, (Center, Center, Face), partial) == 0
            @test maximum_order_difference(∂xᵣᶠᶜᶜ, (Face, Center, Center), partial) > 0

            flat = RectilinearGrid(arch; size=(8, 8), y=(0, 8), z=(0, 8), halo=(3, 3), topology=(Flat, Periodic, Bounded))
            c = CenterField(flat)
            set!(c, (y, z) -> cos(y) * z)
            fill_halo_regions!(c)
            @test maximum(abs, Field(KernelFunctionOperation{Center, Center, Center}(∂xᵣᶠᶜᶜ, flat, Val(4), c))) == 0
        end

        @testset "Mirror symmetry over an immersed ridge [$(typeof(arch))]" begin
            Nx, Ny, Nz = 32, 32, 12
            underlying = RectilinearGrid(arch; size=(Nx, Ny, Nz), x=(0, 2π), y=(0, 2π), z=(-1, 0), halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))
            grid = ImmersedBoundaryGrid(underlying, GridFittedBottom((x, y) -> -1 + 0.8 * exp(-(x - π)^2 - (y - π)^2)))
            u, v, w = XFaceField(grid), YFaceField(grid), ZFaceField(grid)
            set!(v, (x, y, z) -> sin(y - π) * cos(3x) * sin(3z))
            τ₂ = Array(interior(viscous_stress_divergence(u, v, w, ScalarDiffusivity(ν=0.3, stencil_order=4))[2]))
            mirrored_faces = [mod1(Ny + 2 - j, Ny) for j in 1:Ny]
            @test maximum(abs, τ₂ .+ τ₂[:, mirrored_faces, :]) ≤ 1e-12 * maximum(abs, τ₂)
        end

        @testset "Single precision and constants [$(typeof(arch))]" begin
            grid = RectilinearGrid(arch, Float32; size=(8, 8, 8), extent=(1, 1, 1), halo=(3, 3, 3))
            c = CenterField(grid)
            set!(c, 290)
            fill_halo_regions!(c)
            derivative = Field(KernelFunctionOperation{Face, Center, Center}(∂xᵣᶠᶜᶜ, grid, Val(4), c))
            @test eltype(derivative) == Float32
            @test maximum(abs, derivative) == 0
        end

        @testset "High-order interpolation [$(typeof(arch))]" begin
            grid = RectilinearGrid(arch; size=(8, 8, 8), x=(0, 8), y=(0, 8), z=(0, 8), halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))
            c, u = CenterField(grid), XFaceField(grid)
            set!(c, (x, y, z) -> x)
            set!(u, (x, y, z) -> x)
            at_faces = Field(KernelFunctionOperation{Face, Center, Center}(ℑxᶠᵃᵃ, grid, Val(4), c))
            @test Array(interior(at_faces, 3:6, :, :)) ≈ Array(interior(u, 3:6, :, :))
        end

        @testset "Bounded walls [$(typeof(arch))]" begin
            max₂, l1₂ = bounded_tracer_laplacian_errors(arch, 2)
            max₄, l1₄ = bounded_tracer_laplacian_errors(arch, 4)
            # The fallback next to walls is second order
            @test convergence_rate(max₄, Ns) > 1.9
            @test convergence_rate(l1₄, Ns) > 2.9
            @test all(max₄ .≤ max₂) # the largest errors are next to the walls, where both are second order
            @test all(l1₄ .< l1₂)

            # Conservation with no-flux walls
            grid = RectilinearGrid(arch; size=(16, 16, 16), extent=(1, 1, 1), halo=(3, 3, 3),
                                   topology=(Bounded, Bounded, Bounded))
            model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-2, κ=1e-2, stencil_order=4), tracers=:c)
            set!(model, c=(x, y, z) -> exp(-50 * ((x - 0.1)^2 + (y - 0.9)^2 + (z - 0.2)^2)),
                        u=(x, y, z) -> 1e-2 * randn(), v=(x, y, z) -> 1e-2 * randn())
            C₀ = sum(model.tracers.c)
            for _ in 1:10
                time_step!(model, 1e-3)
            end
            @test sum(model.tracers.c) ≈ C₀ rtol=1e-12
        end

        @testset "Immersed boundary [$(typeof(arch))]" begin
            # A flat GridFittedBottom at z = 1/2 must be equivalent to a Bounded domain z ∈ (1/2, 1)
            Nx, Nz = 16, 16
            underlying = RectilinearGrid(arch; size=(Nx, 4, Nz), x=(0, 1), y=(0, 1), z=(0, 1), halo=(3, 3, 3),
                                         topology=(Periodic, Periodic, Bounded))
            immersed = ImmersedBoundaryGrid(underlying, GridFittedBottom(0.5))
            truncated = RectilinearGrid(arch; size=(Nx, 4, Nz÷2), x=(0, 1), y=(0, 1), z=(0.5, 1), halo=(3, 3, 3),
                                        topology=(Periodic, Periodic, Bounded))

            c₀(x, y, z) = sin(2π * x) * cos(4π * z) + z^2
            closure = ScalarDiffusivity(ν=0.1, κ=0.1, stencil_order=4)

            cᴵ, cᵀ = CenterField(immersed), CenterField(truncated)
            set!(cᴵ, c₀); set!(cᵀ, c₀)
            qᴵ = tracer_flux_divergence(cᴵ, closure)
            qᵀ = tracer_flux_divergence(cᵀ, closure)
            above(f) = interior(f, :, :, Nz÷2+1:Nz)
            @test maximum(abs, above(qᴵ) .- interior(qᵀ)) ≤ 1e-12 * maximum(abs, qᵀ)

            # Momentum on the immersed grid: tangential velocity sees fourth-order stencils away from the bottom
            uᴵ, vᴵ, wᴵ = XFaceField(immersed), YFaceField(immersed), ZFaceField(immersed)
            uᵀ, vᵀ, wᵀ = XFaceField(truncated), YFaceField(truncated), ZFaceField(truncated)
            set!(uᴵ, c₀); set!(uᵀ, c₀)
            τᴵ = viscous_stress_divergence(uᴵ, vᴵ, wᴵ, closure)
            τᵀ = viscous_stress_divergence(uᵀ, vᵀ, wᵀ, closure)
            @test maximum(abs, above(τᴵ[1]) .- interior(τᵀ[1])) ≤ 1e-12 * maximum(abs, τᵀ[1])

            # Time stepping conserves tracer and produces no NaN
            model = @test_logs (:warn, r"^The FFT-based pressure_solver") NonhydrostaticModel(immersed; closure, tracers=:c)
            set!(model, c=c₀)
            C₀ = sum(model.tracers.c) # reductions on immersed grids exclude immersed cells
            for _ in 1:10
                time_step!(model, 1e-4)
            end
            @test !isnan(maximum(abs, model.tracers.c))
            @test sum(model.tracers.c) ≈ C₀ rtol=1e-12
        end

        @testset "Decay of a sine wave [$(typeof(arch))]" begin
            # Time-step the diffusion of c = sin(kx) with a small time step so that the error is spatial.
            function decay_error(N, order)
                grid = RectilinearGrid(arch; size=N, x=(0, L), halo=3, topology=(Periodic, Flat, Flat))
                κ, k, T = 1.0, 3, 0.1
                model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(κ=κ, stencil_order=order), tracers=:c)
                set!(model, c=x -> sin(k * x))
                simulation = Simulation(model; Δt=1e-5, stop_time=T, verbose=false)
                run!(simulation)
                exact = CenterField(grid)
                set!(exact, x -> exp(-κ * k^2 * T) * sin(k * x))
                return max_error(model.tracers.c, exact)
            end
            e₂ = [decay_error(N, 2) for N in (16, 32)]
            e₄ = [decay_error(N, 4) for N in (16, 32)]
            @test 1.9 < log2(e₂[1] / e₂[2]) < 2.1
            @test 3.7 < log2(e₄[1] / e₄[2]) < 4.3
        end
    end
end
