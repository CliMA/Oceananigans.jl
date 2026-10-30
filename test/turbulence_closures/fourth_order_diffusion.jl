include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ, with_tracers, stencil_order,
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
    closure = ScalarDiffusivity(κ=κ, order=order)
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
    closure = ScalarDiffusivity(ν=ν, order=order)
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
    closure = ScalarDiffusivity(κ=κ, order=order)
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

@testset "Fourth-order ScalarDiffusivity" begin
    @info "Testing fourth-order ScalarDiffusivity..."

    @testset "Construction" begin
        closure = ScalarDiffusivity(ν=1, κ=2, order=4)
        @test stencil_order(closure) === Val(4)
        @test stencil_order(ScalarDiffusivity(ν=1, κ=2)) === Val(2)
        @test required_halo_size_x(closure) == 2
        @test required_halo_size_y(closure) == 2
        @test required_halo_size_z(closure) == 2
        @test required_halo_size_x(ScalarDiffusivity(ν=1, κ=2)) == 1
        @test summary(closure) == "ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0, κ=2.0, order=4)"
        @test stencil_order(with_tracers((:c,), closure)) === Val(4)
        @test stencil_order(VerticalScalarDiffusivity(κ=1, order=4)) === Val(4)

        @test_throws ArgumentError ScalarDiffusivity(ν=1, order=3)
        @test_throws ArgumentError HorizontalScalarDiffusivity(ν=1, order=4)
        @test_throws ArgumentError VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1, order=4)

        # The largest eigenvalue of the fourth-order Laplacian is 4/3 that of the second-order one
        grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid; closure, tracers=:c)
        @test cell_diffusion_timescale(model) ≈ 3/4 * (1/4)^2 / 2
    end

    for arch in archs
        @testset "Periodic convergence [$(typeof(arch))]" begin
            e₂ = periodic_tracer_laplacian_errors(arch, 2)
            e₄ = periodic_tracer_laplacian_errors(arch, 4)
            r₂, r₄ = convergence_rate(e₂, Ns), convergence_rate(e₄, Ns)
            @info "  Tracer Laplacian errors for N = $Ns: second order $e₂ (rate $r₂), fourth order $e₄ (rate $r₄)"
            @test 1.9 < r₂ < 2.1
            @test 3.9 < r₄ < 4.1

            e₂ = periodic_viscous_errors(arch, 2)
            e₄ = periodic_viscous_errors(arch, 4)
            r₂, r₄ = convergence_rate(e₂, Ns), convergence_rate(e₄, Ns)
            @info "  Viscous term errors for N = $Ns: second order $e₂ (rate $r₂), fourth order $e₄ (rate $r₄)"
            @test 1.9 < r₂ < 2.1
            @test 3.9 < r₄ < 4.1
        end

        @testset "Bounded walls [$(typeof(arch))]" begin
            max₂, l1₂ = bounded_tracer_laplacian_errors(arch, 2)
            max₄, l1₄ = bounded_tracer_laplacian_errors(arch, 4)
            @info "  Bounded Laplacian max errors: second order $max₂ (rate $(convergence_rate(max₂, Ns))), " *
                  "fourth order $max₄ (rate $(convergence_rate(max₄, Ns)))"
            @info "  Bounded Laplacian L1 errors: second order $l1₂ (rate $(convergence_rate(l1₂, Ns))), " *
                  "fourth order $l1₄ (rate $(convergence_rate(l1₄, Ns)))"
            # The fallback next to walls is second order
            @test convergence_rate(max₄, Ns) > 1.9
            @test convergence_rate(l1₄, Ns) > 2.9
            @test all(max₄ .≤ max₂) # the largest errors are next to the walls, where both are second order
            @test all(l1₄ .< l1₂)

            # Conservation with no-flux walls
            grid = RectilinearGrid(arch; size=(16, 16, 16), extent=(1, 1, 1), halo=(3, 3, 3),
                                   topology=(Bounded, Bounded, Bounded))
            model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-2, κ=1e-2, order=4), tracers=:c)
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
            closure = ScalarDiffusivity(ν=0.1, κ=0.1, order=4)

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
            model = NonhydrostaticModel(immersed; closure, tracers=:c)
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
                model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(κ=κ, order=order), tracers=:c)
                set!(model, c=x -> sin(k * x))
                simulation = Simulation(model; Δt=1e-5, stop_time=T, verbose=false)
                run!(simulation)
                exact = CenterField(grid)
                set!(exact, x -> exp(-κ * k^2 * T) * sin(k * x))
                return max_error(model.tracers.c, exact)
            end
            e₂ = [decay_error(N, 2) for N in (16, 32)]
            e₄ = [decay_error(N, 4) for N in (16, 32)]
            @info "  Decay errors for N = (16, 32): second order $e₂, fourth order $e₄"
            @test 1.9 < log2(e₂[1] / e₂[2]) < 2.1
            @test 3.7 < log2(e₄[1] / e₄[2]) < 4.3
        end
    end
end
