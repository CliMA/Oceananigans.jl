include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.ImmersedBoundaries: TopLoad, mask_immersed_field!
using Oceananigans.Models.NonhydrostaticModels: ImmersedTopFreeSurfacePoissonSolver, correct_surface_vertical_velocity!
using Oceananigans.Operators: divᶜᶜᶜ, Azᶜᶜᶜ
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

sloping_top_height(x) = min(0, -0.93 + 1.6x)

function immersed_top_grid(arch, FT, ImmersedBoundary; top_load=nothing)
    underlying_grid = RectilinearGrid(arch, FT, size=(16, 20), x=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Flat, Bounded))
    return ImmersedBoundaryGrid(underlying_grid, ImmersedBoundary(-1; top_height=sloping_top_height, top_load))
end

immersed_top_pressure_solver(grid) = ConjugateGradientPoissonSolver(grid; reltol=100eps(eltype(grid)), abstol=0, maxiter=10_000)

function immersed_top_model(grid, free_surface)
    return NonhydrostaticModel(grid; free_surface, buoyancy=BuoyancyTracer(), tracers=:b,
                               pressure_solver=immersed_top_pressure_solver(grid))
end

restore_surface_vertical_velocity!(w, solver, p) = nothing
restore_surface_vertical_velocity!(w, solver::ImmersedTopFreeSurfacePoissonSolver, p) = correct_surface_vertical_velocity!(solver, w, p)

# The top face of a top-covered column is masked after the step, so it is restored from the solver
function velocity_divergence(model)
    w = deepcopy(model.velocities.w)
    restore_surface_vertical_velocity!(w, model.pressure_solver, model.pressures.pNHS)
    U = (model.velocities.u, model.velocities.v, w)
    grid = model.grid
    div = Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, grid, U...))
    compute!(div)
    mask_immersed_field!(div)
    return Array(interior(div))
end

function test_nonhydrostatic_immersed_top_rest_state(FT, arch, ImmersedBoundary, free_surface)
    top_load = TopLoad(BuoyancyTracer(), (; b = (x, z) -> z / 2))
    grid = immersed_top_grid(arch, FT, ImmersedBoundary; top_load)
    model = immersed_top_model(grid, free_surface)
    set!(model, b=(x, z) -> z / 2)

    for _ in 1:10
        time_step!(model, 0.01)
    end

    tol = 100 * eps(FT)
    @test maximum(abs, interior(model.velocities.u)) ≤ tol
    @test maximum(abs, interior(model.velocities.w)) ≤ tol
    isnothing(free_surface) || @test maximum(abs, interior(model.free_surface.displacement)) ≤ tol

    return nothing
end

function test_nonhydrostatic_immersed_top_divergence(FT, arch, ImmersedBoundary, free_surface)
    grid = immersed_top_grid(arch, FT, ImmersedBoundary)
    model = immersed_top_model(grid, free_surface)
    set!(model, u=(x, z) -> sin(π * x) * (1 + z), b=(x, z) -> x / 10)

    for _ in 1:3
        time_step!(model, 0.01)
    end

    @test maximum(abs, velocity_divergence(model)) ≤ 1000 * eps(FT)
    @test all(isfinite, Array(interior(model.velocities.u)))

    return nothing
end

# Without an immersed top, the solver matches the Fourier-tridiagonal free-surface solver
function test_open_top_free_surface(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(16, 8), x=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Flat, Bounded))
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-1; top_height=1))

    free_surface = ImplicitFreeSurface(gravitational_acceleration=1)
    model = immersed_top_model(grid, free_surface)
    reference = NonhydrostaticModel(underlying_grid; free_surface, buoyancy=BuoyancyTracer(), tracers=:b)
    @test model.pressure_solver isa ImmersedTopFreeSurfacePoissonSolver

    for m in (model, reference)
        set!(m, u=(x, z) -> sin(π * x) * (1 + z) / 10)
    end

    for _ in 1:5
        time_step!(model, 0.01)
        time_step!(reference, 0.01)
    end

    η  = Array(interior(model.free_surface.displacement))
    ηʳ = Array(interior(reference.free_surface.displacement))
    @test maximum(abs, ηʳ) > 1e-4
    @test isapprox(η, ηʳ; rtol=sqrt(eps(FT)))
    @test isapprox(Array(interior(model.velocities.u)), Array(interior(reference.velocities.u)); rtol=sqrt(eps(FT)))
    @test isapprox(Array(interior(model.velocities.w)), Array(interior(reference.velocities.w)); rtol=sqrt(eps(FT)))

    return nothing
end

# A wave in the open ocean displaces the surface beneath the immersed top and conserves volume
function test_immersed_top_free_surface_wave(FT, arch, ImmersedBoundary)
    grid = immersed_top_grid(arch, FT, ImmersedBoundary)
    model = immersed_top_model(grid, ImplicitFreeSurface(gravitational_acceleration=1))

    η = model.free_surface.displacement
    set!(η, (x, z) -> exp(-(x - 0.8)^2 / 0.01) / 100)
    area = Field(KernelFunctionOperation{Center, Center, Nothing}(Azᶜᶜᶜ, grid))
    compute!(area)
    volume(η) = sum(Array(interior(area))[:, :, 1] .* Array(interior(η))[:, :, 1])
    V₀ = volume(η)

    for _ in 1:50
        time_step!(model, 0.01)
    end

    ηᵢ = Array(interior(η))[:, 1, 1]
    top_covered = sloping_top_height.(Array(xnodes(grid, Center()))) .< -1 / 20
    @test all(isfinite, ηᵢ)
    @test maximum(abs, ηᵢ[top_covered]) > 1e-5
    @test isapprox(volume(η), V₀; atol=100 * eps(FT) * abs(V₀))
    @test maximum(abs, velocity_divergence(model)) ≤ 1000 * eps(FT)

    return nothing
end

function test_immersed_top_free_surface_needs_conjugate_gradient(arch)
    grid = immersed_top_grid(arch, Float64, GridFittedBottom)
    pressure_solver = FFTBasedPoissonSolver(grid.underlying_grid)
    @test_throws ArgumentError NonhydrostaticModel(grid; pressure_solver, free_surface=ImplicitFreeSurface())

    return nothing
end

@testset "Nonhydrostatic immersed top" begin
    for arch in archs, FT in float_types
        for ImmersedBoundary in (GridFittedBottom, PartialCellBottom), free_surface in (nothing, ImplicitFreeSurface(gravitational_acceleration=1))
            label = "$ImmersedBoundary, $(isnothing(free_surface) ? "rigid lid" : "free surface") [$FT, $(typeof(arch))]"
            @testset "Rest state, $label" test_nonhydrostatic_immersed_top_rest_state(FT, arch, ImmersedBoundary, free_surface)
            @testset "Divergence, $label" test_nonhydrostatic_immersed_top_divergence(FT, arch, ImmersedBoundary, free_surface)
        end
        @testset "Open-top free surface [$FT, $(typeof(arch))]" test_open_top_free_surface(FT, arch)
        for ImmersedBoundary in (GridFittedBottom, PartialCellBottom)
            @testset "Free-surface wave, $ImmersedBoundary [$FT, $(typeof(arch))]" test_immersed_top_free_surface_wave(FT, arch, ImmersedBoundary)
        end
    end
    @testset "Free surface beneath an immersed top needs a conjugate-gradient solver" test_immersed_top_free_surface_needs_conjugate_gradient(first(archs))
end
