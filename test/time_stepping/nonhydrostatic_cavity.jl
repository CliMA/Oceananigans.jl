include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.ImmersedBoundaries: GridFittedCavity, PartialCellCavity, CavityLoad, mask_immersed_field!
using Oceananigans.Models.NonhydrostaticModels: CavityFreeSurfacePoissonSolver, correct_surface_vertical_velocity!
using Oceananigans.Operators: divᶜᶜᶜ, Azᶜᶜᶜ
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

cavity_ceiling(x) = min(0, -0.93 + 1.6x)

function cavity_grid(arch, FT, Cavity; ice_load=nothing)
    underlying_grid = RectilinearGrid(arch, FT, size=(16, 20), x=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Flat, Bounded))
    return ImmersedBoundaryGrid(underlying_grid, Cavity(-1, cavity_ceiling; ice_load))
end

cavity_pressure_solver(grid) = ConjugateGradientPoissonSolver(grid; reltol=100eps(eltype(grid)), abstol=0, maxiter=10_000)

function cavity_model(grid, free_surface)
    return NonhydrostaticModel(grid; free_surface, buoyancy=BuoyancyTracer(), tracers=:b,
                               pressure_solver=cavity_pressure_solver(grid))
end

# The top face of an ice-covered column is masked after the step, so it is restored from the solver
function velocity_divergence(model)
    w = deepcopy(model.velocities.w)
    solver = model.pressure_solver
    solver isa CavityFreeSurfacePoissonSolver && correct_surface_vertical_velocity!(solver, w, model.pressures.pNHS)
    U = (model.velocities.u, model.velocities.v, w)
    grid = model.grid
    div = Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, grid, U...))
    compute!(div)
    mask_immersed_field!(div)
    return Array(interior(div))
end

function test_nonhydrostatic_cavity_rest_state(FT, arch, Cavity, free_surface)
    ice_load = CavityLoad(BuoyancyTracer(), (; b = (x, z) -> z / 2))
    grid = cavity_grid(arch, FT, Cavity; ice_load)
    model = cavity_model(grid, free_surface)
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

function test_nonhydrostatic_cavity_divergence(FT, arch, Cavity, free_surface)
    grid = cavity_grid(arch, FT, Cavity)
    model = cavity_model(grid, free_surface)
    set!(model, u=(x, z) -> sin(π * x) * (1 + z), b=(x, z) -> x / 10)

    for _ in 1:3
        time_step!(model, 0.01)
    end

    @test maximum(abs, velocity_divergence(model)) ≤ 1000 * eps(FT)
    @test all(isfinite, Array(interior(model.velocities.u)))

    return nothing
end

# Without ice, the cavity solver matches the Fourier-tridiagonal free-surface solver
function test_open_cavity_free_surface(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(16, 8), x=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Flat, Bounded))
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-1, 1))

    free_surface = ImplicitFreeSurface(gravitational_acceleration=1)
    model = cavity_model(grid, free_surface)
    reference = NonhydrostaticModel(underlying_grid; free_surface, buoyancy=BuoyancyTracer(), tracers=:b)
    @test model.pressure_solver isa CavityFreeSurfacePoissonSolver

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

# A wave in the open ocean displaces the surface beneath the ice and conserves volume
function test_cavity_free_surface_wave(FT, arch, Cavity)
    grid = cavity_grid(arch, FT, Cavity)
    model = cavity_model(grid, ImplicitFreeSurface(gravitational_acceleration=1))

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
    ice_covered = cavity_ceiling.(Array(xnodes(grid, Center()))) .< -1 / 20
    @test all(isfinite, ηᵢ)
    @test maximum(abs, ηᵢ[ice_covered]) > 1e-5
    @test isapprox(volume(η), V₀; atol=100 * eps(FT) * abs(V₀))
    @test maximum(abs, velocity_divergence(model)) ≤ 1000 * eps(FT)

    return nothing
end

function test_cavity_free_surface_needs_conjugate_gradient(arch)
    grid = cavity_grid(arch, Float64, GridFittedCavity)
    pressure_solver = FFTBasedPoissonSolver(grid.underlying_grid)
    @test_throws ArgumentError NonhydrostaticModel(grid; pressure_solver, free_surface=ImplicitFreeSurface())

    return nothing
end

@testset "Nonhydrostatic cavity" begin
    for arch in archs, FT in float_types
        @info "  Testing nonhydrostatic cavity [$FT, $(typeof(arch))]..."
        for Cavity in (GridFittedCavity, PartialCellCavity), free_surface in (nothing, ImplicitFreeSurface(gravitational_acceleration=1))
            label = "$Cavity, $(isnothing(free_surface) ? "rigid lid" : "free surface") [$FT, $(typeof(arch))]"
            @testset "Rest state, $label" test_nonhydrostatic_cavity_rest_state(FT, arch, Cavity, free_surface)
            @testset "Divergence, $label" test_nonhydrostatic_cavity_divergence(FT, arch, Cavity, free_surface)
        end
        @testset "Open cavity free surface [$FT, $(typeof(arch))]" test_open_cavity_free_surface(FT, arch)
        for Cavity in (GridFittedCavity, PartialCellCavity)
            @testset "Free-surface wave, $Cavity [$FT, $(typeof(arch))]" test_cavity_free_surface_wave(FT, arch, Cavity)
        end
    end
    @testset "Cavity free surface needs a conjugate-gradient solver" test_cavity_free_surface_needs_conjugate_gradient(first(archs))
end
