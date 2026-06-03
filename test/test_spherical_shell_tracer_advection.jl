include("dependencies_for_runtests.jl")

using Oceananigans.Advection: WENO, cell_advection_timescale
using Oceananigans.Operators: Vᶜᶜᶜ

# Volume-weighted tracer mass Σ Vᵢⱼₖ cᵢⱼₖ — the quantity a finite-volume flux scheme
# conserves on a closed surface (NOT the unweighted Σc, since cell volumes vary).
function tracer_mass(model)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    c = model.tracers.c
    m = zero(eltype(grid))
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m += Vᶜᶜᶜ(i, j, k, grid) * @allowscalar c[i, j, k]
    end
    return m
end

@testset "OctaHEALPix SphericalShellGrid tracer advection (time stepping)" begin
    FT = Float64
    grid = SphericalShellGrid(CPU(), FT; mapping = OctaHEALPixMapping(8),
                              z = (zero(FT), one(FT)), radius = one(FT), halo = (3, 3, 3))

    model = HydrostaticFreeSurfaceModel(grid; tracers = :c, buoyancy = nothing,
                                        coriolis = nothing, free_surface = nothing,
                                        momentum_advection = nothing,
                                        tracer_advection = WENO(FT; order = 3, bounds = (zero(FT), one(FT))))

    # zonal solid-body rotation (flow runs along the fold, never across it); set!
    # rotates these geographic components into the grid's intrinsic frame.
    half = convert(FT, 1//2)
    set!(model, u = (λ, φ, z) -> cosd(φ), v = (λ, φ, z) -> zero(FT))
    set!(model, c = (λ, φ, z) -> exp(-(acos(clamp(cosd(φ) * cosd(λ), -one(FT), one(FT))) / half)^2 / 2))

    dt = convert(FT, 1//5) * cell_advection_timescale(grid, model.velocities)

    M₀ = tracer_mass(model)

    # Regression for the validate_boundary_condition_topology(::Nothing, ::QuadFolded)
    # ambiguity fix: the first time_step! must run.
    time_step!(model, dt)
    @test !any(isnan, Array(interior(model.tracers.c)))

    for _ in 1:20
        time_step!(model, dt)
    end

    c = Array(interior(model.tracers.c))
    @test !any(isnan, c)

    # Volume-weighted mass is conserved to machine precision for along-fold flow.
    M = tracer_mass(model)
    @test isapprox(M, M₀; rtol = 1e-12)

    # Bounds-preserving WENO should keep the tracer inside its prescribed bounds.
    @test maximum(c) ≤ one(FT) + 1e-6
    @test minimum(c) ≥ -1e-3
end
