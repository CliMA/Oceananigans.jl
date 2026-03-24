using Oceananigans.Models.NonhydrostaticModels: BackgroundField, BackgroundFields

N² = 1e-6
@inline linear_stratification(z, t, p) = p.N² * z
background_b = BackgroundField(linear_stratification, parameters=(; N²))

function total_buoyancy_with_background_closure_fluxes(arch)
    grid = RectilinearGrid(arch, size=10, z=(0, 1), topology=(Flat, Flat, Bounded))

    background_fields = BackgroundFields(; background_closure_fluxes=true, b=background_b)

    # With background closure fluxes, the boundary conditions are on the perturbation b.
    # Total buoyancy B = B̄ + b, where B̄(z) = N² z.
    # For zero total flux at the bottom: ∂B/∂z = 0 → ∂b/∂z = -N²
    # For ∂B/∂z = N² at the top: ∂b/∂z = 0
    b_bcs = FieldBoundaryConditions(
        bottom = GradientBoundaryCondition(-N²),
        top    = GradientBoundaryCondition(0.0),
    )

    model = NonhydrostaticModel(grid; background_fields, tracers=:b,
                                buoyancy=BuoyancyTracer(),
                                boundary_conditions=(; b=b_bcs))

    simulation = Simulation(model, Δt=0.1, stop_iteration=5)
    run!(simulation)

    B̄ = model.background_fields.tracers.b
    b = model.tracers.b
    return interior(compute!(Field(B̄ + b)))
end

function total_buoyancy_without_background_fields(arch)
    grid = RectilinearGrid(arch, size=10, z=(0, 1), topology=(Flat, Flat, Bounded))

    # Without background fields, the boundary conditions are directly on B.
    # Zero total flux at bottom: ∂B/∂z = 0
    # ∂B/∂z = N² at the top
    B_bcs = FieldBoundaryConditions(
        bottom = GradientBoundaryCondition(0),
        top    = GradientBoundaryCondition(N²),
    )

    model = NonhydrostaticModel(grid; tracers=:b, buoyancy=BuoyancyTracer(),
                                boundary_conditions=(; b=B_bcs))

    Bᵢ(z) = linear_stratification(z, 0, (; N²))
    set!(model, b=Bᵢ)

    simulation = Simulation(model, Δt=0.1, stop_iteration=5)
    run!(simulation)

    return interior(model.tracers.b)
end

@testset "Background closure flux divergence" begin
    for arch in archs
        @info "  Testing background closure flux divergence [$(typeof(arch))]..."
        B_with    = total_buoyancy_with_background_closure_fluxes(arch)
        B_without = total_buoyancy_without_background_fields(arch)
        @test all(isapprox.(B_with, B_without, rtol=1e-10))
    end
end
