include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Models.HydrostaticFreeSurfaceModels: TracerTendencyCallsite
using Oceananigans.TurbulenceClosures: shear_production
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: explicit_buoyancy_flux

# Discrete energy budget of CATKE's shear production and buoyancy flux in a single column.
#
# For one time step, with φ* the state before the vertically-implicit solve and φ⁺ the state after it,
#
#   loss of mean kinetic energy to the solve:  Σ Δzᶜ (u*² + v*² - u⁺² - v⁺²) / 2  =  Δt Σ Δzᶜ P,
#   gain of potential energy in the solve:     - Σ Δzᶜ z (b⁺ - b*)                =  - Δt Σ Δzᶜ wb,
#
# where P and wb are the shear production and buoyancy flux at cell centers in CATKE's TKE equation
# (Burchard 2002, Ocean Modelling 4, 347–361). The diffusive fluxes of the solve vanish at the boundary
# faces (boundary fluxes are explicit), so φ* = φ⁺ - Δt δᶜ(κ δᶠφ⁺) / Δzᶜ is computed from the solved state.

function pre_solve_state(φ⁺, κ, Δz, Δt)
    Nz = length(φ⁺)
    F = zeros(Nz + 1)
    for k in 2:Nz
        F[k] = κ[k] * (φ⁺[k] - φ⁺[k-1]) / Δz
    end
    return φ⁺ .- Δt .* (F[2:Nz+1] .- F[1:Nz]) ./ Δz
end

column_interior(φ) = Array(interior(φ, 1, 1, :))

function catke_column_energy_budget(arch, timestepper; Nz = 32, H = 128, Δt = 600, steps = 6)
    grid = RectilinearGrid(arch, size = Nz, z = (-H, 0), topology = (Flat, Flat, Bounded))
    closure = CATKEVerticalDiffusivity()

    boundary_conditions = (u = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4)),
                           b = FieldBoundaryConditions(top = FluxBoundaryCondition(1e-7)))

    model = HydrostaticFreeSurfaceModel(grid; closure, boundary_conditions, timestepper,
                                        coriolis = FPlane(f = 1e-4),
                                        buoyancy = BuoyancyTracer(), tracers = :b)

    set!(model, b = z -> 1e-5 * z, e = 1e-6)

    # What the TKE step uses, as it stands after the velocities and tracers are stepped
    u⁻ = XFaceField(grid)
    v⁻ = YFaceField(grid)
    κu = ZFaceField(grid)
    κc = ZFaceField(grid)

    function store_tke_step_inputs!(model)
        parent(u⁻) .= parent(model.closure_fields.previous_velocities.u)
        parent(v⁻) .= parent(model.closure_fields.previous_velocities.v)
        parent(κu) .= parent(model.closure_fields.κu)
        parent(κc) .= parent(model.closure_fields.κc)
        return nothing
    end

    callbacks = [Callback(store_tke_step_inputs!; callsite = TracerTendencyCallsite())]

    Δz = H / Nz
    zᶜ = Array(znodes(grid, Center()))
    u, v, w = model.velocities
    b = model.tracers.b

    P = Field(KernelFunctionOperation{Center, Center, Center}(shear_production, grid, κu, u⁻, u, v⁻, v))
    wb = Field(KernelFunctionOperation{Center, Center, Center}(explicit_buoyancy_flux, grid, closure,
                                                               model.velocities, model.tracers,
                                                               model.buoyancy, (; κc)))

    kinetic_energy_residuals = Float64[]
    potential_energy_residuals = Float64[]

    for n in 1:steps
        time_step!(model, Δt; callbacks)

        u⁺ = column_interior(u)
        v⁺ = column_interior(v)
        b⁺ = column_interior(b)
        κuᶠ = Array(interior(κu, 1, 1, :))
        κcᶠ = Array(interior(κc, 1, 1, :))

        u★ = pre_solve_state(u⁺, κuᶠ, Δz, Δt)
        v★ = pre_solve_state(v⁺, κuᶠ, Δz, Δt)
        b★ = pre_solve_state(b⁺, κcᶠ, Δz, Δt)

        kinetic_energy_loss = sum(Δz .* (u★.^2 .+ v★.^2 .- u⁺.^2 .- v⁺.^2) ./ 2)
        potential_energy_gain = - sum(Δz .* zᶜ .* (b⁺ .- b★))

        compute!(P)
        compute!(wb)
        ∫P = sum(Δz .* column_interior(P))
        ∫wb = sum(Δz .* column_interior(wb))

        push!(kinetic_energy_residuals, (kinetic_energy_loss - Δt * ∫P) / kinetic_energy_loss)
        push!(potential_energy_residuals, (potential_energy_gain + Δt * ∫wb) / potential_energy_gain)
    end

    return kinetic_energy_residuals, potential_energy_residuals
end

@testset "CATKE column energy budget [$(summary(arch)), $timestepper]" for arch in archs,
                                                                             timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
    rᴷ, rᴾ = catke_column_energy_budget(arch, timestepper)
    @test maximum(abs, rᴷ) < 1e-12
    @test maximum(abs, rᴾ) < 1e-10
end
