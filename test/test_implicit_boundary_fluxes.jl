include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans: PrescribedVelocityFields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, CATKEVerticalDiffusivity

#####
##### A single column relaxed by a surface drag flux  J = λ (c_surf − c★).
##### The dimensionless drag number is β = λ Δt / Δz. An explicit flux is unstable for
##### β > 2; the implicit (Patankar) treatment embeds the linear part λ c_surf in the
##### vertical-solver diagonal and is unconditionally stable.
#####

# Explicit form: the whole affine flux is evaluated explicitly.
@inline drag_flux(i, j, grid, clock, fields, p) = @inbounds p.λ * (fields.c[i, j, grid.Nz] - p.c★)

# Implicit-explicit split: explicit part Fₑ = −λ c★, coefficient λ.
@inline drag_explicit_part(i, j, grid, clock, fields, p) = -p.λ * p.c★
@inline drag_coefficient(i, j, grid, clock, fields, p)   =  p.λ

drag_bc(implicit, λ, c★) = implicit ?
    FluxBoundaryCondition(drag_explicit_part; time_discretization=IMEXFluxTimeDiscretization(drag_coefficient), discrete_form=true, parameters=(; λ, c★)) :
    FluxBoundaryCondition(drag_flux; discrete_form=true, parameters=(; λ, c★))

# `closure = :auto` builds a zero-diffusivity vertically-implicit closure for the implicit BC
# (and none otherwise); pass `closure = nothing` to exercise the implicit BC with no closure.
function relaxed_column(arch, Δt, nsteps; implicit, closure=:auto, λ=0.05, c★=1.0, c₀=0.0)
    grid = RectilinearGrid(arch; size=(1, 1, 4), extent=(1, 1, 4), topology=(Periodic, Periodic, Bounded))
    top = drag_bc(implicit, λ, c★)

    actual_closure = closure !== :auto ? closure :
                     implicit ? VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=0) : nothing

    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, buoyancy=nothing, closure=actual_closure,
                                        velocities=PrescribedVelocityFields(),
                                        boundary_conditions=(; c=FieldBoundaryConditions(top=top)))
    set!(model, c=c₀)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end

    cprofile = Array(interior(model.tracers.c))[1, 1, :]
    return (cmax = maximum(abs, filter(isfinite, cprofile); init=0.0), csurf = cprofile[end])
end

@inline mom_drag_u(i, j, grid, clock, fields, p) = @inbounds p.λ * fields.u[i, j, grid.Nz]
@inline mom_drag_v(i, j, grid, clock, fields, p) = @inbounds p.λ * fields.v[i, j, grid.Nz]

# CATKE: the TKE surface flux (∝ u★³) is a *derived* boundary condition that inherits stability
# from the momentum BC, and reads the realized stress through `total_boundary_flux`.
function catke_drag_column(arch, Δt, nsteps; implicit, λ=0.05, u₀=1.0)
    grid = RectilinearGrid(arch; size=4, z=(-4, 0), topology=(Flat, Flat, Bounded))
    if implicit   # drag toward 0: explicit part 0, coefficient λ (β = λ Δt / Δz = 5 at Δt = 100)
        u_top = IMEXFluxBoundaryCondition(0.0, λ)
        v_top = IMEXFluxBoundaryCondition(0.0, λ)
    else
        u_top = FluxBoundaryCondition(mom_drag_u; discrete_form=true, parameters=(; λ))
        v_top = FluxBoundaryCondition(mom_drag_v; discrete_form=true, parameters=(; λ))
    end
    model = HydrostaticFreeSurfaceModel(grid; closure=CATKEVerticalDiffusivity(),
        buoyancy=BuoyancyTracer(), tracers=(:b,), coriolis=nothing, momentum_advection=nothing,
        boundary_conditions=(u=FieldBoundaryConditions(top=u_top), v=FieldBoundaryConditions(top=v_top)))
    set!(model, u=u₀, v=u₀)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    fa(x) = maximum(abs, filter(isfinite, Array(interior(x))); init=0.0)
    return (umax=fa(model.velocities.u), vmax=fa(model.velocities.v), emax=fa(model.tracers.e))
end

# Combined with adaptive implicit vertical advection (AIVA): a field can carry an AIVA advection
# scheme *and* an implicit-explicit flux BC; the diagonal must sum both contributions.
function aiva_drag_column(arch, Δt, nsteps; implicit, λ=0.05, c★=1.0, c₀=0.0)
    grid = RectilinearGrid(arch; size=4, z=(-4, 0), topology=(Flat, Flat, Bounded), halo=3)
    advection = WENO(time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
    top = drag_bc(implicit, λ, c★)
    closure = implicit ? VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=0) : nothing
    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, buoyancy=nothing, closure,
        momentum_advection=nothing, tracer_advection=advection, velocities=PrescribedVelocityFields(),
        boundary_conditions=(; c=FieldBoundaryConditions(top=top)))
    set!(model, c=c₀)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    cprofile = Array(interior(model.tracers.c))[1, 1, :]
    return (cmax = maximum(abs, filter(isfinite, cprofile); init=0.0), csurf = cprofile[end])
end

@testset "Implicit-explicit flux boundary conditions" begin
    for arch in archs
        @testset "Tracer drag [$(typeof(arch))]" begin
            λ, c★ = 0.05, 1.0   # Δz = 1 ⇒ β = λ Δt / Δz: β = 5 at Δt = 100, β = 0.5 at Δt = 10

            explicit_unstable = relaxed_column(arch, 100.0, 8; implicit=false)
            implicit_highΔt   = relaxed_column(arch, 100.0, 8; implicit=true)
            explicit_lowΔt    = relaxed_column(arch, 10.0, 80; implicit=false)

            @test explicit_unstable.cmax > 1e3                 # explicit β = 5 blows up
            @test isfinite(implicit_highΔt.cmax)               # implicit β = 5 stays finite
            @test implicit_highΔt.cmax ≤ 1.01                  # ... and bounded by the relaxation target

            # The high-Δt implicit solution matches the well-resolved low-Δt explicit reference.
            @test isapprox(implicit_highΔt.csurf, explicit_lowΔt.csurf; atol=1e-3)
            @test isapprox(implicit_highΔt.csurf, c★;                   atol=1e-3)

            # Consistency: in the well-resolved regime (β = 0.5) implicit and explicit agree.
            implicit_lowΔt = relaxed_column(arch, 10.0, 80; implicit=true)
            @test implicit_lowΔt.csurf ≈ explicit_lowΔt.csurf

            # The implicit BC forces the vertical solver to exist even with no implicit closure.
            implicit_no_closure = relaxed_column(arch, 100.0, 8; implicit=true, closure=nothing)
            @test isfinite(implicit_no_closure.cmax)
            @test isapprox(implicit_no_closure.csurf, c★; atol=1e-3)

            # An implicit-explicit flux BC is only valid on vertical boundaries.
            @test_throws ErrorException HydrostaticFreeSurfaceModel(
                RectilinearGrid(arch; size=(1, 1, 4), extent=(1, 1, 4), topology=(Bounded, Periodic, Bounded));
                tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields(),
                boundary_conditions=(; c=FieldBoundaryConditions(east=drag_bc(true, λ, c★))))
        end

        @testset "CATKE momentum drag [$(typeof(arch))]" begin
            # The explosion in u/v propagates to e via u★³; the implicit drag keeps all bounded.
            explicit = catke_drag_column(arch, 100.0, 10; implicit=false)
            implicit = catke_drag_column(arch, 100.0, 10; implicit=true)

            @test explicit.umax > 1e3
            @test explicit.emax > 1e3
            @test isfinite(implicit.umax) && implicit.umax ≤ 1.01
            @test isfinite(implicit.emax)
            @test implicit.emax < 1       # TKE injection stays small once the stress is bounded
            @test implicit.emax > 0       # ... but nonzero: u★ correctly includes the implicit drag
        end

        @testset "AIVA advection + implicit flux BC [$(typeof(arch))]" begin
            c★ = 1.0   # AIVA tracer advection on both; vary only the surface drag BC (β = 5).
            explicit = aiva_drag_column(arch, 100.0, 8; implicit=false)
            implicit = aiva_drag_column(arch, 100.0, 8; implicit=true)

            @test explicit.cmax > 1e3                              # explicit BC blows up even with AIVA
            @test isfinite(implicit.cmax) && implicit.cmax ≤ 1.01  # implicit BC + AIVA stays bounded
            @test isapprox(implicit.csurf, c★; atol=1e-3)          # ... and relaxes to the target
        end
    end
end
