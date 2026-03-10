# # Jamart & Ozer (1986) rotating basin
#
# Wind-driven flow in a closed rectangular basin comparing five Coriolis
# discretization schemes. Near immersed boundaries, conventional schemes
# include masked velocity points in the Coriolis interpolation, producing
# spurious depth-averaged boundary currents. The active-weighted schemes
# eliminate this artifact by dividing by the number of wet stencil nodes.
#
# Reference:
#   Jamart & Ozer (1986), J. Geophys. Res., 91(C9), 10621–10631.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: EnstrophyConserving, EnergyConserving
using Oceananigans.Operators: Δz
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization,
                                       AbstractTurbulenceClosure,
                                       implicit_linear_coefficient
using Oceananigans.ImmersedBoundaries: inactive_cell

using Printf
using CairoMakie

#####
##### Physical parameters (Jamart & Ozer 1986, Section 2.1)
#####

Lx = 600kilometers
Ly = 1200kilometers
H  = 100        # depth [m]
f₀ = 1.19e-4    # Coriolis parameter at 55°N [s⁻¹]
τʸ = -0.1       # southward wind stress [N m⁻²]
ρ₀ = 1000.0     # reference density [kg m⁻³]
Aᵥ = 0.065      # vertical eddy viscosity [m² s⁻¹] (650 cm² s⁻¹)

#####
##### Grid
#####

Δ  = 20kilometers
Nz = 20

Nx = Int(Lx / Δ) + 2   # +1 wall cell on each side
Ny = Int(Ly / Δ) + 2

grid = RectilinearGrid(CPU(),
                       topology = (Bounded, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 4),
                       x = (-Δ, Lx + Δ),
                       y = (-Δ, Ly + Δ),
                       z = MutableVerticalDiscretization((-H, 0)))

# Land columns (bottom = 0) form the basin walls
wall_bottom(x, y) = (x < 0 || x > Lx || y < 0 || y > Ly) ? 0.0 : -H

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(wall_bottom))

#####
##### Implicit bottom drag closure
#####
#
# Bottom drag -κu is a linear term that can be absorbed into the diagonal
# of the implicit vertical diffusion tridiagonal solve, making it
# unconditionally stable (no CFL constraint on κ).

struct ImplicitBottomDrag{T} <: AbstractTurbulenceClosure{VerticallyImplicitTimeDiscretization, 1}
    κ :: T
end

# Returns -κ/Δz at the bottom-most active cell, zero elsewhere.
# This adds +Δt κ/Δz to the tridiagonal diagonal → unconditionally stable damping.
@inline function implicit_linear_coefficient(i, j, k, grid, closure::ImplicitBottomDrag,
                                             K, id, ℓx, ℓy, ℓz, Δt, clock, fields)
    at_bottom = inactive_cell(i, j, k - 1, grid) & !inactive_cell(i, j, k, grid)
    return ifelse(at_bottom, -closure.κ / Δz(i, j, k, grid, ℓx, ℓy, ℓz), zero(grid))
end

#####
##### Boundary conditions and closure
#####

κ_drag = Aᵥ / (H / Nz)   # = Aᵥ / Δz ≈ 0.013 m/s (strong, approximates no-slip)

v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τʸ / ρ₀))

closure = (VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=Aᵥ),
           ImplicitBottomDrag(κ_drag))

#####
##### Simulation helper
#####

function run_jamart(grid, scheme; label, Δt=600, stop_time=144hours)
    coriolis     = FPlane(f=f₀, scheme=scheme)
    free_surface = SplitExplicitFreeSurface(grid; substeps=30)

    model = HydrostaticFreeSurfaceModel(grid;
        coriolis,
        closure,
        free_surface,
        momentum_advection  = nothing,
        tracer_advection    = nothing,
        tracers             = (),
        buoyancy            = nothing,
        boundary_conditions = (; v=v_bcs))

    simulation = Simulation(model; Δt, stop_time)

    wall_clock = Ref(time_ns())

    function progress(sim)
        elapsed = (time_ns() - wall_clock[]) * 1e-9
        @info @sprintf("[%s] t=%s, iter=%d, max|u|=%.2e m/s (%.1fs)",
                       label, prettytime(sim.model.clock.time),
                       sim.model.clock.iteration,
                       maximum(abs, sim.model.velocities.u), elapsed)
        wall_clock[] = time_ns()
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))

    @info "Running $label..."
    run!(simulation)

    return model
end

#####
##### Run all five schemes
#####

schemes = [
    (EnstrophyConserving(),                "Enstrophy"),
    (EnergyConserving(),                   "Energy"),
    (EENConserving(),                      "EEN"),
    (ActiveWeightedEnstrophyConserving(),  "AW-Enstrophy"),
    (ActiveWeightedEnergyConserving(),     "AW-Energy"),
]

results = Dict{String, Any}()

for (scheme, label) in schemes
    results[label] = run_jamart(grid, scheme; label)
end

#####
##### Post-processing
#####

function depth_averaged_velocity(model)
    u, v = model.velocities.u, model.velocities.v
    nz = size(u, 3)
    ū = sum(interior(u), dims=3)[:, :, 1] ./ nz
    v̄ = sum(interior(v), dims=3)[:, :, 1] ./ nz
    return ū, v̄
end

# Interior grid coordinates (exclude wall cells)
xc = xnodes(grid, Center())[2:Nx-1] ./ 1e3
yc = ynodes(grid, Center())[2:Ny-1] ./ 1e3

# Subsampled positions for quiver plot
skip = 2
iq = 2:skip:Nx-1
jq = 2:skip:Ny-1
xq = xnodes(grid, Center())[iq] ./ 1e3
yq = ynodes(grid, Center())[jq] ./ 1e3

#####
##### Plot (cf. Jamart Figs. 4 and 6)
#####

fig = Figure(size = (500 * length(schemes), 900))

for (col, (_, label)) in enumerate(schemes)
    model = results[label]
    ū, v̄ = depth_averaged_velocity(model)
    η = interior(model.free_surface.displacement, 2:Nx-1, 2:Ny-1, 1)

    ax = Axis(fig[1, col];
              title = "$label\nη [cm]",
              xlabel = "x [km]",
              ylabel = col == 1 ? "y [km]" : "",
              aspect = DataAspect())

    co = contourf!(ax, xc, yc, η .* 100;
                   colormap=:balance, levels=range(-8, 8, length=33))

    arrows2d!(ax, repeat(xq, 1, length(yq)),
                  repeat(yq', length(xq), 1),
                  ū[iq, jq], v̄[iq, jq];
                  lengthscale=3e4, tipwidth=6, tiplength=6,
                  shaftwidth=0.8, color=:black)

    col == length(schemes) && Colorbar(fig[1, col+1], co, label="η [cm]")

    @info @sprintf("[%s] max|ū|=%.4e, max|v̄|=%.4e m/s", label,
                   maximum(abs, ū), maximum(abs, v̄))
end

save("jamart_basin_comparison.png", fig, px_per_unit=2)
@info "Saved jamart_basin_comparison.png"
