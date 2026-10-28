# Two-layer-like zonally re-entrant channel of Jansen, Held, Adcroft & Hallberg (2015),
# "Energy budget-based backscatter in an eddy permitting primitive equation model", Ocean Modelling 94.
#
# Usage:
#
#     ARCH=GPU julia --project jansen_2015_channel.jl <grid spacing in km> [backscatter] [weno]
#
# e.g. `ARCH=GPU julia --project jansen_2015_channel.jl 20 backscatter weno`; omit `ARCH=GPU` to run on CPU.
# `weno` swaps the energy-conserving momentum advection of the paper for WENO. The simulation spins up for
# AVERAGING_YEARS (default 15, as in the paper) and then averages for AVERAGING_YEARS more.

using Oceananigans
using Oceananigans.Units
using Oceananigans: UpdateStateCallsite
using Oceananigans.Advection: EnergyConserving
using Oceananigans.Grids: bottommost_active_node
using Oceananigans.Operators: ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂xᶠᶠᶜ, ∂yᶠᶠᶜ, ℑxyᶜᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑyᵃᶠᵃ,
                             Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶠᶜᶜ, Δzᶜᶠᶜ
using Printf

if get(ENV, "ARCH", "CPU") == "GPU"
    using CUDA
end

Δ = parse(Float64, ARGS[1]) * kilometers
backscatter = "backscatter" in ARGS
weno = "weno" in ARGS
averaging_time = parse(Float64, get(ENV, "AVERAGING_YEARS", "15")) * 365days

arch = get(ENV, "ARCH", "CPU") == "GPU" ? GPU() : CPU()
name = @sprintf("jansen_2015_%gkm", Δ / kilometers) * (backscatter ? "_backscatter" : "") * (weno ? "_weno" : "")

#####
##### Parameters (Section 2.1)
#####

Lx = Ly = 1600kilometers
H = 2000
Δb = 9.81 * 7e-4             # reduced gravity of the layer interface
τ = 365days                  # zonal-mean interface restoring time scale
Cᴰ = 0.03                    # quadratic bottom drag coefficient
U₀ = 0.1                     # unresolved background bottom velocity
Cˢ = 0.06                    # biharmonic Smagorinsky coefficient

f₀ = 1.1e-4
β = (1.3e-4 - 1.1e-4) / Ly

#####
##### Grid and bathymetry (Fig. 2)
#####

Nx = Ny = round(Int, Lx / Δ)
z = vcat(range(-H, -1000, length=5)[1:end-1], range(-1000, 0, length=21))

underlying_grid = RectilinearGrid(arch; size=(Nx, Ny, length(z) - 1), halo=(5, 5, 4),
                                  x=(0, Lx), y=(0, Ly), z, topology=(Periodic, Bounded, Bounded))

ridge(x, y) = 750 * exp(-(x - 350kilometers)^2 / 2(65kilometers)^2) *
              (1 + tanh((abs(y - 1000kilometers) - 220kilometers) / 50kilometers)) / 2 *
              (1 + tanh((y - 80kilometers) / 40kilometers)) / 2

seamount(x, y) = 540 * exp(-((x - 1000kilometers)^2 + (y - 1000kilometers)^2) / 2(130kilometers)^2)

bottom_height(x, y) = -H + ridge(x, y) + seamount(x, y)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

#####
##### Zonal-mean interface restoring (Fig. 1)
#####

interface_height(y) = -500 - 250 * tanh((y - Ly / 2) / 250kilometers)
two_layer_buoyancy(y, z) = Δb * (1 + tanh((z - interface_height(y)) / 25)) / 2

b★ = Field{Nothing, Center, Center}(grid)
set!(b★, (y, z) -> two_layer_buoyancy(y, z))

b̄ = Field{Nothing, Center, Center}(grid)

@inline zonal_mean_restoring(i, j, k, grid, clock, fields, p) =
    @inbounds - (p.b̄[1, j, k] - p.b★[1, j, k]) / p.τ

b_forcing = Forcing(zonal_mean_restoring, discrete_form=true, parameters=(; b̄, b★, τ))

#####
##### Quadratic bottom drag on the bottom-most active cell, |U| = √(U₀² + u² + v²)
#####

@inline function u_bottom_drag(i, j, k, grid, clock, fields, p)
    u = @inbounds fields.u[i, j, k]
    v = ℑxyᶠᶜᵃ(i, j, k, grid, fields.v)
    drag = - p.Cᴰ * sqrt(p.U₀^2 + u^2 + v^2) * u / Δzᶠᶜᶜ(i, j, k, grid)
    return ifelse(bottommost_active_node(i, j, k, grid, Face(), Center(), Center()), drag, zero(grid))
end

@inline function v_bottom_drag(i, j, k, grid, clock, fields, p)
    u = ℑxyᶜᶠᵃ(i, j, k, grid, fields.u)
    v = @inbounds fields.v[i, j, k]
    drag = - p.Cᴰ * sqrt(p.U₀^2 + u^2 + v^2) * v / Δzᶜᶠᶜ(i, j, k, grid)
    return ifelse(bottommost_active_node(i, j, k, grid, Center(), Face(), Center()), drag, zero(grid))
end

u_forcing = Forcing(u_bottom_drag, discrete_form=true, parameters=(; Cᴰ, U₀))
v_forcing = Forcing(v_bottom_drag, discrete_form=true, parameters=(; Cᴰ, U₀))

#####
##### Biharmonic Smagorinsky viscosity ν₄ = Cˢ Δ⁴ |D| (Eq. 1)
#####

@inline shear_strainᶠᶠᶜ(i, j, k, grid, u, v) = ∂yᶠᶠᶜ(i, j, k, grid, u) + ∂xᶠᶠᶜ(i, j, k, grid, v)

@inline function biharmonic_smagorinsky_viscosity(i, j, k, grid, clock, fields, p)
    tension = ∂xᶜᶜᶜ(i, j, k, grid, fields.u) - ∂yᶜᶜᶜ(i, j, k, grid, fields.v)
    shear = ℑxyᶜᶜᵃ(i, j, k, grid, shear_strainᶠᶠᶜ, fields.u, fields.v)
    Δ⁴ = (Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid))^2
    return p.Cˢ * Δ⁴ * sqrt(tension^2 + shear^2)
end

biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=biharmonic_smagorinsky_viscosity, discrete_form=true,
                                                             loc=(Center, Center, Center), parameters=(; Cˢ))

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1)

closure = backscatter ? (biharmonic_viscosity, EnergyBudgetBackscatter(), convective_adjustment) :
                        (biharmonic_viscosity, convective_adjustment)

#####
##### Model
#####

model = HydrostaticFreeSurfaceModel(grid; closure,
                                    momentum_advection = weno ? WENOVectorInvariant(order=5) : VectorInvariant(vorticity_scheme=EnergyConserving()),
                                    tracer_advection = WENO(order=5),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = BetaPlane(; f₀, β),
                                    tracers = :b,
                                    forcing = (b=b_forcing, u=u_forcing, v=v_forcing))

set!(model, b = (x, y, z) -> two_layer_buoyancy(y, z) + 1e-3 * Δb * randn())

#####
##### Simulation
#####

simulation = Simulation(model; Δt = 20minutes * Δ / 20kilometers, stop_time = 2averaging_time)

zonal_mean_buoyancy = Field(Average(model.tracers.b, dims=1), data=b̄.data)
compute!(zonal_mean_buoyancy)
simulation.callbacks[:zonal_mean] = Callback(model -> compute!(zonal_mean_buoyancy), callsite=UpdateStateCallsite())

wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("%s, iteration %d, max|u|: (%.2e, %.2e, %.2e) m s⁻¹, wall time: %s",
                   prettytime(sim), iteration(sim), maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(1e-9 * (time_ns() - wall_time[])))
    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(30days))

#####
##### Output
#####

u, v, w = model.velocities
b = model.tracers.b

@inline upper_layer_v(i, j, k, grid, v, b, interface_buoyancy) = @inbounds v[i, j, k] * (ℑyᵃᶠᵃ(i, j, k, grid, b) > interface_buoyancy)
upper_layer_transport = Integral(KernelFunctionOperation{Center, Face, Center}(upper_layer_v, grid, v, b, Δb / 2), dims=3)

averages = (; u, v, b, uu = u * u, vv = v * v, V₁ = upper_layer_transport)
snapshots = (; U = Integral(u, dims=3), V = Integral(v, dims=3), ζ = Field(∂x(v) - ∂y(u), indices=(:, :, grid.Nz)))

if backscatter
    e = model.closure_fields[2].e
    averages = merge(averages, (; e))
    snapshots = merge(snapshots, (; e))
end

simulation.output_writers[:averages] = JLD2Writer(model, averages; filename = name * "_averages",
                                                  schedule = AveragedTimeInterval(averaging_time, window=averaging_time),
                                                  overwrite_files = true)

simulation.output_writers[:snapshots] = JLD2Writer(model, snapshots; filename = name * "_snapshots",
                                                   schedule = TimeInterval(10days),
                                                   overwrite_files = true)

simulation.output_writers[:checkpointer] = Checkpointer(model; prefix = name * "_checkpoint",
                                                        schedule = TimeInterval(365days),
                                                        cleanup = true)

run!(simulation, pickup = any(startswith(name * "_checkpoint"), readdir(pwd())))
