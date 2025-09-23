using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.Grids: yspacings, xspacings, zspacings
using Oceananigans.TurbulenceClosures
using CairoMakie
using CUDA
using Adapt
using Statistics
using JLD2
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxzᶠᵃᶜ, ℑxzᶜᵃᶠ, ℑyzᵃᶠᶜ, ℑyzᵃᶜᶠ, Δzᶜᶜᶜ, yspacing, xspacing, zspacing
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using Oceananigans.TurbulenceClosures: RiBasedVerticalDiffusivity
using Oceananigans.TurbulenceClosures: DiffusiveFormulation, AdvectiveFormulation
using Oceananigans.BuoyancyFormulations: buoyancy_frequency

κ_skew = 1000
κ_symmetric = nothing;

skew_flux_formulation = AdvectiveFormulation()

# struct CubicTapering{FT}
#     a  :: FT
#     b  :: FT
#     c  :: FT
#     d  :: FT
#     S1 :: FT
#     S2 :: FT
# end

# function CubicTapering(S1, S2)
#     A = [ S1^3  S1^2 S1 1;
#           S2^3  S2^2 S2 1;
#          3S1^2 2S1    1 0;
#          3S2^2 2S2    1 0]

#     b = [1, 0, 0, 0]

#     a, b, c, d = A \ b

#     return CubicTapering(a, b, c, d, S1, S2)
# end

# import Oceananigans.TurbulenceClosures: tapering_factor

# function tapering_factor(Sx, Sy, slope_limiter::CubicTapering) 
#     S = sqrt(Sx^2 + Sy^2)
#     a = slope_limiter.a
#     b = slope_limiter.b
#     c = slope_limiter.c
#     d = slope_limiter.d
#     ϵ = a * S^3 + b * S^2 + c * S + d
#     ϵ = ifelse(S < slope_limiter.S1, 1, ϵ)
#     ϵ = ifelse(S > slope_limiter.S2, 0, ϵ)
#     return max(min(ϵ, 1), 0)
# end

tapering_threshold = 5e-3
filename = "abernathey_channel_2D_noTforcing_taper_$(tapering_threshold)"
if skew_flux_formulation isa DiffusiveFormulation
    filename *= "_diffusive_0"
elseif skew_flux_formulation isa AdvectiveFormulation
    filename *= "_advective"
end
filename *= "_$(κ_skew)_$(κ_symmetric)"

data_folder = "./Output"
sim_folder = "$(data_folder)/$(filename)"
mkpath(sim_folder)

timestep = 20minutes

const Ly = 2000kilometers
const Lz = 3kilometers
const Δy = 100kilometers
const ny = Int(Ly / Δy)
const nz = 30
const ρ₀ = 1025

z_faces = ExponentialCoordinate(nz, -3000, 0; scale = 1000)
coriolis = BetaPlane(f₀ = -0.83e-4, β = 1.87e-11) 

buoyancy = SeawaterBuoyancy()

function Tₐ(y)
    return 0 + 8 * (y) / Ly
end

function Tᵢ(y, z)
    h = 1kilometer
    return Tₐ(y) * (exp(z/h) - exp(-Lz/h)) / (1 - exp(-Lz/h)) + 1.0
end

function Sᵢ(y, z)
    return 35
end

function uᵢ(y, z)
    # Pseudorandom initial condition. 
    return 0 #1e-2 * (mod(Base.MathConstants.golden * (y ^ 2 + z ^ 2), 1) - 0.5)
end

function vᵢ(y, z)
    # Rough attempt to add some noise to the velocity field
    return 0 #1e-2 * (mod(Base.MathConstants.golden * (y ^ 2 + z ^ 2), 1) - 0.5)
end

function cᵢ(y, z)
    return 0.5 * (1 - y / Ly) + 0.5 * (z / -Lz)
end

function surface_momentum_flux(i, j, grid, clock, fields)
    y = ynode(j, grid, Center())

    # Add 50km wide buffer zones with zero wind forcing at the North and South boundaries
    function smooth_window(x, x_min, x_max)
        # Returns 1 if x in [x_min, x_max], 0 otherwise, without branching
        return 0.5 * (sign(x - x_min) - sign(x - x_max))
    end

    window = smooth_window(y, 50kilometers, Ly - 50kilometers)

    time = clock.time
    ramp = min(1, time / 20days) # ramp up wind stress over 5 days

    τ = 0.2 * sin(π * (y-(100/2)kilometers)/ (2000-100)kilometers) ^ 2 # N / m^2
    τρ = τ / ρ₀ # [m/s] m / s , flux of velocity.
    
    return - ramp * window * τρ
end

wind_flux_bc = FluxBoundaryCondition(surface_momentum_flux, discrete_form=true)

@inline μ = 2.1e-3 # dimensionless

function u_bottom_drag(i, j, grid, clock, fields)
    drag_coeff = 2.1e-3 # dimensionless
    u = @inbounds fields.u[1, j, 1]
    v = ℑxyᶠᶜᵃ(1, j, 1, grid, fields.v)
    return - drag_coeff * u * sqrt(u^2 + v^2)
end

function v_bottom_drag(i, j, grid, clock, fields)
    drag_coeff = 2.1e-3 # dimensionless
    u = ℑxyᶜᶠᵃ(1, j, 1, grid, fields.u)
    v = @inbounds fields.v[1, j, 1]
    return - drag_coeff * v * sqrt(u^2 + v^2)
end

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form=true)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form=true)

# u_north_wall_drag_bc = FluxBoundaryCondition(u_north_wall_drag, discrete_form=true)
# u_south_wall_drag_bc = FluxBoundaryCondition(u_south_wall_drag, discrete_form=true)

u_bcs = FieldBoundaryConditions(bottom=u_bottom_drag_bc, top=wind_flux_bc)
v_bcs = FieldBoundaryConditions(bottom=v_bottom_drag_bc)

grid = RectilinearGrid(CPU(), 
                       size = (ny, nz),
                       y = (0, Ly),
                       z = z_faces,
                       topology = (Flat, Bounded, Bounded),
                       halo = (7, 7))

free_surface = SplitExplicitFreeSurface(grid; cfl=0.8, fixed_Δt=timestep + 5minutes)
obl_closure  = RiBasedVerticalDiffusivity()
eddy_closure = IsopycnalSkewSymmetricDiffusivity(; κ_skew, κ_symmetric, skew_flux_formulation) #, slope_limiter=CubicTapering(1e-3, 1e-2))
closure = (obl_closure, eddy_closure)

model = HydrostaticFreeSurfaceModel(; grid = grid,
                                      coriolis = coriolis,
                                      buoyancy = buoyancy,
                                      free_surface,
                                      timestepper = :QuasiAdamsBashforth2,
                                      tracers = (:T, :S, :c),
                                      closure = closure,
                                      momentum_advection = WENO(order=5),
                                      tracer_advection = WENO(order=5),
                                      boundary_conditions = (u=u_bcs, v=v_bcs))
                                    #   boundary_conditions = (T=T_bcs, u=u_bcs, v=v_bcs))

set!(model, T=Tᵢ, S=Sᵢ, u=uᵢ, v=vᵢ, c=cᵢ)
simulation = Simulation(model, Δt=25minutes, stop_time= 20 * 365days) #, stop_iteration=10)

using Printf

wall_clock = Ref(time_ns())

function print_progress(sim)
    u, v, w = model.velocities
    ve = model.diffusivity_fields[2].v
    we = model.diffusivity_fields[2].w
    T, S = model.tracers
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(v): (%6.3e, %6.3e) m/s, max(ve): (%6.3e, %6.3e) m/s, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(abs, v), maximum(abs, w), maximum(abs, ve), maximum(abs, we), maximum(abs, T), maximum(abs, S), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, print_progress, IterationInterval(1000))

u, v, w = model.velocities
ve = model.diffusivity_fields[2].v
we = model.diffusivity_fields[2].w
T, S, c = model.tracers
b = BuoyancyField(model)
N² = Field(buoyancy_frequency(model))

outputs = (; u, v, w, T, S, b, N², c, ve, we)

simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              schedule = TimeInterval(365days),
                                              filename = "$(sim_folder)/instantaneous_timeseries",
                                              overwrite_existing = true)

# simulation.output_writers[:yearly_average] = JLD2Writer(model, outputs;
#                                               schedule = AveragedTimeInterval(365days, window=365days),
#                                               filename = "$(sim_folder)/yearly_average",
#                                               overwrite_existing = true)

# simulation.output_writers[:decadal_average] = JLD2Writer(model, outputs;
#                                               schedule = AveragedTimeInterval(3650days, window=3650days),
#                                               filename = "$(sim_folder)/decadal_average",
#                                               overwrite_existing = true)

run!(simulation)
#%%
yC = Array(ynodes(grid, Center())) ./ 1e3
zC = Array(znodes(grid, Center())) ./ 1e3
zF = Array(znodes(grid, Face())) ./ 1e3
#%%
fig = Figure()
ax = Axis(fig[1, 1], title = "buoyancy at t = $(prettytime(simulation.model.clock.time)), $(filename)", xlabel = "x (km)", ylabel = "z (km)")
cf = contourf!(ax, yC, zC, Array(interior(b, 1, :, :)), levels = 10)
Colorbar(fig[1, 2], cf, label = "buoyancy (m/s²)")
save("$(sim_folder)/buoyancy.png", fig)
#%%
fig = Figure()
ax = Axis(fig[1, 1], title = "temperature at t = $(prettytime(simulation.model.clock.time)), $(filename)", xlabel = "y (km)", ylabel = "z (km)")
cf = contourf!(ax, yC, zC, Array(interior(T, 1, :, :)), levels = 10)
Colorbar(fig[1, 2], cf, label = "temperature (ᵒC)")
save("$(sim_folder)/temperature.png", fig)
#%%
fig = Figure()
ax = Axis(fig[1, 1], title = "salinity at t = $(prettytime(simulation.model.clock.time)), $(filename)", xlabel = "y (km)", ylabel = "z (km)")
cf = contourf!(ax, yC, zC, Array(interior(S, 1, :, :)), levels = 10)
Colorbar(fig[1, 2], cf, label = "salinity (psu)")
save("$(sim_folder)/salinity.png", fig)
#%%
fig = Figure()
ax = Axis(fig[1, 1], title = "N² at t = $(prettytime(simulation.model.clock.time)), $(filename)", xlabel = "y (km)", ylabel = "z (km)")
cf = contourf!(ax, yC, zF[2:nz-1], Array(interior(N², 1, :, 2:nz-1)), levels = 20)
Colorbar(fig[1, 2], cf, label = "N² (s⁻²)")
save("$(sim_folder)/N2.png", fig)
#%%
fig = Figure()
ax = Axis(fig[1, 1], title = "c at t = $(prettytime(simulation.model.clock.time)), $(filename)", xlabel = "y (km)", ylabel = "z (km)")
cf = contourf!(ax, yC, zC, Array(interior(c, 1, :, :)), levels = 10)
Colorbar(fig[1, 2], cf, label = "c (nondimensional)")
save("$(sim_folder)/c.png", fig)
#%%

b = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "b")
T = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "T")
S = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "S")
u  = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "u")
v  = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "v")
w  = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "w")
ve = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "ve")
we = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "we")

x, y,  z = nodes(b)
_, yF, _ = nodes(v)
_, _, zF = nodes(w)

iter = Observable(1)
bn  = @lift(interior(b[$iter],  1, :, :))
Tn  = @lift(interior(T[$iter],  1, :, :))
Sn  = @lift(interior(S[$iter],  1, :, :))
un  = @lift(interior(u[$iter],  1, :, :))
vn  = @lift(interior(v[$iter],  1, :, :))
wn  = @lift(interior(w[$iter],  1, :, :))
ven = @lift(interior(ve[$iter], 1, :, :))
wen = @lift(interior(we[$iter], 1, :, :))

maxv = maximum(abs, v)
maxw = maximum(abs, w)

fig = Figure()
ax  = Axis(fig[1, 1])
contourf!(ax, y, z, bn; colormap=:jet, levels=range(extrema(b[1])..., length=10))
ax  = Axis(fig[1, 2])
contourf!(ax, y, z, Tn; colormap=:jet, levels=range(extrema(T[1])..., length=10))
ax  = Axis(fig[2, 1])
contourf!(ax, y, zF, wn; colormap=:jet, levels=range(-maxw, maxw, length=10))
ax  = Axis(fig[2, 2])
contourf!(ax, yF, z, vn; colormap=:jet, levels=range(-maxv, maxv, length=10))
ax  = Axis(fig[2, 3])
contourf!(ax, yF, z, ven; colormap=:jet, levels=range(-maxv, maxv, length=10))
ax  = Axis(fig[2, 4])
contourf!(ax, y, zF, wen; colormap=:jet, levels=range(-maxw, maxw, length=10))


GLMakie.record(fig, "Output/" * filename * "/output.mp4", 1:length(b)) do i
    @info "step $i";
    iter[] = i;
end