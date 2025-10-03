using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyFormulations
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
using Oceananigans.TurbulenceClosures: RiBasedVerticalDiffusivity, AbstractScalarDiffusivity
using Oceananigans.TurbulenceClosures: VerticalFormulation
using Oceananigans.BuoyancyFormulations: buoyancy_frequency

κ_skew = 1000
κ_symmetric = 1000

filename = "file_restoring"

data_folder = "./Output"
sim_folder = "$(data_folder)/$(filename)"
mkpath(sim_folder)

const Ly = 2000kilometers
const Lz = 3kilometers
const Δy = 100kilometers
const ny = Int(Ly / Δy)
const nz = 30
const ρ₀ = 1025

z_faces = ExponentialCoordinate(nz, -3000, 0; scale = 1000)
const coriolis = BetaPlane(f₀ = -0.83e-4, β = 1.87e-11) 
const buoyancy = SeawaterBuoyancy()

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
u_bcs = FieldBoundaryConditions(bottom=u_bottom_drag_bc, top=wind_flux_bc)
v_bcs = FieldBoundaryConditions(bottom=v_bottom_drag_bc)

grid = RectilinearGrid(CPU(), 
                       size = (ny, nz),
                       y = (0, Ly),
                       z = z_faces,
                       topology = (Flat, Bounded, Bounded),
                       halo = (7, 7))

using Oceananigans.TurbulenceClosures: EddyEvolvingStreamfunction, IsopycnalDiffusivity, FluxTapering                       

obl_closure  = RiBasedVerticalDiffusivity()
#redi_closure = IsopycnalSkewSymmetricDiffusivity(; κ_symmetric) 
redi_closure = IsopycnalDiffusivity(; κ_symmetric) 

@inline function my_ν(i, j, k, grid, clock, fields) 
    y = ynode(j, grid, Center())
    f = coriolis.f₀ + coriolis.β * y
    N² = max(1e-20, ∂z_b(i, j, k, grid, buoyancy, fields))
    return min(5.0, 1000 / N² * f^2) # f2 / N2 < 0.01
end

eddy_closure = EddyAdvectiveClosure(; κ_skew) #, tapering=EddyEvolvingStreamfunction(500days)) # EddyAdvectiveClosure(; κ_skew, tapering=FluxTapering(0.01)) #  VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=my_ν, discrete_form=true, loc=(Center(), Center(), Face())) 
closure = (obl_closure, eddy_closure)

model = HydrostaticFreeSurfaceModel(; grid = grid,
                                      coriolis = coriolis,
                                      buoyancy = buoyancy,
                                      free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=25minutes),
                                      timestepper = :QuasiAdamsBashforth2,
                                      tracers = (:T, :S, :c),
                                      closure = closure,
                                      momentum_advection = WENO(order=5),
                                      tracer_advection = WENO(order=5),
                                      boundary_conditions = (u=u_bcs, v=v_bcs))

set!(model, T=Tᵢ, S=Sᵢ, u=uᵢ, v=vᵢ, c=cᵢ)
simulation = Simulation(model, Δt=25minutes, stop_time=20 * 365days) #, stop_iteration=80333)

using Printf

wall_clock = Ref(time_ns())

diff = filter(x -> hasproperty(x, :v), model.diffusivity_fields)
if isempty(diff)
    ve = Oceananigans.Fields.YFaceField(grid)
    we = Oceananigans.Fields.ZFaceField(grid)
else
    ve = diff[1].v
    we = diff[1].w
end

function print_progress(sim)
    u, v, w = model.velocities
    T, S = model.tracers
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(v): (%6.3e, %6.3e) m/s, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(abs, v), maximum(abs, w), maximum(abs, T), maximum(abs, S), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, print_progress, IterationInterval(1000))

u, v, w = model.velocities
T, S, c = model.tracers
b = BuoyancyField(model)
N² = Field(buoyancy_frequency(model))
by = Field(∂y(b))

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

using GLMakie

b  = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "b")
T  = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "T")
S  = FieldTimeSeries("Output/" * filename * "/instantaneous_timeseries.jld2", "S")
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

fig = Figure(fontsize=12, size=(500, 2000))
ax  = Axis(fig[1, 1:2], title = "Buoyancy", xlabel = "y (km)", ylabel = "z (m)")
contourf!(ax, y, z, bn; colormap=:jet, levels=range(extrema(b[1])..., length=20))
xlims!(ax, extrema(y))
ylims!(ax, extrema(z))
ax  = Axis(fig[2, 1], title = "V-velocity", xlabel = "y (km)", ylabel = "z (m)")
contourf!(ax, yF, z, vn; colormap=:jet, levels=range(-maxv, maxv, length=10))
xlims!(ax, extrema(yF))
ylims!(ax, extrema(z))
ax  = Axis(fig[2, 2], title = "W-velocity", xlabel = "y (km)", ylabel = "z (m)")
contourf!(ax, y, zF, wn; colormap=:jet, levels=range(-maxw, maxw, length=10))
xlims!(ax, extrema(y))
ylims!(ax, extrema(zF))
ax  = Axis(fig[3, 1], title = "Eddy V-velocity", xlabel = "y (km)", ylabel = "z (m)")
contourf!(ax, yF, z, ven; colormap=:jet, levels=range(-maxv, maxv, length=10))
xlims!(ax, extrema(yF))
ylims!(ax, extrema(z))
ax  = Axis(fig[3, 2], title = "Eddy W-velocity", xlabel = "y (km)", ylabel = "z (m)")
contourf!(ax, y, zF, wen; colormap=:jet, levels=range(-maxw, maxw, length=10))
xlims!(ax, extrema(y))
ylims!(ax, extrema(zF))

GLMakie.record(fig, "Output/" * filename * "/output.mp4", 1:length(b)) do i
    @info "step $i";
    iter[] = i;
end