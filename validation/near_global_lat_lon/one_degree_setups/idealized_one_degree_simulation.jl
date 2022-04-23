using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Printf
using JLD2
using GLMakie
using SeawaterPolynomials.TEOS10
using Oceananigans.Operators: Δx, Δy

using Oceananigans: fields

arch = CPU()
filename = "near_global_one_degree"

include("one_degree_artifacts.jl")
# bathymetry_path = download_bathymetry() # not needed because we uploaded to repo
bathymetry = jldopen(bathymetry_path)["bathymetry"]

include("one_degree_interface_heights.jl")
z = one_degree_interface_heights()

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (360, 150, 48),
                                        halo = (4, 4, 4),
                                        latitude = (-75, 75),
                                        z,
                                        longitude = (-180, 180),
                                        precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

@show grid

@inline ν₄(i, j, k, grid, lx, ly, lz) = (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 +
                                              1 / Δy(i, j, k, grid, lx, ly, lz)^2))^2 / 1days

biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=ν₄, discrete_form=true)
background_vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-2, κ=1e-4)
dynamic_vertical_diffusivity = RiBasedVerticalDiffusivity()

free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)
equation_of_state = LinearEquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=35.0)

Δh = 100kilometers # 1 degree
νhᵢ = κh = Δh^2 / 1day

function closure_tuple(; νh, κ_skew=0, κ_symmetric=κ_skew)
    gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_skew,
                                                                    κ_symmetric = κ_symmetric,
                                                                    slope_limiter = FluxTapering(1e-2))
    horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh, κ=κh)
    return (horizontal_diffusivity, background_vertical_diffusivity, dynamic_vertical_diffusivity, biharmonic_viscosity)
end

@inline T_reference(φ) = max(-1.0, 30.0 * cos(1.2 * π * φ / 180))
@inline T_relaxation(λ, φ, t, T, tᵣ) = 1 / tᵣ * (T - T_reference(φ))
T_top_bc = FluxBoundaryCondition(T_relaxation, field_dependencies=:T, parameters=30days)
T_bcs = FieldBoundaryConditions(top=T_top_bc)

@inline surface_stress_x(λ, φ, t, p) = p.τ₀ * (1 + exp(-φ^2 / 200)) - (p.τ₀ + p.τˢ) * exp(-(φ + 50)^2 / 200) -
                                                                      (p.τ₀ + p.τᴺ) * exp(-(φ - 50)^2 / 200)

u_top_bc = FluxBoundaryCondition(surface_stress_x, parameters=(τ₀=6e-5, τˢ=2e-4, τᴺ=5e-5))
u_bcs = FieldBoundaryConditions(top=u_top_bc)

model = HydrostaticFreeSurfaceModel(; grid, free_surface, buoyancy,
                                    momentum_advection = VectorInvariant(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    tracers = :T,
                                    closure = closure_tuple(; νh=νhᵢ),
                                    boundary_conditions = (u=u_bcs, T=T_bcs),
                                    tracer_advection = WENO5(grid=underlying_grid))

Tᵢ(λ, φ, z) = T_reference(φ)
set!(model, T=Tᵢ)

simulation = Simulation(model; Δt=20minutes, stop_iteration=1000)

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    u, v, w = sim.model.velocities
    T = sim.model.tracers.T
    η = sim.model.free_surface.η
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)
    ηmax = maximum(abs, η)
    Tmax = maximum(T)
    Tmin = minimum(T)

    msg1 = @sprintf("Iteration: %d, time: %s, wall time: %s", iteration(sim), prettytime(sim), prettytime(elapsed))
    msg2 = @sprintf("├── max(u): (%.2e, %.2e, %.2e) m s⁻¹", umax, vmax, wmax)
    msg3 = @sprintf("├── extrema(T): (%.2f, %.2f) ᵒC", Tmin, Tmax)
    msg4 = @sprintf("└── max|η|: %.2e m", ηmax)

    @info string(msg1, '\n', msg2, '\n', msg3, '\n', msg4)

    wall_clock[] = time_ns()

    return nothing
end

simulation.callbacks[:p] = Callback(progress, IterationInterval(10))

u, v, w = model.velocities
KE = @at (Center, Center, Center) u^2 + v^2

simulation.output_writers[:surface] = JLD2OutputWriter(model, merge(fields(model), (; KE)),
                                                       schedule = IterationInterval(100),
                                                       filename = filename * "_surface.jld2",
                                                       indices = (:, :, grid.Nz), 
                                                       overwrite_existing = true)

run!(simulation)

ut = FieldTimeSeries(filename * "_surface.jld2", "u")
vt = FieldTimeSeries(filename * "_surface.jld2", "v")
Tt = FieldTimeSeries(filename * "_surface.jld2", "T")
Kt = FieldTimeSeries(filename * "_surface.jld2", "KE")
Nt = length(ut.times)

fig = Figure(resolution=(1800, 900))
ax = Axis(fig[1, 1], xlabel="Longitude", ylabel="Latitude", title="Surface kinetic energy")
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value

KEⁿ = @lift interior(Kt[$n], :, :, 1) ./ 2

heatmap!(ax, KEⁿ, colorrange=(0, 1))
    
display(fig)

