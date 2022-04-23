using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Printf
using JLD2
using GLMakie
using SeawaterPolynomials.TEOS10
using Oceananigans.Operators: Δx, Δy, ζ₃ᶠᶠᶜ

using Oceananigans: fields
using Oceananigans.Grids: ynode

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

#####
##### Closures
#####

include("variable_biharmonic_diffusion_coefficient.jl") # defines VariableBiharmonicDiffusionCoefficient
vitd = VerticallyImplicitTimeDiscretization()
background_vertical_diffusivity = VerticalScalarDiffusivity(vitd, ν=1e-2, κ=1e-4)
dynamic_vertical_diffusivity = RiBasedVerticalDiffusivity()

function closure_tuple(; νh = (100kilometers)^2 / 1day, κ_skew=1e3, κ_symmetric=κ_skew, biharmonic_time_scale=1day)
    ν₄ = VariableBiharmonicDiffusivity(biharmonic_time_scale)
    biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=ν₄, discrete_form=true)

    gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_skew,
                                                                    κ_symmetric = κ_symmetric,
                                                                    slope_limiter = FluxTapering(1e-2))

    horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh, κ=κh)

    return (horizontal_diffusivity,
            background_vertical_diffusivity,
            dynamic_vertical_diffusivity,
            biharmonic_viscosity,
            gent_mcwilliams_diffusivity)
end

free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)
equation_of_state = LinearEquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=35.0)

@inline T_reference(φ) = max(-1.0, 30.0 * cos(1.2 * π * φ / 180))

@inline function T_relaxation(i, j, grid, clock, fields, tᵣ)
    φ = ynode(Center(), j, grid)
    return @inbounds 1 / tᵣ * (fields.T[i, j, grid.Nz] - T_reference(φ))
end

T_top_bc = FluxBoundaryCondition(T_relaxation, discrete_form=true, parameters=30days)
T_bcs = FieldBoundaryConditions(top=T_top_bc)

@inline surface_stress_x(φ, τ₀, τˢ, τᴺ) = τ₀ * (1 - exp(-φ^2 / 200)) - (τ₀ + τˢ) * exp(-(φ + 50)^2 / 200) -
                                                                       (τ₀ + τᴺ) * exp(-(φ - 50)^2 / 200)

@inline function surface_stress_x(i, j, grid, clock, fields, p)
    φ = ynode(Center(), j, grid)
    return surface_stress_x(φ, p.τ₀, p.τˢ, p.τᴺ)
end

u_top_bc = FluxBoundaryCondition(surface_stress_x, discrete_form=true, parameters=(τ₀=6e-5, τˢ=2e-4, τᴺ=5e-5))
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

simulation = Simulation(model; Δt=20minutes, stop_time=30days)

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
ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, computed_dependencies=(u, v))

simulation.output_writers[:surface] = JLD2OutputWriter(model, merge(fields(model), (; KE, ζ)),
                                                       schedule = TimeInterval(3hours),
                                                       filename = filename * "_surface.jld2",
                                                       indices = (:, :, grid.Nz), 
                                                       overwrite_existing = true)

run!(simulation)

ut = FieldTimeSeries(filename * "_surface.jld2", "u")
vt = FieldTimeSeries(filename * "_surface.jld2", "v")
Tt = FieldTimeSeries(filename * "_surface.jld2", "T")
Kt = FieldTimeSeries(filename * "_surface.jld2", "KE")
t = ut.times
Nt = length(t)

fig = Figure(resolution=(1800, 900))
ax = Axis(fig[2, 1], xlabel="Longitude", ylabel="Latitude")
slider = Slider(fig[3, 1], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("Surface kinetic energy at ", prettytime(t[$n]))
Label(fig[1, 1], title, tellwidth=false)

KEⁿ = @lift interior(Kt[$n], :, :, 1) ./ 2

hm = heatmap!(ax, KEⁿ, colorrange=(0, 0.1))
Colorbar(fig[2, 2], hm, label="Surface kinetic energy (m² s⁻²)")
    
display(fig)

record(fig, filename * ".mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end

