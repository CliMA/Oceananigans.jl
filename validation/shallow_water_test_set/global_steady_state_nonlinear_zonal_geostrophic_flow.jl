
using Oceananigans
using Oceananigans.Utils: prettytime
using Oceananigans.Units
using Oceananigans.Operators
using Printf
using Oceananigans.Fields: @compute 
using JLD2

h₀ = 1e3

grid = LatitudeLongitudeGrid(size = (80, 80, 1), longitude = (-180, 180), latitude = (-80, 80), z = (-h₀, 0), halo = (5, 5, 5), precompute_metrics = true)

coriolis = HydrostaticSphericalCoriolis()

free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)

for (adv, scheme) in enumerate([VectorInvariant(), WENO5(vector_invariant=true)])

    model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                        tracers = (),
                                        momentum_advection = scheme, 
                                        buoyancy = nothing,
                                        coriolis = coriolis,
                                        closure = nothing)

    g  = model.free_surface.gravitational_acceleration 
    Ω  = model.coriolis.rotation_rate
    α  = 0.0
    a  = 6.37122e6
    η₀ = 2.94e4 / g 
    u₀ = 2π*a/(12days) 

    uᵢ(λ, φ, z) =   u₀ * (cos(φ)*cos(α) + cos(λ)*sin(φ)*sin(α))
    vᵢ(λ, φ, z) = - u₀ * (sin(φ)*sin(α))
    ηᵢ(λ, φ)    = η₀ -  1/g * (a*Ω*u₀ + u₀^2/2)*(sin(φ)*cos(α) - cos(λ)*cos(φ)*sin(α) )^2
    u, v, w = model.velocities
    η = model.free_surface.η

    set!(u, uᵢ)
    set!(v, vᵢ)
    set!(η, ηᵢ)

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    wave_propagation_time_scale = h₀ * model.grid.Δλᶜᵃᵃ / gravity_wave_speed

    # Time step restricted on the gravity wave speed. If using the implicit free surface method it is possible to increase it
    Δt =  10wave_propagation_time_scale

    simulation = Simulation(model, Δt = Δt, stop_time = 10days)

    progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    u, v, w = model.velocities
    η=model.free_surface.η

    output_fields = (; u = u, v = v, η = η)

    u₁ = Field((Face, Center, Center), grid)
    v₁ = Field((Center, Face, Center), grid)
    η₁ = Field((Center, Center, Nothing), grid)

    set!(u₁, uᵢ)
    set!(v₁, vᵢ)
    set!(η₁, ηᵢ)

    @compute l₁ᵥ = Field((u - u₁)^2 + (v - v₁)^2)
    @compute l₀ᵥ = Field(u₁^2 + v₁^2)

    @compute l₁ₕ = Field((η - η₁)^2)
    @compute l₀ₕ = Field(η₁^2)
    
    using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval

    simulation.output_writers[:fields_full] = JLD2OutputWriter(model, (; lv = l₁ᵥ, l0v = l₀ᵥ, lh = l₁ₕ, l0h = l₀ₕ),
                                                          schedule = TimeInterval(100wave_propagation_time_scale),
                                                          prefix = "rh_$(adv)_diag_",
                                                          force = true)

    run!(simulation)
 end


file1 = jldopen("rh_1_diag_.jld2")
file2 = jldopen("rh_2_diag_.jld2")

iterations = parse.(Int, keys(file1["timeseries/t"]))

d1 = zeros(length(iterations))
d2 = zeros(length(iterations))
h1 = zeros(length(iterations))
h2 = zeros(length(iterations))

for (i, idx) in enumerate(iterations)
    d1[i] = sum(file1["timeseries/lv/$idx"]) / sum(file1["timeseries/l0v/$idx"])
    d2[i] = sum(file2["timeseries/lv/$idx"]) / sum(file2["timeseries/l0v/$idx"])
    h1[i] = sum(file1["timeseries/lh/$idx"]) / sum(file1["timeseries/l0h/$idx"])
    h2[i] = sum(file2["timeseries/lh/$idx"]) / sum(file2["timeseries/l0h/$idx"])
end
