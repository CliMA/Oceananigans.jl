using Oceananigans.Coriolis: hack_sind
using Oceananigans.Units
using Statistics
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Printf
using Random

include("compute_rpe.jl")

@inline transform(φ, p) = (p.φinit + φ) / p.Lφ * 2π - π/2

@inline function bᵢ(λ, φ, z, p) 
    x = transform(φ, p)
    b = ifelse(x < 0, 0, ifelse(x > π, 1, 1 - (π - x - sin(π - x) * cos(π - x)) / π))
    return p.N² * z + p.Δb * b
end

@inline function uᵢ(λ, φ, z, p) 
    f   = 2 * p.coriolis.rotation_rate * hack_sind(φ)
    x   = transform(φ, p)
    ∂b∂x = p.Δb * ifelse(x < 0, 0, ifelse(x > π, 0, (sin(x)^2 - cos(x)^2 + 1) / π))
    ∂x∂φ = 1 / p.Lφ * 2π
    ∂φ∂y = 1 / p.R * 180 / π
    return - 1 / f * (p.Lz + z) * ∂b∂x * ∂x∂φ * ∂φ∂y
end

function baroclinic_adjustment_simulation(resolution, filename, FT::DataType = Float64; 
                                          arch = CPU(), 
                                          horizontal_closure = nothing,
                                          momentum_advection = VectorInvariant(), 
                                          tracer_advection = WENO(FT; order = 7),
                                          background_νz = 1e-4,
                                          φ₀ = - 50,
                                          stop_time = 200days)
    
    # Domain
    Lz = 1kilometers     # depth [m]
    Ny = Base.Int(20 / resolution)
    Nz = 50
    Δt = 10minutes

    grid = LatitudeLongitudeGrid(arch, FT;
                                 topology = (Periodic, Bounded, Bounded),
                                 size = (Ny, Ny, Nz), 
                                 longitude = (-10, 10),
                                 latitude = (φ₀-10, φ₀+10),
                                 z = (-Lz, 0),
                                 halo = (6, 6, 6))
    
    vertical_closure = VerticalScalarDiffusivity(FT; κ = 0, ν = background_νz)

    closures = isnothing(horizontal_closure) ? vertical_closure : (vertical_closure, horizontal_closure)

    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.015 # [m/s²] buoyancy difference

    coriolis = HydrostaticSphericalCoriolis(FT)

    # Parameters
    param = (; N², Δb, Lz, Lφ = grid.Ly, 
               φinit = - (φ₀-10),
               coriolis,
               R = grid.radius)

    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.75)
    @info "Building a model..."

    buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState())

    model = HydrostaticFreeSurfaceModel(; grid,
                                          coriolis,
                                          buoyancy = buoyancy,
                                          closure = closures,
                                          tracers = (:T, :S),
                                          momentum_advection,
                                          tracer_advection,
                                          free_surface)

    ϵb = 1e-3 * Δb # noise amplitude
    Random.seed!(1234)

    Tᵢᵣ(x, y, z) = (bᵢ(x, y, z, param) + ϵb * randn()) / 2e-4 / 10 
    uᵢᵣ(x, y, z) = uᵢ(x, y, z, param)
    Sᵢᵣ(x, y, z) = 32.5 - (grid.Lz + z) / grid.Lz * 0.2

    set!(model, T=Tᵢᵣ, S=Sᵢᵣ)

    #####
    ##### Simulation building
    #####

    simulation = Simulation(model; Δt, stop_time)

    # add progress callback
    wall_clock = [time_ns()]

    RPE = Field(RPEDensityOperation(grid, tracers = model.tracers, buoyancy = model.buoyancy, reference_density = 1020.0))
    compute!(RPE)

    function progress(sim)
        u  = sim.model.velocities.u
        T  = sim.model.tracers.T

        wtime = prettytime(1e-9 * (time_ns() - wall_clock[1]))

        compute!(RPE)
        msg0 = @sprintf("Time: %s wall time: %s, ", prettytime(sim.model.clock.time), wtime)
        msg2 = @sprintf("extrema u: %.2e %.2e ",  maximum(u),  minimum(u))
        msg3 = @sprintf("extrema Δz: %.2e %.2e ", maximum(T),  minimum(T))
        msg4 = @sprintf("total RPE: %6.3e ", total_RPE(RPE))
        @info msg0 * msg2 * msg3 * msg4
    
        wall_clock[1] = time_ns()

        return nothing
    end
    
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
    
    outputs = merge(model.velocities, model.tracers, (; RPE))

    simulation.output_writers[:outputs] = JLD2OutputWriter(model, outputs;
                                                           overwrite_existing = true,
                                                           schedule = TimeInterval(10days),
                                                           filename)

    return simulation
end