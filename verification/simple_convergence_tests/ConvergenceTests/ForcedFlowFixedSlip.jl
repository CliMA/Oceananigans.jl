module ForcedFlowFixedSlip

using Printf

using Oceananigans, Oceananigans.Forcing, Oceananigans.BoundaryConditions, Oceananigans.OutputWriters,
        Oceananigans.Fields

# Functions that define the forced flow problem

 ξ(t) = 1 + sin(t^2)
ξ′(t) = 2t * cos(t^2)

 f(x, t) =   cos(x - ξ(t))
fₓ(x, t) = - sin(x - ξ(t))

 g(y) = y^3 - y^2
g′(y) = 3y^2 - 2y

H₁(y) = 1/4 * y^4 - 1/3 * y^3 - 3y^2
H₂(y) = 3y^4 - 2y^3
H₃(y) = 1/4 * y^4 - 1/3 * y^3 - 6y^2

F₁(x, y, t) =     ξ′(t) * fₓ(x, t) * H₁(y) + 
                f(x, t) * fₓ(x, t) * H₂(y) - 
                f(x, t) * H₃(y) 

F₂(x, y, t) = 3y^5 - 5y^4 + 2y^3

u_xy(x, y, z, t) = f(x, t) * g′(y)
v_xy(x, y, z, t) = - fₓ(x, t) * g(y)

u_xz(x, y, z, t) = f(x, t) * g′(z)
w_xz(x, y, z, t) = - fₓ(x, t) * g(z)

uᵢ_xz(x, y, z) = u_xz(x, y, z, 0)
wᵢ_xz(x, y, z) = w_xz(x, y, z, 0)

uᵢ_xy(x, y, z) = u_xy(x, y, z, 0)
vᵢ_xy(x, y, z) = v_xy(x, y, z, 0)

function setup_xz_simulation(Nx, Nz, CFL)
    u_forcing = SimpleForcing((x, y, z, t) -> F₁(x, z, t))
    w_forcing = SimpleForcing((x, y, z, t) -> F₂(x, z, t))

    grid = RegularCartesianGrid(size=(Nx, 1, Nz), x=(0, 2π), y=(0, 1), z=(0, 1), 
                                topology=(Periodic, Periodic, Bounded))

    u_bcs = UVelocityBoundaryConditions(grid, bottom = BoundaryCondition(Value, 0),
                                              top = UVelocityBoundaryCondition(Value, :z, (x, y, t) -> f(x, t)))

    model = IncompressibleModel(       architecture = CPU(),
                                               grid = grid,
                                           coriolis = nothing,
                                           buoyancy = nothing,
                                            tracers = nothing,
                                            closure = ConstantIsotropicDiffusivity(ν=1),
                                boundary_conditions = (u=u_bcs,),
                                            forcing = ModelForcing(u=u_forcing, w=w_forcing))

    set!(model, u=uᵢ_xz, w=wᵢ_xz)

    h = min(2π/Nx, 1/Nz)
    Δt = h * CFL # Velocity scale = 1
    simulation = Simulation(model, Δt=Δt, stop_time=π, progress_frequency=1)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, FieldOutputs(model.velocities);
                                                          dir = dir, force = true,
                                                          prefix = @sprintf("forced_fixed_slip_xz_Nx%d_Δt%.1e", Nx, Δt),
                                                          interval = stop_iteration * Δt / 2)

    return simulation
end

function setup_xz_simulation(Nx, Ny, CFL)
    u_forcing = SimpleForcing((x, y, z, t) -> F₁(x, z, t))
    v_forcing = SimpleForcing((x, y, z, t) -> F₂(x, z, t))

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, 2π), y=(0, 1), z=(0, 1), 
                                topology=(Periodic, Bounded, Bounded))

    u_bcs = UVelocityBoundaryConditions(grid, south = BoundaryCondition(Value, 0),
                                              north = UVelocityBoundaryCondition(Value, :y, (x, z, t) -> f(x, t)))

    model = IncompressibleModel(       architecture = CPU(),
                                               grid = grid,
                                           coriolis = nothing,
                                           buoyancy = nothing,
                                            tracers = nothing,
                                            closure = ConstantIsotropicDiffusivity(ν=1),
                                boundary_conditions = (u=u_bcs,),
                                            forcing = ModelForcing(u=u_forcing, v=v_forcing))

    set!(model, u=uᵢ_xy, v=vᵢ_xy)

    h = min(2π/Nx, 1/Ny)
    Δt = h * CFL # Velocity scale = 1
    simulation = Simulation(model, Δt=Δt, stop_time=π, progress_frequency=1)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, FieldOutputs(model.velocities);
                                                          dir = dir, force = true,
                                                          prefix = @sprintf("forced_fixed_slip_xz_Nx%d_Δt%.1e", Nx, Δt),
                                                          interval = stop_iteration * Δt / 2)

    return simulation
end

function setup_and_run_forced_flow_xz(args...; kwargs...)
    simulation = forced_flow_simulation_xz(args...; kwargs...)
    @time run!(simulation)
    return simulation
end

function setup_and_run_forced_flow_xy(args...; kwargs...)
    simulation = forced_flow_simulation_xy(args...; kwargs...)
    @time run!(simulation)
    return simulation
end

end # module
