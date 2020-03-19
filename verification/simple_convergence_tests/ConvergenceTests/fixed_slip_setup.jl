using Oceananigans, Oceananigans.Forcing, Oceananigans.BoundaryConditions, Oceananigans.OutputWriters
using PyPlot
using Statistics

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

function forced_flow_simulation_xz(Nx, Nz, CFL)
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

    return simulation
end

function forced_flow_simulation_xy(Nx, Ny, CFL)
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

    return simulation
end

function compute_error_xz(simulation)
    model = simulation.model
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    u, v, w = model.velocities

    XU = repeat(grid.xF[1:end-1], 1, Nz)
    ZU = repeat(reshape(grid.zC, 1, Nz), Nx, 1)
    u_sim = interior(u)[:, 1, :]

    u_analytical = u_xz.(XU, 0, ZU, model.clock.time)

    absolute_error = @. abs(u_sim - u_analytical)

    L₁ = mean(absolute_error)
    L★ = maximum(absolute_error)

    return L₁, L★
end

function compute_error_xy(simulation)
    model = simulation.model
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    u, v, w = model.velocities

    XU = repeat(grid.xF[1:end-1], 1, Ny)
    YU = repeat(reshape(grid.yC, 1, Ny), Nx, 1)
    u_sim = interior(u)[:, :, 1]

    u_analytical = u_xy.(XU, YU, 0, model.clock.time)

    absolute_error = @. abs(u_sim - u_analytical)

    L₁ = mean(absolute_error)
    L★ = maximum(absolute_error)

    return L₁, L★
end

function setup_and_run_forced_flow_xz(args...; kwargs...)
    simulation = forced_flow_simulation_xz(args...; kwargs...)
    @time run!(simulation)
    @show compute_error_xz(simulation)
    plot_forced_flow_xz(simulation)
    return simulation
end

function setup_and_run_forced_flow_xy(args...; kwargs...)
    simulation = forced_flow_simulation_xy(args...; kwargs...)
    @time run!(simulation)
    @show compute_error_xy(simulation)
    plot_forced_flow_xy(simulation)
    return simulation
end

function plot_forced_flow_xz(simulation)

    model = simulation.model
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    u, v, w = model.velocities

    XU = repeat(grid.xF[1:end-1], 1, Nz)
    ZU = repeat(reshape(grid.zC, 1, Nz), Nx, 1)
    u_sim = interior(u)[:, 1, :]

    XW = repeat(grid.xC, 1, Nz+1)
    ZW = repeat(reshape(grid.zF, 1, Nz+1), Nx, 1)
    w_sim = interior(w)[:, 1, :]

    u_analytical = u_xz.(XU, 0, ZU, model.clock.time)
    u_err = @. abs(u_sim - u_analytical)

    fig, axs = subplots(ncols=2, figsize=(10, 6))

    sca(axs[1])
    contourf(XU, ZU, u_sim)

    sca(axs[2])
    contourf(XU, ZU, u_err)

    return nothing
end



