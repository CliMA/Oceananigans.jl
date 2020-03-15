using Oceananigans, Oceananigans.Forcing, Oceananigans.BoundaryConditions

# Functions that define the forced flow problem

 ξ(t) = 1 + sin(t^2)
ξ′(t) = 2t * cos(t^2)

 f(x, t) =   cos(x - ξ(t))
fₓ(x, t) = - sin(x - ξ(t))

 g(y) = y^3 - y^2
g′(y) = 3y^2 - 2y

H₁(y) = 1/4 * y^4 - 1/3 * y^3 - 3y^2
H₂(y) = 3y^4 - 2y^3
H₁(y) = 1/4 * y^4 - 1/3 * y^3 - 6y^2

Fᵤ(x, y, t) =     ξ′(t) * fₓ(x, t) * H₁(y) + 
                f(x, t) * fₓ(x, t) * H₂(y) - 
                f(x, t) * H₃(y) 

Fᵥ(x, y, t) = 3y^5 - 5y^4 + 2y^3

uᵢ_xy(x, y, z) = f(x, 0) * g′(y)
vᵢ_xy(x, y, z) = - fₓ(x, 0) * g(y)

uᵢ_xz(x, y, z) = f(x, 0) * g′(z)
wᵢ_xy(x, y, z) = - fₓ(x, 0) * g(z)

function forced_flow_xy_model(Nx, Ny, Δt)

    u_forcing = SimpleForcing(Fᵤ)
    v_forcing = SimpleForcing(Fᵥ)

    grid = RegularCartesianGrid(size=(Nx, Ny, 1), length(2π, 1, 0), topology=(Periodic, Bounded, Flat))

    u_slip_velocity = BoundaryFunction{:y, Face, Cell}((x, z, t) -> f(x, t))

    u_bcs = UVelocityBoundaryConditions(grid, south = BoundaryCondition(Value, 0),
                                              north = BoundaryFunction(Value, slip_velocity))

    model = IncompressibleModel(       architecture = CPU(),
                                               grid = grid,
                                           coriolis = nothing,
                                           buoyancy = nothing,
                                            tracers = nothing,
                                            closure = ConstantIsotropicDiffusivity(ν=1),
                                boundary_conditions = (u=u_bcs,),
                                            forcing = (u=u_forcing, v=v_forcing))

    set!(model, u=uᵢ_xy, v=vᵢ_xy)

    field_writer = JLD2OutputWriter(model, FieldOutputs(model.velocities); interval=2π/100,
                                    prefix="forced_flow_verification_experiment", force=true)


    simulation = Simulation(model, Δt=Δt, stop_time=4π, progress_frequency=10)
    simulation.output_writers[:fields] = field_writer

    return simulation
end
