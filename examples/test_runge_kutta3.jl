using Oceananigans

grid = RegularCartesianGrid(size=(1, 1, 16), x=(0, 1), y=(0, 1), z=(-0.5, 0.5))

closure = IsotropicDiffusivity(κ=1.0)

model = IncompressibleModel(timestepper=:RungeKutta3, grid=grid, closure=closure)

width = 0.1

initial_temperature(x, y, z) = exp(-z^2 / (2width^2))

set!(model, T=initial_temperature)

diffusion_time_scale = model.grid.Δz^2 / model.closure.κ.T

simulation = Simulation(model, Δt = 0.1 * diffusion_time_scale, stop_iteration = 10)

run!(simulation)


#=
# Method 1
model_fields = fields(model)
    
for i in 1:length(solution_names)
for (i, field) in enumerate(model_fields)
    @inbounds Gⁿ = model.timestepper.Gⁿ[i]
    @inbounds G⁻ = model.timestepper.G⁻[i]
    
    solution_event = substep_solution_kernel!(f, Δt, γⁿ, ζⁿ, Gⁿ, G⁻; dependencies=barrier)
    
    push!(events, solution_event)


# Method 2
model_fields = fields(model)    # only do this once
    
for (i, field) in enumerate(model_fields)
     G = model.timestepper.Gⁿ[i]
     rk3_substep_field!(field, G, Δt, γ¹, nothing)
end
    

=#
