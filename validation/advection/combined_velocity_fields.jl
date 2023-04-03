using Oceananigans

# for `test_velocity_combination`
struct SinkingParticles <: AbstractBiogeochemistry end
biogeochemical_drift_velocity(::SinkingParticles, ::Val{:A}) = (u = ConstantField(-1), v = ZeroField(), w = ZeroField())

grid = RectilinearGrid(arch; size = 40, halo = 5, x = (0, 1), topology = (Periodic, Flat, Flat))

u(args...) = -1

model = NonhydrostaticModel(; grid, 
                              tracers=:A, 
                              advection = UpwindBiased(),
                              background_fields = (; u))

A₀(x, y, z) = x > 0.4 && x < 0.6 ? 1.0 : 0.0

set!(model, u = 1, A = A₀)

Δt = 0.2 / 1.0 * grid.Δxᶜᵃᵃ[1]

A_hist = zeros(1000, grid.Nx)

#=for step in 1:10
    A_hist[step, :] = model.tracers.A[1:grid.Nx, 1, 1]
    time_step!(model, Δt)
end=#

A_truth = [A₀(x, 0, 0) for x in nodes(grid, Center(), Center(), Center())[1]]

#@test all([all(A_hist[it, :] .≈ A_truth) for it in 1:1000])

model = NonhydrostaticModel(; grid, 
                              tracers=:A, 
                              advection = UpwindBiased(),
                              biogeochemistry = SinkingParticles())

set!(model, u = 1, A = A₀)

A_hist = zeros(1000, grid.Nx)

#=for step in 1:1000
    A_hist[step, :] = model.tracers.A[1:grid.Nx, 1, 1]
    time_step!(model, Δt)
end=#

#@test all([all(A_hist[it, :] .≈ A_truth) for it in 1:1000])

model = NonhydrostaticModel(; grid, 
                              tracers=:A, 
                              advection = UpwindBiased(),
                              biogeochemistry = SinkingParticles(),
                              background_fields = (; u))

set!(model, u = 2, A = A₀)

A_hist = zeros(1000, grid.Nx)

#=for step in 1:1000
    A_hist[step, :] = model.tracers.A[1:grid.Nx, 1, 1]
    time_step!(model, Δt/2)
end=#

#@test all([all(A_hist[it, :] .≈ A_truth) for it in 1:1000])
