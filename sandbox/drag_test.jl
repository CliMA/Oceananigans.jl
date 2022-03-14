using GLMakie
using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Architectures: device
using Oceananigans.Operators: Δzᵃᵃᶜ, Δxᶜᵃᵃ

Nz = 64 # Resolution
ext_NZ = 80 # extra nodes in solid region
ν = 1e-2 # Viscosity
U = 1

@info "Checking u bottom BCs with drag"
grid = RectilinearGrid(size = Nz,
                              z = (0, 1),
                              halo = 1,
                              topology = (Flat, Flat, Bounded))
ext_grid = RectilinearGrid(size = ext_NZ,
                              z = (-.25, 1),
                              halo = 1,
                              topology = (Flat, Flat, Bounded))

flat_bottom(x, y) = 0
immersed_grid = ImmersedBoundaryGrid(ext_grid, GridFittedBottom(flat_bottom))

const κVK = 0.4 # van Karman's const
const z0 = 0.02 # roughness, user defined in future?
@inline drag_C(delta) = -(κVK ./ log(0.5*delta/z0)).^2 
@inline τˣᶻ_BC(x, y, t, u, v, w) = drag_C(grid.Δzᵃᵃᶜ) * u * (u^2 + v^2)^0.5


#####
##### Two ways to specify a boundary condition: "intrinsically", and with a forcing function
#####

u_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(τˣᶻ_BC, field_dependencies = (:u, :v, :w)))

kwargs = (closure = ScalarDiffusivity(ν=ν),
          advection = nothing,
          tracers = nothing,
          coriolis = nothing,
          buoyancy = nothing)

not_immersed_model = NonhydrostaticModel(grid = grid; boundary_conditions = (; u=u_bcs), kwargs...)
immersed_model = NonhydrostaticModel(grid = immersed_grid; kwargs...)
                          
for model in (immersed_model, not_immersed_model)
    # Linear stratification
    set!(model, u = U)

    Δt = 1e-1 * grid.Δzᵃᵃᶜ^2 / ν

    simulation = Simulation(model, Δt = Δt, stop_time = 1.0,)

    progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    if model.grid isa ImmersedBoundaryGrid
        prefix = "immersed_stokes_first_problem_drag"
    else
        prefix = "not_immersed_stokes_first_problem_drag"
    end

    simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u=model.velocities.u),
                                                          schedule = TimeInterval(0.01),
                                                          prefix = prefix,
                                                          force = true)

    run!(simulation)

    @info """
        Simulation complete.
        Runtime: $(prettytime(simulation.run_wall_time))
    """
end

immersed_filepath = "immersed_stokes_first_problem_drag.jld2"
not_immersed_filepath = "not_immersed_stokes_first_problem_drag.jld2"

z = znodes(Center, grid)
zi = znodes(Center, ext_grid)

uti = FieldTimeSeries(immersed_filepath, "u", grid=ext_grid)
utn = FieldTimeSeries(not_immersed_filepath, "u", grid=grid)

times = uti.times
Nt = length(times)
n = Observable(1)
uii(n) = interior(uti[n])[1, 1, :]
uin(n) = interior(utn[n])[1, 1, :]
upi = @lift uii($n)
upn = @lift uin($n)

fig = Figure(resolution=(400, 600))

ax = Axis(fig[1, 1], xlabel="u(z)", ylabel="z")
lines!(ax, upi, zi, label="immersed", linewidth=4, linestyle=:solid, color = :red)
lines!(ax, upn, z, label="not immersed", linewidth=4, color = :blue, linestyle="--")
ylims!(ax, -.1,1)
axislegend()
current_figure()

title_gen(n) = @sprintf("Stokes first problem at t = %.2f", times[n])
title_str = @lift title_gen($n)
ax_t = fig[0, :] = Label(fig, title_str)
prefix = "bottom_u_drag"
record(fig, prefix * ".mp4", 1:Nt, framerate=8) do nt
    n[] = nt
end

display(fig)
