using GLMakie
using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Architectures: device
using Oceananigans.Operators: Δzᵃᵃᶜ, Δxᶜᵃᵃ

Nz = 16 # Resolution
Lz = 1
ext_Lz = 1.5Lz
ext_NZ = Int(ext_Lz/(Lz/Nz)) # extra nodes in solid region
ν = 1e-2 # Viscosity
U = 1
topo = (Periodic, Periodic, Bounded)

@info "Checking u bottom BCs with drag"
grid = RectilinearGrid(size = (10, 10, Nz),
                       x=(0, 1),
                       y=(0, 1),
                       z = (1-Lz, 1),
                       halo = (1,1,1),
                       topology = topo)
ext_grid = RectilinearGrid(size = (10, 10, ext_NZ),
                       x=(0, 1),
                       y=(0, 1),
                       z = (1-ext_Lz, 1),
                       halo = (1,1,1),
                       topology = topo)

flat_bottom(x, y) = 0
immersed_grid = ImmersedBoundaryGrid(ext_grid, GridFittedBottom(flat_bottom))

const κVK = 0.4 # van Karman's const
const z0 = 0.02 # roughness, user defined in future?
@inline drag_C(delta) = -(κVK ./ log(0.5*delta/z0)).^2 
@inline τˣᶻ_BC_bottom(x, y, t, u, v, w) = drag_C(grid.Δzᵃᵃᶜ) * u * (u^2 + v^2)^0.5
@inline τˣᶻ_BC_top(x, y, t, u, v, w) = -τˣᶻ_BC_bottom(x, y, t, u, v, w)


#####
##### Two ways to specify a boundary condition: "intrinsically", and with a forcing function
#####

u_drag_bc = FieldBoundaryConditions(bottom = FluxBoundaryCondition(τˣᶻ_BC_bottom, field_dependencies = (:u, :v, :w)),
                                    top    = FluxBoundaryCondition(τˣᶻ_BC_top, field_dependencies = (:u, :v, :w)),
                                    )
u_noslip_bc = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0),
                                      top    = ValueBoundaryCondition(0),
                                      )

kwargs = (closure = ScalarDiffusivity(ν=ν),
          advection = WENO5(),
          tracers = nothing,
          coriolis = nothing,
          buoyancy = nothing)

control_drag_model = NonhydrostaticModel(grid = grid; boundary_conditions = (; u=u_drag_bc), kwargs...)
#control_noslip_model = NonhydrostaticModel(grid = grid; boundary_conditions = (; u=u_noslip_bc), kwargs...)
immersed_model = NonhydrostaticModel(grid = immersed_grid; kwargs...)

models = (immersed_model,)
names = ("immersed_model",)
models = (control_drag_model, immersed_model)
names = ("control_drag_model", "immersed_model")
#models = (control_drag_model, control_noslip_model, immersed_model,)
#names = ("control_drag_model", "control_noslip_model", "immersed_model",)
                          
for (prefix, model) in zip(names, models)
    @info "Now running model $prefix"
    # Linear stratification
    set!(model, u = U)

    Δt = 1e-1 * grid.Δzᵃᵃᶜ^2 / ν

    simulation = Simulation(model, Δt = Δt, stop_time = 1.0,)

    progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u=model.velocities.u),
                                                          schedule = TimeInterval(0.02),
                                                          prefix = prefix,
                                                          force = true)

    simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; u=model.velocities.u),
                                                            schedule = TimeInterval(0.02),
                                                            filepath = "$prefix.nc",
                                                            mode = "c")

    run!(simulation)

    @info """
        Simulation complete.
        Runtime: $(prettytime(simulation.run_wall_time))
    """
end

pause
immersed_filepath = "control_drag_model.jld2"
not_immersed_filepath = "immersed_model.jld2"

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
