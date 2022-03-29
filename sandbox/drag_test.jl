#using GLMakie
using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Architectures: device
using Oceananigans.Operators: Δxᶠᵃᵃ, Δyᵃᶠᵃ, Δzᵃᵃᶠ

Nz = 16 # Resolution
Lz = 1
δ = 1/4
Lz_ext = Lz + 2δ
ext_NZ = Int(Lz_ext/(Lz/Nz)) # extra nodes in solid region
ν = 1e-2 # Viscosity
U = 1

kwargs_model = (closure = ScalarDiffusivity(ν=ν),
                advection = UpwindBiasedFifthOrder(),
                tracers = nothing,
                coriolis = nothing,
                buoyancy = nothing)


const κᵥₖ = 0.4 # van Karman's const
const z0 = 0.02 # roughness, user defined in future?
@inline drag_C(delta) = -(κᵥₖ ./ log(0.5*delta/z0)).^2 

@inline τʸˣ_BC_west(x, y, t, u, v, w) = drag_C(grid.Δxᶠᵃᵃ) * v * (v^2 + w^2)^0.5
@inline τᶻˣ_BC_west(x, y, t, u, v, w) = drag_C(grid.Δxᶠᵃᵃ) * w * (v^2 + w^2)^0.5

@inline τˣʸ_BC_south(x, y, t, u, v, w) = drag_C(grid.Δyᵃᶠᵃ) * u * (u^2 + w^2)^0.5
@inline τᶻʸ_BC_south(x, y, t, u, v, w) = drag_C(grid.Δyᵃᶠᵃ) * w * (u^2 + w^2)^0.5

@inline τˣᶻ_BC_bottom(x, y, t, u, v, w) = drag_C(grid.Δzᵃᵃᶠ) * u * (u^2 + v^2)^0.5
@inline τʸᶻ_BC_bottom(x, y, t, u, v, w) = drag_C(grid.Δzᵃᵃᶠ) * v * (u^2 + v^2)^0.5

@inline τʸˣ_BC_east(x, y, t, u, v, w) = -τʸˣ_BC_west(x, y, t, u, v, w)
@inline τᶻˣ_BC_east(x, y, t, u, v, w) = -τᶻˣ_BC_west(x, y, t, u, v, w)

@inline τˣʸ_BC_north(x, y, t, u, v, w) = -τˣʸ_BC_south(x, y, t, u, v, w)
@inline τᶻʸ_BC_north(x, y, t, u, v, w) = -τᶻʸ_BC_south(x, y, t, u, v, w)

@inline τˣᶻ_BC_top(x, y, t, u, v, w) = -τˣᶻ_BC_bottom(x, y, t, u, v, w)
@inline τʸᶻ_BC_top(x, y, t, u, v, w) = -τˣᶻ_BC_bottom(x, y, t, u, v, w)

topo_main = [Flat, Flat, Bounded]
directions = [:x, :y, :z]
vels = [:u, :v, :w]

is_immersed_x(x, y, z) = (x < 0) | (x > 1)
is_immersed_y(x, y, z) = (y < 0) | (y > 1)
is_immersed_z(x, y, z) = (z < 0) | (z > 1)

for i in 1:3
    @info i

    global topo = circshift(topo_main, i)
    global bounded_dir = directions[i]

    kwargs = Dict(directions[i] => (1-Lz, 1))
    global grid = RectilinearGrid(; size = Nz,
                           kwargs...,
                           halo = 1,
                           topology = topo)
    @show grid

    global kwargs_ext = Dict(bounded_dir => (1-Lz-δ, 1+δ))
    global grid_ext = RectilinearGrid(; size = ext_NZ,
                               kwargs_ext...,
                               halo = 1,
                               topology = topo)
    @show grid_ext

    if bounded_dir == :x
        global immersed_grid = ImmersedBoundaryGrid(grid_ext, GridFittedBoundary(is_immersed_x))
        v_drag_bc = FieldBoundaryConditions(west = FluxBoundaryCondition(τʸˣ_BC_west, field_dependencies = (:u, :v, :w)),
                                            east = FluxBoundaryCondition(τʸˣ_BC_east, field_dependencies = (:u, :v, :w)),
                                            )
        w_drag_bc = FieldBoundaryConditions(west = FluxBoundaryCondition(τᶻˣ_BC_west, field_dependencies = (:u, :v, :w)),
                                            east = FluxBoundaryCondition(τᶻˣ_BC_east, field_dependencies = (:u, :v, :w)),
                                            )
        global boundary_conditions = (v = v_drag_bc, w = w_drag_bc)

    elseif bounded_dir == :y
        global immersed_grid = ImmersedBoundaryGrid(grid_ext, GridFittedBoundary(is_immersed_y))
        u_drag_bc = FieldBoundaryConditions(south = FluxBoundaryCondition(τˣʸ_BC_south, field_dependencies = (:u, :v, :w)),
                                            north = FluxBoundaryCondition(τˣʸ_BC_north, field_dependencies = (:u, :v, :w)),
                                            )
        w_drag_bc = FieldBoundaryConditions(south = FluxBoundaryCondition(τᶻʸ_BC_south, field_dependencies = (:u, :v, :w)),
                                            north = FluxBoundaryCondition(τᶻʸ_BC_north, field_dependencies = (:u, :v, :w)),
                                            )
        global boundary_conditions = (u = u_drag_bc, w = w_drag_bc)

    elseif bounded_dir == :z
        global immersed_grid = ImmersedBoundaryGrid(grid_ext, GridFittedBoundary(is_immersed_z))
        u_drag_bc = FieldBoundaryConditions(bottom = FluxBoundaryCondition(τˣᶻ_BC_bottom, field_dependencies = (:u, :v, :w)),
                                            top    = FluxBoundaryCondition(τˣᶻ_BC_top, field_dependencies = (:u, :v, :w)),
                                            )
        v_drag_bc = FieldBoundaryConditions(bottom = FluxBoundaryCondition(τʸᶻ_BC_bottom, field_dependencies = (:u, :v, :w)),
                                            top    = FluxBoundaryCondition(τʸᶻ_BC_top, field_dependencies = (:u, :v, :w)),
                                            )
        global boundary_conditions = (u = u_drag_bc, v = v_drag_bc)
    end
    @show immersed_grid

    control_drag_model = NonhydrostaticModel(grid = grid; boundary_conditions = boundary_conditions, kwargs_model...)
    immersed_model = NonhydrostaticModel(grid = immersed_grid; kwargs_model...)

    println("\n\n")

    models = (immersed_model,)
    names = ("immersed_model",)
    models = (control_drag_model, immersed_model)
    names = ("control_drag_model", "immersed_model")
                          

    for (model_prefix, model) in zip(names, models)
        name_prefix = "$(bounded_dir)_$model_prefix"
        @info "\nNow running model $name_prefix\n"
        # Linear stratification
        vels_dict = Dict(vels[i] => U for i in 1:3 if topo[i] != Bounded)
        set!(model; vels_dict...)
    
        Δt = 1e-1 * grid.Δzᵃᵃᶜ^2 / ν
        simulation = Simulation(model, Δt = Δt, stop_time = 1.0,)
        
        progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"
    
        simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
    
        simulation.output_writers[:fields] = JLD2OutputWriter(model, (; model.velocities...),
                                                              schedule = TimeInterval(0.02),
                                                              prefix = name_prefix,
                                                              force = true)
    
        simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; model.velocities...),
                                                                schedule = TimeInterval(0.02),
                                                                filepath = "$name_prefix.nc",
                                                                mode = "c")
    
        run!(simulation)
    
        @info """
            Simulation complete.
            Runtime: $(prettytime(simulation.run_wall_time))
        """
    end
end

# I couldn't make the rest of this script work with more recent versions of GLMakie!
# So I switched to `plot_drag.py`
#pause
#immersed_filepath = "control_drag_model.jld2"
#not_immersed_filepath = "immersed_model.jld2"

#z = znodes(Center, grid)
#zi = znodes(Center, grid_ext)

#uti = FieldTimeSeries(immersed_filepath, "u", grid=grid_ext)
#utn = FieldTimeSeries(not_immersed_filepath, "u", grid=grid)

#times = uti.times
#Nt = length(times)
#n = Observable(1)
#uii(n) = interior(uti[n])[1, 1, :]
#uin(n) = interior(utn[n])[1, 1, :]
#upi = @lift uii($n)
#upn = @lift uin($n)

#fig = Figure(resolution=(400, 600))

#ax = Axis(fig[1, 1], xlabel="u(z)", ylabel="z")
#lines!(ax, upi, zi, label="immersed", linewidth=4, linestyle=:solid, color = :red)
#lines!(ax, upn, z, label="not immersed", linewidth=4, color = :blue, linestyle="--")
#ylims!(ax, -.1,1)
#axislegend()
#current_figure()

#title_gen(n) = @sprintf("Stokes first problem at t = %.2f", times[n])
#title_str = @lift title_gen($n)
#ax_t = fig[0, :] = Label(fig, title_str)
#model_prefix = "bottom_u_drag"
#record(fig, model_prefix * ".mp4", 1:Nt, framerate=8) do nt
#    n[] = nt
#end

#display(fig)
