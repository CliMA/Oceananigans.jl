function show_mask(grid)

    #print("grid = ", grid, "\n")
    c = CenterField(CPU(), grid)
    c .= 1

    mask_immersed_field!(c)

    x, y, z = nodes(c)

    return x, y, z, c
end

function nice_divergent_levels(c, clim; nlevels=20)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return (-clim, clim), levels
end

function nan_solid(y, z, v, seamount)
    Ny, Nz = size(v)
    y2 = reshape(y, Ny, 1)
    z2 = reshape(z, 1, Nz)
    v[seamount.(0, y2,z2)] .= NaN
    return nothing
end

function visualize_flow_over_seamount_simulation(prefix)

    h0, L = 0.5, 0.25
    seamount(x, y, z) = z < - 1 + h0*exp(-y^2/L^2)

    filename = prefix * ".jld2"
    file = jldopen(filename)

    grid = file["serialized/grid"]


    xθ, yθ, zθ = nodes((Center, Center, Center), grid)

    θ₀ = file["timeseries/θ/0"][1, :, :]

    iterations = parse.(Int, keys(file["timeseries/t"]))

    anim = @animate for (i, iter) in enumerate(iterations)

        @info "Plotting iteration $iter of $(iterations[end])..."


        θ = file["timeseries/θ/$iter"][1, :, :]
        t = file["timeseries/t/$iter"]

        θ′ = θ .- θ₀


        θmax = maximum(abs, θ)

        print("Max θ = ", θmax)


        nan_solid(yθ, zθ, θ′, seamount)

        θ_title = @sprintf("θ, t = %.2f", t)


    θ_plot = contourf(yθ, zθ, θ'; title = θ_title)

    end

    mp4_title = string(prefix, ".mp4")
    mp4(anim, mp4_title, fps = 16)

    close(file)
end

function find_topography(grid_with_seamount)

    Ny = grid_with_seamount.grid.Ny
    topography_index = zeros(Int64, Ny)
    for (iyC, yC) in enumerate(grid_with_seamount.grid.yᵃᶜᵃ[1:Ny])
        #print("y = ", yC, " z = ", grid_with_seamount.grid.Δzᵃᵃᶜ[1], " ")

        izC = 1
        test = true
        while test && izC <= grid_with_seamount.grid.Nz
          #print(izC, " ")
          test =  solid_node(Center(), Center(), Center(), 1, iyC, izC, grid_with_seamount)
          izC += 1
        end
        #print(" ")
        #print(solid_node(Center(), Center(), Center(), 1,iyC,1, grid_with_seamount)," ")
        #print(solid_node(Center(), Center(), Center(), 1,iyC,2, grid_with_seamount),"\n")
        topography_index[iyC] = izC - 1
    end

    return topography_index
end

function plot_norm_grad_tracer(yθ, zθ, grid_with_seamount, model, scheme)

    G1 = Field{Center, Center, Center}(grid_with_seamount)
    G2 = Field{Center, Center, Center}(grid_with_seamount)

    G1 .= ∂y(model.tracers.θ)
    G2 .= ∂z(model.tracers.θ)

    final_grad=sqrt.(G1.^2 .+ G2.^2 )

    norm(final_grad, Inf)
    norm_title = string("final_grad_", string(nameof(typeof(scheme))))
    grad_plot = contourf(yθ, zθ, (final_grad)[1, :, :]', title=norm_title, xlabel="y", ylabel="z")
    norm_file = string("final_grad_", string(nameof(typeof(scheme))), ".png")
    savefig(grad_plot, norm_file)

    grid = grid_with_seamount.grid

    topography_index = find_topography(grid_with_seamount)

    Theta_topography_slice = zeros(grid.Ny)

    for (iyC, yC) in enumerate(grid_with_seamount.grid.yᵃᶜᵃ[1:grid.Ny])
        Theta_topography_slice[iyC] = (model.tracers.θ)[1, iyC, topography_index[iyC]]
    end

    y_slice=1:grid.Ny

    theta_topopgraphy_file = string("theta_slice", string(nameof(typeof(scheme))))
    theta_slice_plot = plot(y_slice, Theta_topography_slice,  xlabel="y", ylabel="Theta_topography_slice")

    savefig(theta_slice_plot, theta_topopgraphy_file)

end

function simulate_advection(V, W, B, scheme, grid_with_seamount, Δt, stop_time, index)

    grid = grid_with_seamount.grid

    ## Set up Model
    velocities = PrescribedVelocityFields(v=V, w=W)
    model = HydrostaticFreeSurfaceModel(tracer_advection = scheme,
                                        grid = grid_with_seamount,
                                        tracers = :θ,
                                        velocities = velocities,
                                        buoyancy = nothing
                                        )

    set!(model, θ = B)

    ### Plot initial tracer field: tracer_initial.png
    if index == 1
        xθ, yθ, zθ = nodes((Center, Center, Center), grid)
        θplot = contourf(yθ, zθ, interior(model.tracers.θ)[1, :, :]', title="tracer initial", xlabel="y", ylabel="z")
        savefig(θplot, "tracer_initial.png")
    end

    ### Total intial tracer
    tracer_initial_total  = sum(interior(model.tracers.θ))/(grid.Ny*grid.Nz)
    tracer_initial2_total = sum(interior(model.tracers.θ).^2)/(grid.Ny*grid.Nz)

    ### Simulation
    simulation = Simulation(model, Δt = Δt, stop_time = stop_time)#1)

    progress(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|θ|: %.2e",
                                 100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                                 s.model.clock.time, maximum(abs, model.tracers.θ))

    progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    serialize_grid(file, model) = file["serialized/grid"] = model.grid.grid
    data_title = string("flow_over_seamount_", string(nameof(typeof(scheme))))
    simulation.output_writers[:fields] = JLD2OutputWriter(model, model.tracers,
                                                          schedule = TimeInterval(0.02),
                                                          prefix = data_title,
                                                          init = serialize_grid,
                                                          force = true)

    start_time = time_ns()
    run!(simulation)
    finish_time = time_ns()

    simulation_time = (finish_time - start_time)/1e9

    return model, simulation, simulation_time, tracer_initial_total, tracer_initial2_total
end
