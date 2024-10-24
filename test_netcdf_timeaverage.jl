# using Revise
using Oceananigans
using Plots
using NCDatasets
using Test
if isfile("single_decay_windowed_time_average_test.nc")
    rm("single_decay_windowed_time_average_test.nc")
end
run(`sh -c "rm test_iteration*.jld2"`)


function test_simulation(stop_time, Δt, window_nΔt, interval_nΔt, stride, overwrite)

    arch = CPU()
    topo = (Periodic, Periodic, Periodic)
    domain = (x=(0, 1), y=(0, 1), z=(0, 1))
    grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 4); domain...)

    λ1(x, y, z) = x + (1 - y)^2 + tanh(z)
    λ2(x, y, z) = x + (1 - y)^2 + tanh(4z)

    Fc1(x, y, z, t, c1) = - λ1(x, y, z) * c1
    Fc2(x, y, z, t, c2) = - λ2(x, y, z) * c2
    
    c1_forcing = Forcing(Fc1, field_dependencies=:c1)
    c2_forcing = Forcing(Fc2, field_dependencies=:c2)

    model = NonhydrostaticModel(; grid,
                                timestepper = :RungeKutta3,
                                tracers = (:c1, :c2),
                                forcing = (c1=c1_forcing, c2=c2_forcing))

    set!(model, c1=1, c2=1)
    simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

    ∫c1_dxdy = Field(Average(model.tracers.c1, dims=(1, 2)))
    ∫c2_dxdy = Field(Average(model.tracers.c2, dims=(1, 2)))
        
    nc_outputs = Dict("c1" => ∫c1_dxdy, "c2" => ∫c2_dxdy)
    nc_dimensions = Dict("c1" => ("zC",), "c2" => ("zC",))

    single_time_average_nc_filepath = "single_decay_windowed_time_average_test.nc"
    
    window = window_nΔt*Δt
    interval = interval_nΔt*Δt

    single_nc_output = Dict("c1" => ∫c1_dxdy)
    single_nc_dimension = Dict("c1" => ("zC",))

    simulation.output_writers[:single_output_time_average] =
        NetCDFOutputWriter(model, single_nc_output,
                           array_type = Array{Float64},
                           verbose = false,
                           filename = single_time_average_nc_filepath,
                           schedule = AveragedTimeInterval(interval, window = window, stride = stride),
                           dimensions = single_nc_dimension,
                           overwrite_existing = overwrite)
    checkpointer = Checkpointer(model,
                                schedule = TimeInterval(stop_time),
                                prefix = "test",
                                cleanup = true)

    simulation.output_writers[:checkpointer] = checkpointer


    function progress_message(sim) 
        wta = sim.output_writers[:single_output_time_average].outputs["c1"]
        wd = wta.window_start_time
        ws = wta.window_start_iteration
        pc = wta.previous_collection_time
        fo = wta.fetch_operand
        res = sum(wta.result)
        mct = model.clock.time
        interval = wta.schedule.interval
        window = wta.schedule.window
        cll = wta.schedule.collecting
        nat = next_actuation_time(wta.schedule)
        fat = wta.schedule.first_actuation_time
        actuation =  wta.schedule.actuations
        @info string("Iter: ", iteration(sim), ", time: ", prettytime(sim), ", model clock time:", mct, ", collecting: ", cll, ", u-avg: ", res, ", window_start_time: ", wd, ", window_start_iteration: ", ws, ", previous_collection_time:", pc, ", fetch_operand: ", fo, ", next_actuation_time: ", nat, ", first_actuation_time: ", fat, ", actuation: ", actuation, ", interval:", interval, ", window:", window)
    end
        simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))


    return simulation

end
    


function next_actuation_time(sch::AveragedTimeInterval)
    t₀ = sch.first_actuation_time
    N = sch.actuations
    interval = sch.interval
    return t₀ + (N + 1) * interval 
    # the actuation time is the end of the time averaging window
end

    Δt = 0.01 #1/64 # Nice floating-point number
    T1 = 100Δt      # first simulation stop time (s)
    T2 = 2T1      # second simulation stop time (s)
    window_nΔt = 3
    interval_nΔt = 5
    stride = 1
    # Run a simulation that saves data to a checkpoint
    simulation = test_simulation(T1, Δt, window_nΔt, interval_nΔt, stride, true)
    run!(simulation)
    checkpointed_wta = simulation.output_writers[:single_output_time_average].outputs["c1"]
    checkpointed_actuations = checkpointed_wta.schedule.actuations

    # Now try again, but picking up from the previous checkpoint
    N = iteration(simulation)
    checkpoint = "test_iteration$N.jld2"
    simulation = test_simulation(T2, Δt, window_nΔt, interval_nΔt, stride, false)
    simulation.output_writers[:single_output_time_average].outputs["c1"].schedule.actuations = checkpointed_actuations
    run!(simulation, pickup=checkpoint)

    ##### For each λ, horizontal average should evaluate to
    #####
    #####     c̄(z, t) = ∫₀¹ ∫₀¹ exp{- λ(x, y, z) * t} dx dy
    #####             = 1 / (Nx*Ny) * Σᵢ₌₁ᴺˣ Σⱼ₌₁ᴺʸ exp{- λ(i, j, k) * t}
    #####
    ##### which we can compute analytically.

    arch = CPU()
    topo = (Periodic, Periodic, Periodic)
    domain = (x=(0, 1), y=(0, 1), z=(0, 1))
    grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 4); domain...)

    λ1(x, y, z) = x + (1 - y)^2 + tanh(z)
    λ2(x, y, z) = x + (1 - y)^2 + tanh(4z)

    Fc1(x, y, z, t, c1) = - λ1(x, y, z) * c1
    Fc2(x, y, z, t, c2) = - λ2(x, y, z) * c2
    
    c1_forcing = Forcing(Fc1, field_dependencies=:c1)
    c2_forcing = Forcing(Fc2, field_dependencies=:c2)

    model = NonhydrostaticModel(; grid,
                                timestepper = :RungeKutta3,
                                tracers = (:c1, :c2),
                                forcing = (c1=c1_forcing, c2=c2_forcing))


    Nx, Ny, Nz = size(grid)
    xs, ys, zs = nodes(model.tracers.c1)

    c̄1(z, t) = 1 / (Nx * Ny) * sum(exp(-λ1(x, y, z) * t) for x in xs for y in ys)
    c̄2(z, t) = 1 / (Nx * Ny) * sum(exp(-λ2(x, y, z) * t) for x in xs for y in ys)

    rtol = 1e-5 # need custom rtol for isapprox because roundoff errors accumulate (?)

    # Compute time averages...
    c̄1(ts) = 1/length(ts) * sum(c̄1.(zs, t) for t in ts)
    c̄2(ts) = 1/length(ts) * sum(c̄2.(zs, t) for t in ts)

    #####
    ##### Test strided windowed time average against analytic solution
    ##### for *single* NetCDF output
    #####
    single_time_average_nc_filepath = "single_decay_windowed_time_average_test.nc"
    single_ds = NCDataset(single_time_average_nc_filepath)

    attribute_names = ("schedule", "interval", "output time interval",
                       "time_averaging_window", "time averaging window",
                       "time_averaging_stride", "time averaging stride")

    for name in attribute_names
        @test haskey(single_ds.attrib, name) && !isnothing(single_ds.attrib[name])
    end

    window_size = window_nΔt
    window = window_size*Δt
    # @info "    Testing time-averaging of a single NetCDF output [$(typeof(arch))]..."

    for (n, t) in enumerate(single_ds["time"][2:end])
        averaging_times = [t - n*Δt for n in 0:stride:window_size-1 if t - n*Δt >= 0]
        # @info n,t,averaging_times, c̄1(averaging_times), single_ds["c1"][:, n+1], c̄1(averaging_times)./single_ds["c1"][:, n+1]
        @test all(isapprox.(single_ds["c1"][:, n+1], c̄1(averaging_times), rtol=rtol))
    end

    time = single_ds["time"][:]
    data_plot = single_ds["c1"][1:4, :]
    c̄1_timeaverage = zeros(4,length(time[1:end]))
    for (n, t) in enumerate(time[1:end])
        averaging_times = [t - n*Δt for n in 0:stride:window_size-1 if t - n*Δt >= 0]
        # @info n,t,averaging_times, c̄1(averaging_times)
        c̄1_timeaverage[:,n] = c̄1(averaging_times)
        # @test all(isapprox.(single_ds["c1"][:, n+1], c̄1(averaging_times), rtol=rtol))
    end

    # Plot each of the four lines
    pl = plot()
    plot!(time, data_plot[1, :], label="1", color=:blue, legend=:topright)
    plot!(time, data_plot[2, :], label="2", color=:red)
    plot!(time, data_plot[3, :], label="3", color=:orange)
    plot!(time, data_plot[4, :], label="4", color=:green)
    
    plot!(time[1:end],c̄1_timeaverage[1,:], color=:black, linestyle=:dash, label="1-analytic")
    plot!(time[1:end],c̄1_timeaverage[2,:], color=:black, linestyle=:dash, label="2-analytic")
    plot!(time[1:end],c̄1_timeaverage[3,:], color=:black, linestyle=:dash, label="3-analytic")
    plot!(time[1:end],c̄1_timeaverage[4,:], color=:black, linestyle=:dash, label="4-analytic")
    

    tt = 0:window:T2
    for i in 1:length(tt)
    plot!([tt[i], tt[i]],[0,1],color=:grey,label="")
    end
    title!(pl, string("Δt=",Δt,", average window=",window_nΔt,"Δt", ", interval=",interval_nΔt,"Δt")) # Add the title to the plot
    ylims!(pl,(minimum(c̄1_timeaverage[4,:]),maximum(c̄1_timeaverage[4,:])))
    xlims!(pl,(0,T2))
    close(single_ds)
    display(pl)
