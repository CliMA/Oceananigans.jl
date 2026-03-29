include("sharding_test_utils.jl")

function time_step_sharded_model(arch)

    model = sharding_test_model(arch)
    Δt = model.clock.last_Δt

    @info "Compiling first_time_step..."
    r_first_time_step! = @compile sync=true raise=true first_time_step!(model, Δt)

    @info "Compiling time_step..."
    r_time_step! = @compile sync=true raise=true time_step!(model, Δt)

    @info "Running first time step..."
    r_first_time_step!(model, Δt)

    @info "Running time step..."
    for N in 2:100
        r_time_step!(model, Δt)
    end

    return model
end

function run_and_save(arch, filename)
    model = time_step_sharded_model(arch)

    u = interior(model.velocities.u, :, :, 10)
    v = interior(model.velocities.v, :, :, 10)
    c = interior(model.tracers.c, :, :, 10)
    η = interior(model.free_surface.displacement, :, :, 1)

    jldsave(filename; u=Array(u), v=Array(v), c=Array(c), η=Array(η))

    return nothing
end

# No Reactant.Distributed.initialize() needed for single-process
# sharding with virtual devices (--xla_force_host_platform_device_count).
# See GB-25 sharding_utils.jl.

arch = Distributed(ReactantState(), partition = Partition(2, 2))
run_and_save(arch, "distributed_pencil_llg.jld2")
