include("distributed_tests_utils.jl")

using Reactant
using Oceananigans.TimeSteppers: first_time_step!

# Required for Reactant MLIR compilation of sharded models (see GB-25)
Reactant.Compiler.WHILE_CONCAT[] = true

ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
ENV["JULIA_DEBUG"] = "Reactant, Reactant_jll"

# Dispatch on Reactant grids to compile with @compile before time stepping
const ReactantArch = Union{ReactantState, Distributed{<:ReactantState}}
const ReactantTestGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:ReactantArch}

function run_distributed_simulation(grid::ReactantTestGrid)

    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                        tracers = :c,
                                        tracer_advection = WENO(),
                                        momentum_advection = WENOVectorInvariant(order=3),
                                        coriolis = HydrostaticSphericalCoriolis())

    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)
    set!(model, c=ηᵢ, η=ηᵢ)

    Δt = 5minutes

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

Reactant.Distributed.initialize(; single_gpu_per_process=false)

run_function = run_distributed_latitude_longitude_grid
suffix = "llg"

arch = Distributed(ReactantState(), partition = Partition(4, 1))
filename = "distributed_xslab_$(suffix).jld2"
run_function(arch, filename)

arch = Distributed(ReactantState(), partition = Partition(1, 4))
filename = "distributed_yslab_$(suffix).jld2"
run_function(arch, filename)

arch = Distributed(ReactantState(), partition = Partition(2, 2))
filename = "distributed_pencil_$(suffix).jld2"
run_function(arch, filename)
