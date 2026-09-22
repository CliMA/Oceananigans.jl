# Every rank runs the same files in the same order; a failing testset makes the rank exit non-zero.

const mpi_groups = Dict(
    "distributed"                           => ["mpi/architectures", "mpi/models"],
    "distributed_solvers"                   => ["mpi/transpose", "mpi/poisson_solvers", "mpi/conjugate_gradient_solver"],
    "distributed_hydrostatic_model"         => ["mpi/hydrostatic_model", "mpi/split_explicit_boundaries"],
    "distributed_hydrostatic_regression"    => ["regression/hydrostatic"],
    "distributed_nonhydrostatic_regression" => ["regression/nonhydrostatic"],
    "distributed_memory_allocation"         => ["memory_allocation/memory_allocation"],
    "distributed_vertical_coordinate_1"     => ["vertical_coordinate/conservation_explicit"],
    "distributed_vertical_coordinate_2"     => ["vertical_coordinate/conservation_implicit", "vertical_coordinate/conservation_tripolar"],
    "nccl_extension"                        => ["mpi/nccl"],
)

function mpi_tests(group)
    names = String[]
    for g in split(group, ",")
        haskey(mpi_groups, g) || error("Unknown MPI test group $(repr(g)); available: $(join(sort!(collect(keys(mpi_groups))), ", "))")
        append!(names, mpi_groups[g])
    end
    return names
end

include(joinpath(@__DIR__, "dependencies_for_runtests.jl"))
reset_cuda_if_necessary()

@testset "Oceananigans" begin
    for name in mpi_tests(group)
        mod = @eval Main module $(gensym(name)) end
        @testset "$name" begin
            Core.eval(mod, :(include($(joinpath(@__DIR__, "..", name * ".jl")))))
        end
    end
end
