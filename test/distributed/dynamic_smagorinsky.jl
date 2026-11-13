include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using JLD2

# The test filter reads Σ across rank boundaries, so LM and MM on a distributed grid must match
# the serial computation on the same global grid, for both directional and Lagrangian averaging.

distributed_dynamic_smagorinsky_script = """
using MPI
MPI.Init()

using Oceananigans
using Oceananigans.DistributedComputations: Partition
using Oceananigans.TurbulenceClosures: LagrangianAveraging
using JLD2

u₀(x, y, z) = sin(2π * x) * cos(4π * y) * (1 + z) + cos(6π * x * y) / 2
v₀(x, y, z) = cos(2π * x) * sin(2π * y) * z + sin(4π * (x + y)) / 3
w₀(x, y, z) = sin(4π * x) * sin(2π * y) * z * (1 + z)

function dynamic_smagorinsky_model(arch, averaging)
    grid = RectilinearGrid(arch; size=(16, 16, 8), x=(0, 1), y=(0, 1), z=(-1, 0), halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))
    model = NonhydrostaticModel(grid; closure=DynamicSmagorinsky(; averaging))
    set!(model, u=u₀, v=v₀, w=w₀)
    return model
end

function directional_LM_and_MM(arch)
    model = dynamic_smagorinsky_model(arch, (1, 2))
    return Array(interior(model.closure_fields.LM)), Array(interior(model.closure_fields.MM))
end

function lagrangian_LM_and_MM(arch)
    model = dynamic_smagorinsky_model(arch, LagrangianAveraging())
    return Array(interior(model.closure_fields.𝒥ᴸᴹ)), Array(interior(model.closure_fields.𝒥ᴹᴹ))
end

arch = Distributed(CPU(), partition=Partition(2, 2))
LM, MM = directional_LM_and_MM(arch)
LMˢ, MMˢ = directional_LM_and_MM(CPU())
𝒥ᴸᴹ, 𝒥ᴹᴹ = lagrangian_LM_and_MM(arch)
𝒥ᴸᴹˢ, 𝒥ᴹᴹˢ = lagrangian_LM_and_MM(CPU())

nx, ny, _ = size(LM)
ri, rj, _ = arch.local_index
irange = (ri - 1) * nx + 1 : ri * nx
jrange = (rj - 1) * ny + 1 : rj * ny

jldsave("dynamic_smagorinsky_rank\$(arch.local_rank).jld2";
        LM, MM, LMˢ = LMˢ[irange, jrange, :], MMˢ = MMˢ[irange, jrange, :],
        𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹˢ = 𝒥ᴸᴹˢ[irange, jrange, :], 𝒥ᴹᴹˢ = 𝒥ᴹᴹˢ[irange, jrange, :])
"""

@testset "Distributed DynamicSmagorinsky" begin
    mktempdir() do dir
        cd(dir) do
            write("dynamic_smagorinsky.jl", distributed_dynamic_smagorinsky_script)
            run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 dynamic_smagorinsky.jl`)

            for rank in 0:3
                results = jldopen("dynamic_smagorinsky_rank$rank.jld2")
                @test results["LM"] ≈ results["LMˢ"]
                @test results["MM"] ≈ results["MMˢ"]
                @test results["𝒥ᴸᴹ"] ≈ results["𝒥ᴸᴹˢ"]
                @test results["𝒥ᴹᴹ"] ≈ results["𝒥ᴹᴹˢ"]
                close(results)
            end
        end
    end
end
