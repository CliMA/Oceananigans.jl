push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using OrderedCollections
using Oceananigans
using Oceananigans.Grids
using Oceananigans.AbstractOperations
using Oceananigans.Fields
using Oceananigans.Utils
using Benchmarks

FT = Float64
Archs = has_cuda() ? [CPU, GPU] : [CPU]

#####
##### Run benchmarks
#####

tags = ["Architecture", "ID", "Operation"]
suite = BenchmarkGroup(tags)

print_system_info()

for Arch in Archs
    N = Arch == CPU ? (32, 32, 32) : (256, 256, 256)

    grid = RegularRectilinearGrid(FT, size=N, extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), float_type=FT, grid=grid,
                                buoyancy=nothing, tracers=(:α, :β, :γ, :δ, :ζ))

    ε(x, y, z) = randn()
    ε⁺(x, y, z) = abs(randn())
    set!(model, u=ε, v=ε, w=ε, α=ε, β=ε, γ=ε, δ=ε, ζ=ε⁺)

    u, v, w = model.velocities
    α, β, γ, δ, ζ = model.tracers

    dump_field = Field(Center, Center, Center, Arch(), grid)

    test_cases = OrderedDict(
        "-α"      => -α,
        "√ζ"      => √ζ,
        "sin(β)"  => sin(β),
        "cos(γ)"  => cos(γ),
        "exp(δ)"  => exp(δ),
        "tanh(ζ)" => tanh(ζ),
        "α - β"             => α - β,
        "α + β - γ"         => α + β - γ,
        "α * β * γ * δ"     => α * β * γ * δ,
        "α * β - γ * δ / ζ" => α * β - γ * δ / ζ,
        "u^2 + v^2" => u^2 + v^2,
        "√(u^2 + v^2 + w^2)" => √(u^2 + v^2 + w^2),
        "∂x(α)" => ∂x(α),
        "∂y(∂y(β))" => ∂y(∂y(β)),
        "∂z(∂z(∂z(∂z(γ))))" => ∂z(∂z(∂z(∂z(γ)))),
        "∂x(δ + ζ)" => ∂x(δ + ζ),
        "∂x(v) - δy(u)" => ∂x(v) - ∂y(u),
        "∂z(α * β + γ)" => ∂z(α * β + γ),
        "∂x(u) * ∂y(v) + ∂z(w)" => ∂x(u) * ∂y(v) + ∂z(w),
        "∂x(α)^2 + ∂y(α)^2 + ∂z(α)^2" => ∂x(α) + ∂y(α) + ∂z(α)^2,
        "∂x(ζ)^4 + ∂y(ζ)^4 + ∂z(ζ)^4 + 2*∂x(∂x(∂y(∂y(ζ)))) + 2*∂x(∂x(∂z(∂z(ζ)))) + 2*∂y(∂y(∂z(∂z(ζ))))"
            => ∂x(ζ)^4 + ∂y(ζ)^4 + ∂z(ζ)^4 + 2*∂x(∂x(∂y(∂y(ζ)))) + 2*∂x(∂x(∂z(∂z(ζ)))) + 2*∂y(∂y(∂z(∂z(ζ))))
    )

    for (i, (test_name, op)) in enumerate(test_cases)
        computed_field = ComputedField(op)

        compute!(computed_field)  # warmup

        @info "Running abstract operation benchmark: $test_name..."

        trial = @benchmark begin
            @sync_gpu compute!($computed_field)
        end samples=10

        suite[(Arch, i, test_name)] = trial
    end
end

df = benchmarks_dataframe(suite)
sort!(df, :ID)
benchmarks_pretty_table(df, title="Abstract operations benchmarks")

for Arch in Archs
    suite_arch = speedups_suite(suite[@tagged Arch], base_case=(Arch, 1, "-α"))
    df_arch = speedups_dataframe(suite_arch, slowdown=true)
    sort!(df_arch, :ID)
    benchmarks_pretty_table(df_arch, title="Abstract operations relative peformance ($Arch)")
end
