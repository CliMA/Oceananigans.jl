using BenchmarkTools
using DataFrames
using PrettyTables

using Oceananigans
using Oceananigans.Architectures
using JULES

using BenchmarkTools: prettytime, prettymemory

Archs = [CPU]
@hascuda Archs = [CPU, GPU]

Ns = [32, 64]
@hascuda Ns = [32, 256]

Tvars = [Energy, Entropy]
Gases = [DryEarth, DryEarth3]

tags = ["arch", "N", "gases", "tvar"]

suite = BenchmarkGroup(
    "cfl" => BenchmarkGroup(),
    "acoustic_cfl" => BenchmarkGroup()
)

for Arch in Archs, N in Ns, Gases in Gases, Tvar in Tvars
    @info "Running CFL benchmark [$Arch, N=$N, $Tvar, $Gases]..."

    grid = RegularCartesianGrid(size=(N, N, N), extent=(1, 1, 1))
    model = CompressibleModel(architecture=Arch(), grid=grid, thermodynamic_variable=Tvar(),
                              gases=Gases())

    # warmup
    cfl(model, 1)
    acoustic_cfl(model, 1)

    b_cfl = @benchmark cfl($model, 1) samples=10
    display(b_cfl)

    b_acfl = @benchmark acoustic_cfl($model, 1) samples=10
    display(b_acfl)

    key = (Arch, N, Gases, Tvar)
    suite["cfl"][key] = b_cfl
    suite["acoustic_cfl"][key] = b_acfl
end

function benchmarks_to_dataframe(suite)
    df = DataFrame(architecture=[], size=[], gases=[],
                   thermodynamic_variable=[], min=[], median=[],
                   mean=[], max=[], memory=[], allocs=[])
    
    for (key, b) in suite
        Arch, N, Gases, Tvar = key

        d = Dict(
            "architecture" => Arch,
            "size" => "$(N)³",
            "gases" => Gases,
            "thermodynamic_variable" => Tvar,
            "min" => minimum(b.times) |> prettytime,
            "median" => median(b.times) |> prettytime,
            "mean" => mean(b.times) |> prettytime,
            "max" => maximum(b.times) |> prettytime,
            "memory" => prettymemory(b.memory),
            "allocs" => b.allocs
        )

        push!(df, d)
    end

    return df
end

header = ["Arch" "Size" "Gases" "ThermoVar" "min" "median" "mean" "max" "memory" "allocs"]

df = benchmarks_to_dataframe(suite["cfl"])
pretty_table(df, header, title="CFL benchmarks", nosubheader=true)

df = benchmarks_to_dataframe(suite["acoustic_cfl"])
pretty_table(df, header, title="Acoustic CFL benchmarks", nosubheader=true)
