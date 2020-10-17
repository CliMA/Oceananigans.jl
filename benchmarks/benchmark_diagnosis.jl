using BenchmarkTools
using DataFrames
using PrettyTables

using Oceananigans
using Oceananigans.Architectures
using JULES

using BenchmarkTools: prettytime, prettymemory
using JULES: intermediate_thermodynamic_field, compute_temperature!

Archs = [CPU]
@hascuda Archs = [CPU, GPU]

Ns = [32, 64]
@hascuda Ns = [32, 192]

Tvars = [Energy, Entropy]
Gases = [DryEarth, DryEarth3]

tags = ["arch", "N", "gases", "tvar"]

suite = BenchmarkGroup(
    "temperature" => BenchmarkGroup(),
    "pressure" => BenchmarkGroup()
)

function compute_temperature!(model)
    temperature = intermediate_thermodynamic_field(model)

    temperature, total_density, momenta, tracers =
        datatuples(temperature, model.total_density, model.momenta, model.tracers)

    compute_temperature_event =
        launch!(model.architecture, model.grid, :xyz, compute_temperature!,
                temperature, model.grid, model.thermodynamic_variable, model.gases,
                model.gravity, total_density, momenta, tracers,
                dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), compute_p_over_ρ_event)

    return temperature
end

_compute_temperature!(model) = compute_temperature!(model)
_compute_temperature!(model::CompressibleModel{GPU}) = CUDA.@sync compute_temperature!(model)

for Arch in Archs, N in Ns, Gases in Gases, Tvar in Tvars
    @info "Running temperature computation benchmark [$Arch, N=$N, $Tvar, $Gases]..."

    grid = RegularCartesianGrid(size=(N, N, N), extent=(1, 1, 1))
    model = CompressibleModel(architecture=Arch(), grid=grid, thermodynamic_variable=Tvar(),
                              gases=Gases())

    _compute_temperature!(model) # warmup

    b = @benchmark _compute_temperature!(model) samples=10

    key = (Arch, N, Gases, Tvar)
    suite["temperature"][key] = b
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

df = benchmarks_to_dataframe(suite["temperature"])
pretty_table(df, header, title="Temperature computation benchmarks", nosubheader=true)

function gpu_speedups(suite)
    df = DataFrame(size=[], gases=[], thermodynamic_variable=[], speedup=[])

    for N in Ns, Gas in Gases, Tvar in Tvars
        b_cpu = suite[CPU, N, Gas, Tvar] |> median
        b_gpu = suite[GPU, N, Gas, Tvar] |> median
        b_ratio = ratio(b_cpu, b_gpu)

        d = Dict(
            "size" => "$(N)³",
            "gases" => Gas,
            "thermodynamic_variable" => Tvar,
            "speedup" => @sprintf("%.3fx", b_ratio.time)
        )

        push!(df, d)
    end

    return df
end

header = ["Size" "Gases" "ThermoVar" "speedup"]

df = gpu_speedups(suite["temperature"])
pretty_table(df, header, title="Temperature computation speedups", nosubheader=true)
