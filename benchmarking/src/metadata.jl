#####
##### System metadata
#####

struct BenchmarkMetadata
    julia_version::String
    oceananigans_version::String
    architecture::String
    gpu_name::Union{String, Nothing}
    cuda_version::Union{String, Nothing}
    cpu_model::String
    num_threads::Int
    hostname::String
    package_versions::Dict{String, String}
    timestamp::DateTime
end

# Packages whose versions matter for the timings; only those loaded in the session are recorded
const TRACKED_PACKAGES = ("CUDA", "GPUCompiler", "GPUArrays", "KernelAbstractions", "Adapt", "LLVM", "Reactant")

function loaded_package_versions(names = TRACKED_PACKAGES)
    versions = Dict{String, String}()
    for (id, mod) in Base.loaded_modules
        id.name in names || continue
        v = pkgversion(mod)
        isnothing(v) || (versions[id.name] = string(v))
    end
    return versions
end

function BenchmarkMetadata(arch)
    gpu_name = nothing
    cuda_version = nothing

    if arch isa GPU
        try
            gpu_name = CUDA.name(CUDA.device())
            cuda_version = string(CUDA.runtime_version())
        catch
            gpu_name = "Unknown GPU"
            cuda_version = "Unknown"
        end
    end

    cpu_model = "$(Sys.cpu_info()[1].model) ($(Sys.CPU_NAME))"

    return BenchmarkMetadata(
        string(VERSION),
        string(pkgversion(Oceananigans)),
        string(typeof(arch)),
        gpu_name,
        cuda_version,
        cpu_model,
        Threads.nthreads(),
        gethostname(),
        loaded_package_versions(),
        now(UTC)
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::BenchmarkMetadata)
    println(io, "BenchmarkMetadata")
    println(io, "├── julia_version: ", m.julia_version)
    println(io, "├── oceananigans_version: ", m.oceananigans_version)
    println(io, "├── architecture: ", m.architecture)
    if !isnothing(m.gpu_name)
        println(io, "├── gpu_name: ", m.gpu_name)
        println(io, "├── cuda_version: ", m.cuda_version)
    end
    println(io, "├── cpu_model: ", m.cpu_model)
    println(io, "├── num_threads: ", m.num_threads)
    println(io, "├── hostname: ", m.hostname)
    for name in sort!(collect(keys(m.package_versions)))
        println(io, "├── ", name, ": ", m.package_versions[name])
    end
    print(io,   "└── timestamp: ", m.timestamp)
end
