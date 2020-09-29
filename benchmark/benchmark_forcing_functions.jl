using Printf
using TimerOutputs
using Oceananigans
using Oceananigans.Utils

include("benchmark_utils.jl")

#####
##### Benchmark setup and parameters
#####

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Run benchmark across these parameters.
            Ns = [(128, 128, 128)]
   float_types = [Float64]  # Float types to benchmark.
         archs = [CPU()]    # Architectures to benchmark on.
@hascuda archs = [GPU()]    # Benchmark GPU on systems with CUDA-enabled GPUs.

#####
##### Forcing function definitions
#####

@inline function Fu_params_func(i, j, k, grid, clock, model_fields, params)
    if k == 1
        return @inbounds -2*params.K/grid.Δz^2 * (U.u[i, j, 1] - 0)
    elseif k == grid.Nz
        return @inbounds -2*params.K/grid.Δz^2 * (U.u[i, j, grid.Nz] - 0)
    else
        return 0
    end
end

@inline FT_params_func(i, j, k, grid, time, model_fields, params) = @inbounds ifelse(k == 1, -params.λ * (model_fields.T[i, j, 1] - 0), 0)

Fu_params = Forcing(FT_params_func, discrete_form=true, parameters=(K=0.1,))
FT_params = Forcing(FT_params_func, discrete_form=true, parameters=(λ=1e-4,))

const λ = 1e-4
const K = 0.1

@inline function Fu_consts(i, j, k, grid, clock, model_forcing)
    if k == 1
        return @inbounds -2*K/grid.Δz^2 * (U.u[i, j, 1] - 0)
    elseif k == grid.Nz
        return @inbounds -2*K/grid.Δz^2 * (U.u[i, j, grid.Nz] - 0)
    else
        return 0
    end
end

@inline FT_consts_func(i, j, k, grid, time, model_fields) = @inbounds ifelse(k == 1, -λ * (model_fields.T[i, j, 1] - 0), 0)

Fu_consts = Forcing(FT_consts_func, discrete_form=true)
FT_consts = Forcing(FT_consts_func, discrete_form=true)

#####
##### Run benchmarks
#####

for arch in archs, FT in float_types, N in Ns
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 1, 1, 1

    forced_model_params = Model(architecture = arch,
                                float_type = FT,
		                        grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)),
                                forcing = (Fl=Fu_params, FT=FT_params))

    time_step!(forced_model_params, Ni, 1)  # First 1-2 iterations usually slower.

    bn =  benchmark_name(N, "with forcing (params)", arch, FT)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(forced_model_params, 1, 1)
    end

    forced_model_consts = Model(architecture = arch,
                                float_type = FT,
		                        grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)),
                                forcing=(Fu=Fu_consts, FT=FT_consts))

    time_step!(forced_model_consts, Ni, 1)  # First 1-2 iterations usually slower.

    bn =  benchmark_name(N, "with forcing (consts)", arch, FT)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(forced_model_consts, 1, 1)
    end

    unforced_model = Model(architecture = arch, float_type = FT,
			   grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)))

    time_step!(unforced_model, Ni, 1)  # First 1-2 iterations usually slower.

    bn =  benchmark_name(N, "  no forcing         ", arch, FT)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(unforced_model, 1, 1)
    end
end

#####
##### Print benchmark results
#####

println()
print(versioninfo_with_gpu())
print_timer(timer, title="Forcing function benchmarks")
println()
