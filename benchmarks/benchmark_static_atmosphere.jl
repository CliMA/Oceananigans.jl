using Printf
using TimerOutputs

using Oceananigans
using Oceananigans.Architectures
using JULES

const timer = TimerOutput()

Ns = (32, 256)
Tvars = (Energy, Entropy)

Archs = [CPU]
@hascuda Archs = [CPU, GPU]

for Arch in Archs, N in Ns, Tvar in Tvars

    bname = "$N×$N×$N [$Arch, $Tvar]"
    @info "Benchmarking $bname..."

    grid = RegularCartesianGrid(size=(N, N, N), extent=(1, 1, 1), halo=(2, 2, 2))
    model = CompressibleModel(architecture=Arch(), grid=grid, thermodynamic_variable=Tvar())
    time_step!(model, 1)  # warmup to compile

    for i in 1:10
        @timeit timer bname time_step!(model, 1)
    end

end

print_timer(timer, title="Static atmosphere benchmarks", sortby=:name)