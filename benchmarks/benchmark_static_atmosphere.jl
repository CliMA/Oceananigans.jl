using Printf
using TimerOutputs
using Oceananigans
using JULES

const timer = TimerOutput()

grid = RegularCartesianGrid(size=(32, 32, 32), length=(1, 1, 1), halo=(2, 2, 2))
model = CompressibleModel(grid=grid, thermodynamic_variable=PrognosticS())
time_step!(model, Δt=1, Nt=2)  # warmup to compile

for i in 1:10
    @timeit timer "32×32×32 [CPU, Float64]" time_step!(model, Δt=1, Nt=1)
end

print_timer(timer, title="Static atmosphere benchmarks", sortby=:name)
