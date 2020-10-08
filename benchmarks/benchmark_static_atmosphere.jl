using Printf
using TimerOutputs
using Oceananigans
using JULES

const timer = TimerOutput()
const Δt = 1

grid = RegularCartesianGrid(size=(32, 32, 32), extent=(1, 1, 1), halo=(2, 2, 2))
model = CompressibleModel(grid=grid, thermodynamic_variable=Energy())
time_step!(model, Δt)  # warmup to compile

for i in 1:10
    @timeit timer "32×32×32 [CPU, Float64, Energy]" time_step!(model, Δt)
end

model = CompressibleModel(grid=grid, thermodynamic_variable=Entropy())
time_step!(model, Δt)  # warmup to compile

for i in 1:10
    @timeit timer "32×32×32 [CPU, Float64, Entropy]" time_step!(model, Δt)
end

print_timer(timer, title="Static atmosphere benchmarks", sortby=:name)
