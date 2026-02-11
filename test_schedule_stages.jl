using Oceananigans
using Oceananigans.TurbulenceClosures: DynamicSmagorinsky
using Oceananigans.Utils: IterationInterval

println("Testing IterationInterval(1) behavior across RK3 stages")
println("=" ^ 60)

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
closure = DynamicSmagorinsky()  # Uses IterationInterval(1) by default
model = NonhydrostaticModel(grid; closure, timestepper=:RungeKutta3)

schedule = closure.coefficient.schedule

println("\nManually checking schedule at different stages:")
println("-" ^ 60)

# Simulate what happens during time stepping
for iteration in 0:2
    for stage in 1:3
        model.clock.iteration = iteration
        model.clock.stage = stage

        result = schedule(model)
        println("Iteration $iteration, Stage $stage: schedule(model) = $result")
    end
    println()
end

println("=" ^ 60)
println("Observation:")
println("If schedule returns true at every stage, then step_closure_prognostics!")
println("will be called 3x per iteration (once per RK3 stage).")
println("\nIf it returns true only at stage 1, it's called 1x per iteration.")
