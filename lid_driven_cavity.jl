using Oceananigans
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Utils
using Printf

grid = RectilinearGrid(size = (256, 256), halo = (3, 3), topology = (Bounded, Bounded, Flat), extent = (1, 1))

u_bcs = FieldBoundaryConditions(north  = ValueBoundaryCondition(1), south = ValueBoundaryCondition(0))

closure = ScalarDiffusivity(ν=1e-6)

for (adv, scheme) in enumerate([WENO5(), VectorInvariant(), WENO5(vector_invariant=true)])

    model = NonhydrostaticModel(grid = grid, 
                            coriolis = nothing,
                           advection = scheme, 
                            buoyancy = nothing, 
                             closure = closure,
                             tracers = (),
                 boundary_conditions = (; u = u_bcs))

    Δt = 0.00137 

    progress(sim) = @printf("Iter: %s, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(model.velocities.u), maximum(model.velocities.v))

    simulation = Simulation(model, Δt = Δt, stop_iteration=100000)

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=2)

    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


    u, v, w = model.velocities
    ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid; computed_dependencies=(u, v))

    ζ = Field(ζ_op)

    name = string(adv)

    outputs = merge(model.velocities, (; ζ))
    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                        schedule = TimeInterval(10Δt),
                                                        prefix = name * "_orig_lid_cavity",
                                                        force = true)
    run!(simulation)
end
#=

using JLD2, GLMakie, Printf

x = range(0, 1, length = 256)

filename1 = "3_orig_lid_cavity"
filename2 = "2_orig_lid_cavity"
filename3 = "1_orig_lid_cavity"

file1 = jldopen(filename1 * ".jld2")
file2 = jldopen(filename2 * ".jld2")
file3 = jldopen(filename3 * ".jld2")

iterations = parse.(Int, keys(file1["timeseries/t"]))

iter = Observable(0)

u1(iter) = file1["timeseries/u/" * string(iter)][1:end-1, :, 1]
u2(iter) = file2["timeseries/u/" * string(iter)][1:end-1, :, 1]
u3(iter) = file3["timeseries/u/" * string(iter)][1:end-1, :, 1]
v1(iter) = file1["timeseries/ζ/" * string(iter)][:, 1:end-1, 1]
v2(iter) = file2["timeseries/ζ/" * string(iter)][:, 1:end-1, 1]
v3(iter) = file3["timeseries/ζ/" * string(iter)][:, 1:end-1, 1]

uw = @lift u1($iter)
vw = @lift v1($iter)
up = @lift u2($iter)
vp = @lift v2($iter)
uu = @lift u3($iter)
vu = @lift v3($iter)

ud = @lift u2($iter) .- u1($iter)
vd = @lift v2($iter) .- v1($iter)

fig = Figure(resolution = (2500, 2000))

fontsize_theme = Theme(fontsize = 25)
set_theme!(fontsize_theme)

ax1 = Axis(fig[1, 1], title="Vector invariant")
heatmap!(ax1, x, x, vp, colormap=:hot, colorrange = (-0.5, 0.5), iterpolate=true)

ax2 = Axis(fig[1, 2], title="Vector invariant, WENO modification")
heatmap!(ax2, x, x, vw, colormap=:hot, colorrange = (-0.5, 0.5), iterpolate=true)

ax2 = Axis(fig[2, 1], title="WENO fifth order")
heatmap!(ax2, x, x, vu, colormap=:hot, colorrange = (-0.5, 0.5), iterpolate=true)

ax2 = Axis(fig[2, 2], title="ax1 - ax2")
heatmap!(ax2, x, x, vd, colormap=:hot, colorrange = (-0.5, 0.5), iterpolate=true)

display(fig)

record(fig, "difference-vort.mp4", iterations[1:end-2], framerate=50) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end 


using JLD2, GLMakie, Printf

x = range(0, 1, length = 64)

filename1 = "WENO_lid_cavity"

file1 = jldopen(filename1 * ".jld2")

iterations = parse.(Int, keys(file1["timeseries/t"]))

iter = Observable(0)

u1(iter) = file1["timeseries/u/" * string(iter)][1:end-1, :, 1]
v1(iter) = file1["timeseries/v/" * string(iter)][:, 1:end-1, 1]

up = @lift u1($iter)
vp = @lift v1($iter)
# uw = @lift u2($iter)
# vw = @lift v2($iter)

# ud = @lift u2($iter) .- u1($iter)
# vd = @lift v2($iter) .- v1($iter)

fig = Figure(resolution = (2500, 2000))

fontsize_theme = Theme(fontsize = 25)
set_theme!(fontsize_theme)

ax1 = fig[1, 1] = LScene(fig) # make plot area wider
heatmap!(ax1, x, x, up, colormap=:hot, colorrange = (-0.3, 0.3), iterpolate=true)

ax2 = fig[1, 2] = LScene(fig) # make plot area wider
heatmap!(ax2, x, x, vp, colormap=:hot, colorrange = (-0.3, 0.3), iterpolate=true)

# ax2 = fig[2, 1] = LScene(fig) # make plot area wider
# heatmap!(ax2, x, x, ud, colormap=:hot, colorrange = (-0.1, 0.1), iterpolate=true)

display(fig)

record(fig, "difference.mp4", iterations[1:end-2], framerate=10) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end 