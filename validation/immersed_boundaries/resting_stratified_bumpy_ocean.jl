using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, PartialCellBottom
using Printf
using GLMakie

arch = CPU()
tracer_advection = CenteredSecondOrder()
momentum_advection = CenteredSecondOrder()

underlying_grid = RectilinearGrid(arch,
                                  size=(128, 64), halo=(3, 3), 
                                  y = (-1, 1),
                                  z = (-1, 0),
                                  topology=(Flat, Periodic, Bounded))

# A bump
h₀ = 0.5 # bump height
L = 0.25 # bump width
@inline h(y) = h₀ * exp(- y^2 / L^2)
@inline seamount(x, y) = - 1 + h(y)

seamount_field = Field{Center, Center, Nothing}(underlying_grid)
set!(seamount_field, seamount)
fill_halo_regions!(seamount_field)

minimum_fractional_partial_Δz = 0.2
immersed_boundaries = [
                       PartialCellBottom(seamount_field.data;
                                         minimum_fractional_partial_Δz),
                       GridFittedBottom(seamount_field.data)
                      ]

b = []
v = []

function progress(sim)
    vmax = maximum(abs, sim.model.velocities.v)
    @info @sprintf("Iter: %d, time: %.2e, max|v|: %.2e",
                   iteration(sim), time(sim), vmax)

    return nothing
end

for ib in immersed_boundaries
    grid = ImmersedBoundaryGrid(underlying_grid, ib)

    @show grid

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracer_advection,
                                        momentum_advection,
                                        coriolis = FPlane(f=0.1),
                                        tracers = :b,
                                        buoyancy = BuoyancyTracer())

    N² = 1
    bᵢ(x, y, z) = N² * z
    set!(model, b = bᵢ)

    simulation = Simulation(model; Δt=1e-3, stop_iteration=1000)
    simulation.callbacks[:p] = Callback(progress, IterationInterval(100))


    run!(simulation)

    push!(b, Array(interior(model.tracers.b, 1, :, :)))
    push!(v, Array(interior(model.velocities.v, 1, :, :)))
end

b_partial = b[1]
b_full    = b[2]
Δb = b_full .- b_partial

v_partial = v[1]
v_full    = v[2]
Δv = v_full .- v_partial

fig = Figure(resolution=(1200, 1800))

partial_cell_title = @sprintf("PartialCellBottom with ϵ = %.1f",
                              minimum_fractional_partial_Δz)
ax_bp = Axis(fig[1, 2], title=partial_cell_title)
ax_bf = Axis(fig[2, 2], title="GridFittedBottom")
ax_bd = Axis(fig[3, 2], title="Difference (GridFitted - PartialCell)")

# ax_vp = Axis(fig[1, 3])
# ax_vf = Axis(fig[2, 3])
# ax_vd = Axis(fig[3, 3])

color = (:black, 0.5)
linewidth = 3
levels = 15

hmbp = heatmap!(ax_bp, b_partial)
contour!(ax_bp, b_partial; levels, color, linewidth)
Colorbar(fig[1, 1], hmbp, label="Buoyancy", flipaxis=false)

hmbf = heatmap!(ax_bf, b_full)
contour!(ax_bf, b_full; levels, color, linewidth)
Colorbar(fig[2, 1], hmbf, label="Buoyancy", flipaxis=false)

hmbd = heatmap!(ax_bd, Δb)
Colorbar(fig[3, 1], hmbd, label="Buoyancy", flipaxis=false)

#=
hmvp = heatmap!(ax_vp, v_partial)
#contour!(ax_vp, v_partial, levels=15)
#Colorbar(fig[1, 1], hmvp)

hmvf = heatmap!(ax_vf, v_full)
#contour!(ax_vf, v_full, levels=15)
#Colorbar(fig[2, 1], hmvf)

hmvd = heatmap!(ax_vd, Δv)
#Colorbar(fig[3, 1], hmvd)
=#

display(fig)

