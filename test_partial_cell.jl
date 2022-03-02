using Revise

using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, PartialCellBottom
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Plots
using JLD2
using LinearAlgebra
using Oceananigans.ImmersedBoundaries: solid_node
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

include("needed_functions.jl")

h0, L = 0.5, 0.25

grid = RectilinearGrid(size=(16, 8), y=(-1, 1), z=(-1, 0),
                       topology=(Flat, Periodic, Bounded), halo=(3,3))

seamount(x, y) = - 1 + h0*exp(-y^2/L^2)
grid_with_seamount = ImmersedBoundaryGrid(grid, PartialCellBottom(seamount, 0.1))

#seamount(x, y, z) = z < - 1 + h0*exp(-y^2/L^2)
#grid_with_seamount = ImmersedBoundaryGrid(grid, GridFittedBoundary(seamount))

c = CenterField(grid_with_seamount)
c .= 1
mask_immersed_field!(c)
x, y, z = nodes(c)
plt = heatmap(y, z, interior(c)[1,:,:]', xlabel = "y", ylabel = "z", title = "Masked Region")

Ψ = Field{Center, Face, Face}(grid_with_seamount)
V = YFaceField(grid_with_seamount)
W = ZFaceField(grid_with_seamount)
B = Field{Center, Center, Center}(grid_with_seamount)

h(y)    = h0*exp(-y^2/L^2)
ζ(y, z) = z/(h(y) - 1)

set!(Ψ, (x, y, z) -> (1 - ζ(y, z))^2)
set!(B, (x, y, z) -> (1.0 + z) )

fill_halo_regions!(Ψ, CPU())
fill_halo_regions!(B, CPU())

mask_immersed_field!(Ψ)
mask_immersed_field!(B)

velocity_method = "numerical"
#velocity_method = "analytical"

if velocity_method == "numerical"
    print("Velocity is computed numerically. \n\n")
    V.=  ∂z(Ψ)
    W.= -∂y(Ψ)
elseif velocity_method == "analytical"
    print("Velocity is specified analytically. \n\n")
    set!(V, (x, y, z) -> -2 * (1 - ( z / (-1+h0*exp(-y^2/L^2)) ) )/(-1+h0*exp(-y^2/L^2)))
    set!(W, (x, y, z) -> 4*h0*z*y*exp(-y^2/L^2)*(1 - z/(-1 + h0*exp(-y^2/L^2)))/((L^2)*(-1 + h0*exp(-y^2/L^2))^2))
else
    print("must select a valid method! \n\n")
end

fill_halo_regions!(V, CPU())
fill_halo_regions!(W, CPU())

mask_immersed_field!(V)
mask_immersed_field!(W)

D = Field{Center, Center, Center}(grid_with_seamount)
D .= ∂y(V) + ∂z(W)
fill_halo_regions!(D, CPU())

@info """
    Maximum of pointwise divergence is $(maximum(interior(D)[1,:,:])).
"""

stop_time = 1.0
Δt = 1e-3

advection_schemes = (
    #UpwindBiasedFirstOrder(),
    CenteredSecondOrder(),
    #UpwindBiasedThirdOrder(),
    #CenteredFourthOrder(),
    #UpwindBiasedFifthOrder(),
    #WENO5()
    )

sim_times      = Dict()
tracer_errors  = Dict()
tracer2_errors = Dict()

for (index, advection_scheme) in enumerate(advection_schemes)

    # Simulate the solution
    print("Begin simulation with advection scheme ", string(nameof(typeof(advection_scheme))), "\n")
    model, simulation, sim_times[index], tracer_initial, tracer2_initial = simulate_advection(V, W, B, advection_scheme, grid_with_seamount, Δt, stop_time, index)

    # Plot final tracer field
    xθ, yθ, zθ = nodes((Center, Center, Center), grid)
    θplot = contourf(yθ, zθ, interior(model.tracers.θ)[1, :, :]'; title="tracer final", xlabel="y", ylabel="z", label=@sprintf("t = %.3f", model.clock.time))
    tracer_title = string("tracer_final_", string(nameof(typeof(advection_scheme))))
    savefig(θplot, tracer_title)

    # Plot of the gradietn of the final tracer
    plot_norm_grad_tracer(yθ, zθ, grid_with_seamount, model, advection_scheme)

    # Make animation
    animation_title = string("flow_over_seamount_", string(nameof(typeof(advection_scheme))))
    visualize_flow_over_seamount_simulation(animation_title)

    # Compute error in tracer conservation
    tracer_final = sum(interior(model.tracers.θ))/(grid.Ny*grid.Nz)
    tracer_errors[index] = (abs(tracer_initial - tracer_final)/tracer_initial)*100

    tracer2_final = sum(interior(model.tracers.θ).^2)/(grid.Ny*grid.Nz)
    tracer2_errors[index] = (abs(tracer2_initial - tracer2_final)/tracer2_initial)*100

    @info """

    Simulation complete with advection scheme $(advection_scheme)
    Simulation time = $(prettytime(sim_times[index]))
    Error in tracer conservation is $(tracer_errors[index]) percent.
    Error in tracer variance is $(tracer2_errors[index]) percent.
    Output: $(abspath(simulation.output_writers[:fields].filepath))

    """

end
