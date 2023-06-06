using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, PartialCellBottom
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Printf
using CUDA
using CairoMakie

arch = CPU()

tracer_advection = CenteredSecondOrder()

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

use_partial_cells = true
if use_partial_cells
    grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(seamount, 0.1))
else
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(seamount))
end

# Terrain following coordinate
ζ(y, z) = z / (h(y) - 1)

# Calculate streamfunction
Ψᵢ(x, y, z) = (1 - ζ(y, z))^2
Ψ = Field{Center, Face, Face}(grid)
set!(Ψ, Ψᵢ)
fill_halo_regions!(Ψ)
mask_immersed_field!(Ψ)

# Set velocity field from streamfunction
v = YFaceField(grid)
w = ZFaceField(grid)
v .= + ∂z(Ψ)
w .= - ∂y(Ψ)

fill_halo_regions!((v, w))
mask_immersed_field!(v)
mask_immersed_field!(w)

D = compute!(Field(∂y(v) + ∂z(w)))
@info @sprintf("Maximum divergence is %.2e.", maximum(D))

## Set up Model
velocities = PrescribedVelocityFields(; v, w)
model = HydrostaticFreeSurfaceModel(; grid, velocities, tracer_advection,
                                    tracers = :θ,
                                    buoyancy = nothing)

θᵢ(x, y, z) = 1 + z
set!(model, θ = θᵢ)

run_simulation = true

if run_simulation

    # Simulation                             
    stop_time = 1.0
    Δy = grid.Δyᵃᶜᵃ
    @show Δt = 1e-2 * Δy
    simulation = Simulation(model; Δt, stop_time)

    # Diagnostics
    θ   = model.tracers.θ
    θ²  = compute!(Field(Average(θ^2)))
    θ²ᵢ = θ²[1, 1, 1]

    # Residual of variance equation (×2)
    θ²D = compute!(Field(Average(θ^2 * D)))

    # Now, we compute the first set of terms within parenthesis on the right-hand side of the second equation in 
    # Section 2b of https://journals.ametsoc.org/view/journals/mwre/125/9/1520-0493_1997_125_2293_rotbsc_2.0.co_2.xml.
    # The equation is derived in Appendix B, using results e.g. (A8) and (A9) from Appendix A.
    @inline square(i, j, k, grid, θ) = @inbounds θ[i, j, k]^2
    @inline Sʸ(i, j, k, grid, θ) = 2 * ℑyᵃᶠᵃ(i, j, k, grid, θ)^2 - ℑyᵃᶠᵃ(i, j, k, grid, square, θ)
    @inline Sᶻ(i, j, k, grid, θ) = 2 * ℑzᵃᵃᶠ(i, j, k, grid, θ)^2 - ℑzᵃᵃᶠ(i, j, k, grid, square, θ)
    Sʸθ = KernelFunctionOperation{Center, Face, Center}(Sʸ, grid, θ)
    Sᶻθ = KernelFunctionOperation{Center, Center, Face}(Sᶻ, grid, θ)
    R_op = ∂y(v * Sʸθ) + ∂z(w * Sᶻθ)
    Rθ² = compute!(Field(Average(R_op)))

    function progress(s)
        compute!(θ²)
        θ²ₙ = θ²[1, 1, 1]
        Δθ² = (θ²ᵢ - θ²ₙ) / θ²ᵢ

        compute!(θ²D)
        compute!(Rθ²)

        msg = @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|θ|: %.2e, Δ⟨θ²⟩: %.2e",100 * time(s) / s.stop_time,
                       iteration(s), time(s), maximum(abs, s.model.tracers.θ), Δθ²)

        msg *= @sprintf(", ⟨θ²D⟩: %.2e, ⟨Rθ²⟩: %.2e", θ²D[1, 1, 1], Rθ²[1, 1, 1])

        @info msg

        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    simulation.output_writers[:fields] = JLD2OutputWriter(model, (θ = model.tracers.θ,),
                                                          schedule = TimeInterval(0.02),
                                                          filename = "tracer_advection_over_bump",
                                                          overwrite_existing = true)

    run!(simulation)

end

# # A neat movie

# We open the JLD2 file, and extract the `grid` and the iterations we ended up saving at.

filename = "tracer_advection_over_bump.jld2"

θ_timeseries = FieldTimeSeries(filename, "θ"; architecture = CPU())

times = θ_timeseries.times

xGrid, yGrid, zGrid = nodes(θ_timeseries[1])

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

n = Observable(1)

θGrid = @lift interior(θ_timeseries[$n], 1, :, :)

extrema_reduction_factor = 1.0

θlims = extrema(θ_timeseries.data) .* extrema_reduction_factor

fig = Figure(resolution = (850, 750))
title_θ = @lift "Tracer Concentration after " *string(round(times[$n], digits = 1))*" seconds"
ax_θ = Axis(fig[1,1]; xlabel = "Meridional Distance (m)", ylabel = "Depth (m)", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = title_θ, titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_θ = heatmap!(ax_θ, yGrid, zGrid, θGrid; colorrange = θlims, colormap = :balance)
Colorbar(fig[1,2], hm_θ; label = "Tracer Concentration", labelsize = 22.5, labelpadding = 10.0, ticksize = 17.5)

frames = 1:length(times)

CairoMakie.record(fig, filename * ".mp4", frames, framerate = 8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end

nothing # hide