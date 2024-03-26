using Oceananigans
using Oceananigans.Models: ShallowWaterModel
   
Nx, Ny = 50, 128
Lx, Ly = 2π, 20
H₀     = 10
α, β   = 0.1, -0.1
Lb, Hb = α * Ly /2, β * H₀

f, g, U  = 1, 1, 1
Δη = -f * U / g 

grid = RectilinearGrid(size = ( Nx, Ny),
                          x = (  0, Lx),
                          y = (-Ly/2, Ly/2),
                   topology = (Periodic, Bounded, Flat))

η(x, y) =   Δη * tanh(y)
b(x, y) = - H₀ + Hb * tanh(y/Lb)  # FJP: sign in front of Hb?

h̄(x, y) = η(x, y) - b(x, y)
ū(x, y) =  U * sech(y)^2
 
 uⁱ(x, y) = ū(x, y) + 1e-4 * exp(-y^2) * randn()
uhⁱ(x, y) = uⁱ(x, y) * h̄(x, y)

ū̄h(x, y) = ū(x, y) * h̄(x, y)

model = ShallowWaterModel(; grid, coriolis = FPlane(f=f), 
                gravitational_acceleration = g,
                               timestepper = :RungeKutta3,
                        momentum_advection = WENO(),
                                bathymetry = b
                        )

set!(model, uh = ū̄h, h = h̄)

uh, vh, h = model.solution

u = uh / h
v = vh / h

# Plot the Geostrophic State 

using CairoMakie

x = xnodes(grid, Center())
y = ynodes(grid, Center())

fig = Figure()
Axis(fig[1, 1], xlabel = "y", ylabel = "depth", title = "Geostrophic State")
lines!(y, h[1, 1:Ny, 1] .+ b.(0, y), linewidth=4, linestyle=:solid, color=:blue, label="η(y)")
lines!(y, 0*y, linewidth=2, linestyle=:dash, color=:blue, label="0")
lines!(y, b.(0, y), linewidth=4, color=:black, label="b(y)")
lines!(y, -H₀ .+ 0*y, linewidth=2, linestyle = :dash, color=:black, label="H₀")
axislegend(position = :lt)

CairoMakie.save("Geostrophic_State.png", fig)

ω = Field(∂x(v) - ∂y(u))
compute!(ω)

ωⁱ = Field((Face, Face, Nothing), model.grid)
ωⁱ .= ω

ω′ = Field(ω - ωⁱ)

set!(model, uh = uhⁱ)

simulation = Simulation(model, Δt = 1e-2, stop_time = 100)

using LinearAlgebra: norm

perturbation_norm(args...) = norm(v)

fields_filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_fields.nc")
simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; ω, ω′),
                                                        filename = fields_filename,
                                                        schedule = TimeInterval(2),
                                                        overwrite_existing = true)

growth_filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_perturbation_norm.nc")
simulation.output_writers[:growth] = NetCDFOutputWriter(model, (; perturbation_norm),
                                                        filename = growth_filename,
                                                        schedule = IterationInterval(1),
                                                        dimensions = (; perturbation_norm = ()),
                                                        overwrite_existing = true)

run!(simulation)

using NCDatasets, Printf, CairoMakie

x, y = xnodes(ω), ynodes(ω)

fig = Figure(size = (1200, 660))

axis_kwargs = (xlabel = "x", ylabel = "y")
ax_ω  = Axis(fig[2, 1]; title = "Total vorticity, ω", axis_kwargs...)
ax_ω′ = Axis(fig[2, 3]; title = "Perturbation vorticity, ω - ω̄", axis_kwargs...)

n = Observable(1)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

times = ds["time"][:]

ω = @lift ds["ω"][:, :, 1, $n]
hm_ω = heatmap!(ax_ω, x, y, ω, colorrange = (-1, 1), colormap = :balance)
Colorbar(fig[2, 2], hm_ω)

ω′ = @lift ds["ω′"][:, :, 1, $n]
hm_ω′ = heatmap!(ax_ω′, x, y, ω′, colormap = :balance)
Colorbar(fig[2, 4], hm_ω′)

title = @lift @sprintf("t = %.1f", times[$n])
fig[1, 1:4] = Label(fig, title, fontsize=24, tellwidth=false)

current_figure() #hide
fig

frames = 1:length(times)

record(fig, "shallow_water_Bickley_jet.mp4", frames, framerate=12) do i
    n[] = i
end

close(ds)

#ds2 = NCDataset(simulation.output_writers[:growth].filepath, "r")
ds2 = NCDataset("shallow_water_Bickley_jet_perturbation_norm.nc", "r")

     t = ds2["time"][:]
norm_v = ds2["perturbation_norm"][:]

close(ds2)

using Polynomials: fit

I = 5000:6000

linear_fit_polynomial = fit(t[I], log.(norm_v[I]), 1, var = :t)

constant, slope = linear_fit_polynomial[0], linear_fit_polynomial[1]

best_fit = @. exp(0.95*constant + slope * t)

fig2 = Figure()
Axis(fig2[1,1], yscale = log10, xlabel = "time", ylabel = "norm(v)", title = "growth of perturbation norm")
lines!(t[2:end], norm_v[2:end];
      linewidth = 4, color=:blue, 
      label = "norm(v)")

lines!(t[I], best_fit[I], 
       linewidth = 4, color=:red,
       label = "best fit")

axislegend(position = :rb)

CairoMakie.save("growth_rate.png", fig2)

println("Numerical growth rate is approximated to be ", round(slope, digits=3), ",\n",
        "which is very close to the theoretical value of 0.139.")
