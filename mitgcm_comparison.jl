using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using Oceananigans.BuoyancyFormulations: buoyancy_frequency

using GLMakie

Nx = 1
Ny = 32
Nz = 25

z_spacing = [50., 50., 55., 60., 65., 70., 80., 95., 120., 155., 200., 260., 320., 400., 480.]
z_faces = [0, cumsum(z_spacing)...] 
z_faces = reverse( - z_faces) 

grid = RectilinearGrid(size=(Ny-1, 15), y=(0, 31*100kilometers), z=z_faces, topology=(Flat, Bounded, Bounded))
zC = znodes(grid, Center())
yC = ynodes(grid, Center())

Tfile = "../MITgcm/verification/front_relax/input/Tini_+10l.bin"
Sfile = "../MITgcm/verification/front_relax/input/Sini_Ydir.bin"
bfile = "../MITgcm/verification/front_relax/input/bathy_inZ.bin"

function read_field(filename, Nx, Ny, Nz)
    data = Array{Float64}(undef, Nx*Ny*Nz)
    read!(filename, data)
    data = bswap.(data) |> Array{Float64}
    data = reshape(data, Nx, Ny, Nz)
    return reverse(data, dims=3)
end

Tini = read_field(Tfile, Nx, Ny, Nz)
Sini = read_field(Sfile, Nx, Ny, Nz)
bini = read_field(bfile, Nx, Ny, 1)

dir = "../MITgcm/verification/front_relax/run/dy100km/"

# Make video from the MITGCM:
files = readdir(dir)
Tfiles = filter(f -> occursin("T.000", f), files)
Sfiles = filter(f -> occursin("S.000", f), files)
Tfiles = filter(f -> occursin(".data", f), Tfiles)
Sfiles = filter(f -> occursin(".data", f), Sfiles)
sort!(Tfiles)
sort!(Sfiles)

iter = Observable(1)
Tn_mit = @lift(read_field(dir * Tfiles[$iter], Nx, Ny, Nz)[1, 1:end-1, 11:end])
Sn_mit = @lift(read_field(dir * Sfiles[$iter], Nx, Ny, Nz)[1, 1:end-1, 11:end])

fig = Figure(size = (1200, 400))
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
contourf!(ax1, yC, zC, Tn_mit)
contourf!(ax2, yC, zC, Sn_mit)

GLMakie.record(fig, "evolution_mitgcm.mp4", 1:length(Tfiles)) do i
    @info "step $i";
    iter[] = i;
end

visc = (HorizontalScalarDiffusivity(ν=300), VerticalScalarDiffusivity(ν=2e-4, κ=3e-5))
gmcl = IsopycnalSkewSymmetricDiffusivity(κ_skew=1000.0, skew_flux_formulation=Oceananigans.TurbulenceClosures.AdvectiveFormulation())
buoy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-4, haline_contraction=0), gravitational_acceleration=10)
fsef = ImplicitFreeSurface(gravitational_acceleration=10)

model = HydrostaticFreeSurfaceModel(; grid,
                                      buoyancy=buoy,
                                      tracers=(:T, :S),
                                      closure=(visc..., gmcl),
                                      free_surface=fsef,
                                      coriolis=FPlane(f=1e-4),
                                      timestepper=:QuasiAdamsBashforth2,
                                      momentum_advection=Centered(),
                                      tracer_advection=Centered())

set!(model, S=Sini[1, 1:end-1, 11:end], T=Tini[1, 1:end-1, 11:end])

simulation = Simulation(model, Δt=1800, stop_time=100*365days) 

using Printf

wall_clock = Ref(time_ns())

function print_progress(sim)
    (; u, v, w) = model.diffusivity_fields[3]
    T, S = model.tracers
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(abs, u), maximum(abs, v), maximum(abs, w), maximum(abs, T), maximum(abs, S), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, print_progress, IterationInterval(1000))

u, v, w = model.velocities
v = model.diffusivity_fields[3].v
w = model.diffusivity_fields[3].w

T, S = model.tracers
N² = Field(buoyancy_frequency(model))
b = BuoyancyField(model)

outputs = (; u, v, w, b, T, S, N²)

simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              schedule = TimeInterval(365days),
                                              filename = "instantaneous_timeseries",
                                              overwrite_existing = true)

run!(simulation)

T = FieldTimeSeries("instantaneous_timeseries.jld2", "T")
S = FieldTimeSeries("instantaneous_timeseries.jld2", "S")

Tn = @lift(interior(T[$iter], 1, :, :))
Sn = @lift(interior(S[$iter], 1, :, :))

fig = Figure(size = (1200, 1000))
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
contourf!(ax1, yC, zC, Tn)
contourf!(ax2, yC, zC, Sn)

GLMakie.record(fig, "evolution_oceananigans.mp4", 1:length(T)) do i
    @info "step $i";
    iter[] = i;
end

dT = @lift($Tn - $Tn_mit)
dS = @lift($Sn - $Sn_mit)

fig = Figure(size = (1200, 400))
ax1 = Axis(fig[1, 2], title="Temperature Oceananigans")
ax2 = Axis(fig[1, 3], title="Salinity Oceananigans")
ax3 = Axis(fig[2, 2], title="Temperature MITgcm")
ax4 = Axis(fig[2, 3], title="Salinity MITgcm")
ax5 = Axis(fig[3, 2], title="Temperature difference")
ax6 = Axis(fig[3, 3], title="Salinity difference")
cf = contourf!(ax1, yC, zC, Tn)
Colorbar(fig[1, 1], cf, label="°C")
cf = contourf!(ax2, yC, zC, Sn)
Colorbar(fig[1, 4], cf, label="psu")
cf = contourf!(ax3, yC, zC, Tn_mit)
Colorbar(fig[2, 1], cf, label="°C")
cf = contourf!(ax4, yC, zC, Sn_mit)
Colorbar(fig[2, 4], cf, label="psu")
cf = contourf!(ax5, yC, zC, dT, colormap=:balance)
Colorbar(fig[3, 1], cf, label="°C")
cf = contourf!(ax6, yC, zC, dS, colormap=:balance)
Colorbar(fig[3, 4], cf, label="psu")

GLMakie.record(fig, "combined_evolution.mp4", 1:length(Tfiles)) do i
    @info "step $i";
    iter[] = i;
end