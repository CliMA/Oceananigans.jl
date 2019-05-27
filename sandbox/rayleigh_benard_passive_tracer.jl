using Oceananigans, Printf

include("utils.jl")

function makeplot(axs, model)
    sca(axs[1])
    plotxzslice(model.tracers.S, cmap="RdBu_r")
    title("Passive tracer")

    sca(axs[2])
    plotxzslice(model.tracers.T, cmap="RdBu_r")
    title("Buoyancy")

    axs[1].axis("off"); axs[1].set_aspect(1)
    axs[2].axis("off"); axs[2].set_aspect(1)
    axs[1].tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    axs[2].tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    tight_layout()

    return nothing
end

#
# Parameters
#

      α = 2                 # aspect ratio
      n = 1                 # resolution multiple
     Ra = 1e6               # Rayleigh number
Nx = Ny = 16n * α           # horizontal resolution
Lx = Ly = 1.0 * α           # horizontal extent
     Nz = 16n               # vertical resolution
     Lz = 1.0               # vertical extent
     Pr = 0.7               # Prandtl number
      a = 1e-1              # noise amplitude for initial condition
     Δb = 1.0               # buoyancy differential

# Rayleigh and Prandtl determine transport coefficients
ν = sqrt(Δb * Pr * Lz^3 / Ra)
κ = ν / Pr

# 
# Model setup
# 

arch = CPU()
@hascuda arch = GPU() # use GPU if it's available

model = Model(
     arch = arch, 
        N = (Nx, Ny, Nz), 
        L = (Lx, Ly, Lz), 
        ν = ν, 
        κ = κ,
      eos = LinearEquationOfState(βT=1., βS=0.),
constants = PlanetaryConstants(g=1., f=0.)
)

# Constant buoyancy boundary conditions on "temperature"
model.boundary_conditions.T.z.top = BoundaryCondition(Value, 0.0)
model.boundary_conditions.T.z.bottom = BoundaryCondition(Value, Δb)

# Force salinity as a passive tracer (βS=0)
S★(x, z) = exp(4z) * sin(2π/Lx * x)
FS(grid, u, v, w, T, S, i, j, k) = 1/10 * (S★(grid.xC[i], grid.zC[k]) - S[i, j, k])
model.forcing = Forcing(FS=FS)

ArrayType = typeof(model.velocities.u.data)
Δt = 0.01 * min(model.grid.Δx, model.grid.Δy, model.grid.Δz)^2 / ν

#prefix = @sprintf("rayleigh_benard_Ra%d_", Ra)
#nc_writer = NetCDFOutputWriter(dir=".", prefix=prefix, frequency=100)
#push!(model.output_writers, nc_writer)

# 
# Initial condition setup for creating regression test data
#

ξ(z) = a * rand() * z * (Lz + z) # noise, damped at the walls
b₀(x, y, z) = (ξ(z) - z) / Lz

x, y, z = model.grid.xC, model.grid.yC, model.grid.zC 
x, y, z = reshape(x, Nx, 1, 1), reshape(y, 1, Ny, 1), reshape(z, 1, 1, Nz)

model.tracers.T.data .= ArrayType(b₀.(x, y, z))

fig, axs = subplots(nrows=2)

@printf("""
    Crunching Rayleigh-Benard convection with

        N : %d, %d, %d
       Ra : %.0e

    Let's spin the gears.

""", Nx, Ny, Nz, Ra)

for i = 1:100
    walltime = @elapsed time_step!(model, 1000, Δt)

    makeplot(axs, model)
    gcf()

    wb = buoyancy_flux(model)
    @printf("i: %d, t: %.2e, CFL: %.4f, Nuʷᵇ: %.3f, wall: %s\n", model.clock.iteration,
                model.clock.time, cfl(Δt, model), 1 + mean(wb.data) * Lz^2 / (κ*Δb), 
                prettytime(1e9*walltime))
end
