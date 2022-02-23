# printing
using Printf, Statistics 
# Oceananigans
using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using Oceananigans.Utils
using Oceananigans.AbstractOperations
using Oceananigans.Advection
using Oceananigans.TurbulenceClosures: Vertical, Horizontal

const hydrostatic = false

arch = CPU()
FT   = Float64

## units
const kilometer = 1000 # meters
const day = 86400      # seconds
const meter = 1
const hour = 60

## Domain
const Lx = 1000.0kilometer 
const Ly = 2000.0kilometer
const Lz = 2985

# Discretization
Δt = 300.0
maxΔt = 300.0

end_time = 60day 
advection   = WENO5()
timestepper = :RungeKutta3
# Rough target resolution
Δx = Δy = 5kilometer # 5km
Δz = 100meter
# Multiple of 16 gridpoints
const Nx = Int(192 * 0.5)
const Ny = Int(400 * 0.5)
const Nz = Int(32 * 0.5)
# Create Grid
topology = (Periodic, Bounded, Bounded)
grid = RectilinearGrid(arch, topology=topology, 
                            size=(Nx, Ny, Nz), 
                            x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

# Parameters
const f = -1e-4
const β = 1 * 10^(-11)
coriolis = FPlane(FT, f=f)
coriolis = BetaPlane(FT, f₀ = f, β = β)

α  = 2e-4     # [K⁻¹] Thermal expansion coefficient 
g  = 9.8061   # [m/s²] gravitational constant
cᵖ = 3994.0   # [J/K]  heat capacity
ρ  = 999.8    # [kg/m³] density
const h = 1000.0     # [m] e-folding length scale for northern sponge
const ΔB = 8 * α * g # [m/s²] total change in buoyancy from surface to bottom
eos = LinearEquationOfState(FT, α=α, β=0)
buoyancy = BuoyancyTracer()

κh = 0.5e-5 # [m²/s] horizontal diffusivity
νh = 12.0   # [m²/s] horizontal viscocity
κv = 0.5e-5 # [m²/s] vertical diffusivity
νv = 3e-4   # [m²/s] vertical viscocity

vertical_closure = ScalarDiffusivity(ν = νv,
                                     κ = κv,
                                     isotropy = Vertical())

horizontal_closure = ScalarDiffusivity(ν = νh,
                                       κ = κh,
                                       isotropy = Horizontal())
                                       
parameters = (
    Ly = Ly,                   # y-domain length
    τ = 0.2,                   # [N m⁻²] Zonal stress
    ρ = ρ,                     # [kg / m³]
    μ = 1.1e-3,                # [m/s]  linear drag
    H = Lz,                    # [m]
    h = h,                     # [m]    relexaction profile scale
    ΔB = ΔB,                   # [m/s²] buoyancy jump
    Lz = Lz,                   # [m]
    Lsponge = 900kilometer,   # [m]
    λᵗ = 7.0day,               # [s]
    Qᵇ = 10/(ρ * cᵖ) * α * g,  # [m² / s³]
    Qᵇ_cutoff = Ly * 5/6.      # [m]
)

# Momentum Boundary Conditions
@inline windshape(y, p) = sin( π * y / p.Ly)
@inline windstress(x, y, t, p) = - p.τ / p.ρ * windshape(y, p)
@inline ulineardrag(i, j, grid, clock, state, p) = @inbounds - p.μ * state.u[i, j, 1]
@inline vlineardrag(i, j, grid, clock, state, p) = @inbounds - p.μ * state.v[i, j, 1]

# Zonal Velocity
top_u_bc = BoundaryCondition(Flux, windstress, parameters = parameters)
bottom_u_bc =  BoundaryCondition(Flux, ulineardrag, discrete_form = true, parameters = parameters)
u_bcs = FieldBoundaryConditions(top = top_u_bc, bottom = bottom_u_bc)

# Meridional Velocity
bottom_v_bc =  BoundaryCondition(Flux, vlineardrag, discrete_form = true, parameters = parameters)
v_bcs = FieldBoundaryConditions(bottom = bottom_v_bc)

# Buoyancy Boundary Conditions Forcing. Note: Flux convention opposite of Abernathy
@inline cutoff(j, grid, p ) = grid.yᵃᶜᵃ[j] > p.Qᵇ_cutoff ? -0.0 : 1.0
@inline surface_flux(j, grid, p) = p.Qᵇ * cos(3π * grid.yᵃᶜᵃ[j] / p.Ly) * cutoff(j, grid, p)
@inline relaxation(i, j, grid, clock, state, p) = @inbounds surface_flux(j, grid, p)
top_b_bc = BoundaryCondition(Flux, relaxation, discrete_form = true, parameters = parameters)
b_bcs = FieldBoundaryConditions(top = top_b_bc)

# Save boundary conditions as named tuple
bcs = (b = b_bcs,  u = u_bcs, v = v_bcs,)

# Forcing Functions
# Sponge layers
relu(y) = (abs(y) + y) * 0.5
# Northern Wall Relaxation
@inline relaxation_profile_north(k, grid, p) = p.ΔB * ( exp(grid.zᵃᵃᶜ[k]/p.h) - exp(-p.Lz/p.h) ) / (1 - exp(-p.Lz/p.h))
function Fb_function(i, j, k, grid, clock, state, p)
    return @inbounds - (1/p.λᵗ)  * (state.b[i,j,k] 
        - relaxation_profile_north(k, grid, p)) * relu( (grid.yᵃᶜᵃ[j]-p.Lsponge) / (p.Ly - p.Lsponge))
end
Fb = Forcing(Fb_function, parameters = parameters, discrete_form = true)

# Record forcings
forcings = (b = Fb, ) 

# Convective Parameterization
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0,
                                                                background_κz = 5e-6,
                                                                background_νz = 3e-4)

# Model Setup
if hydrostatic 
    model = HydrostaticFreeSurfaceModel(
            grid = grid,
            free_surface = ImplicitFreeSurface(),
            momentum_advection = WENO5(),
            tracer_advection = WENO5(),
            buoyancy = BuoyancyTracer(),
            coriolis = coriolis,
            closure = (horizontal_closure, vertical_closure, convective_adjustment),
            tracers = (:b,),
            boundary_conditions = bcs,
            forcing = forcings,
            )
else
    model = NonhydrostaticModel(
                    grid = grid,
                coriolis = coriolis,
                buoyancy = buoyancy,
                    closure = (horizontal_closure, vertical_closure),
                    tracers = (:b,),
        boundary_conditions = bcs,
                    forcing = forcings,
                advection = advection,
                timestepper = timestepper,
    )
end

# Timestep
Δt_wizard = TimeStepWizard(cfl = 1.0, Δt = Δt, max_change = 1.05, max_Δt = maxΔt)
cfl = AdvectiveCFL(Δt_wizard)

# Create Simulation
Ni = 1000
function print_progress(simulation)
    model = simulation.model
    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    @printf("[%05.2f%%] i: %d, t: %.2e days, umax: (%6.3e, %6.3e, %6.3e) m/s, CFL: %6.4e, next Δt: %.1e s\n",
            progress, i, t / day, umax, vmax, wmax, cfl(model), Δt_wizard.Δt)
    println(" ")
end

simulation = Simulation(model, Δt=Δt_wizard, 
                        stop_time=end_time,
                        iteration_interval=Ni, 
                        progress=print_progress)
## Run
run!(simulation)