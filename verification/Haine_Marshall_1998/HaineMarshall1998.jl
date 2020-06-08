using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.Utils

arch = CPU()
FT   = Float64

Lx = 50kilometer
Ly = 30kilometer
Lz = 2kilometer

Δx = Δy = 250meter
Δz = 40meter

Nx = Int(Lx / Δx)
Ny = Int(Ly / Δy)
Nz = Int(Lz / Δz)

topology = (Periodic, Bounded, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

f = 1e-4
coriolis = FPlane(FT, f=f)

α = 2e-4  # Thermal expansion coefficient [K⁻¹]
eos = LinearEquationOfState(FT, α=α, β=0)
buoyancy = SeawaterBuoyancy(FT, equation_of_state=eos, constant_salinity=true)

κh = νh = 5.0   # Horizontal diffusivity and viscosity [m²/s]
κv = νv = 0.02  # Vertical diffusivity and viscosity [m²/s]
closure = ConstantAnisotropicDiffusivity(FT, νh=νh, νv=νv, κh=κh, κv=κv)

B_params = (
    Ly = Ly,
    B½ = 1.96e-7,    # Buoyancy flux at midchannel [m²/s³]
    Lᶠ = 10kilometer # Characteristic length scale of the forcing [m]
)
B(x, y, p) = p.B½ * (tanh(2 * (y - p.Ly/2) / p.Lᶠ) + 1)  # Surface buoyancy flux [m²/s³]
B_bf = BoundaryFunction{:z, Cell, Cell}(B, B_params)
top_b_bc = FluxBoundaryCondition(B_bf)
b_bcs = TracerBoundaryConditions(grid, top=top_b_bc)

top_C_bc = ValueBoundaryCondition(1.0)
C_bcs = TracerBoundaryConditions(grid, top=top_C_bc)

model = IncompressibleModel(
           architecture = arch,
             float_type = FT,
                   grid = grid,
               coriolis = coriolis,
               buoyancy = buoyancy,
                closure = closure,
                tracers = (:b, :C),
    boundary_conditions = (b=b_bcs,)
)

Tₛ  = 12.0     # Surface temperature [°C]
Nₜₕ = 8.37e-4  # Uniform vertical stratification [s⁻¹]
B₀(x, y, z) = Nₜₕ^2 * z

set!(model, b=B₀, C=0.0)

