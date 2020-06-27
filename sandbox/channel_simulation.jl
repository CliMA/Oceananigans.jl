using Printf
using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using Oceananigans.Utils

arch = CPU()
FT   = Float64

filename_1 = "Windstress_Convection_Example_Constant"
const scale = 20;
Lx = scale * 50kilometer # 1000km
Ly = scale * 50kilometer # 1000km
Lz = 2kilometer  # 3km

Δx = Δy = scale * 250meter # 8x larger?
Δz = 40meter

Nx = Int(Lx / Δx)
Ny = Int(Ly / Δy)
Nz = Int(Lz / Δz)

topology = (Periodic, Bounded, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

f = -1e-4
β = 1 * 10^(-11)
coriolis = FPlane(FT, f=f)
coriolis = BetaPlane(FT, f₀ = f, β = β)

α = 2e-4  # Thermal expansion coefficient [K⁻¹]
eos = LinearEquationOfState(FT, α=α, β=0)
buoyancy = BuoyancyTracer()
# buoyancy = SeawaterBuoyancy(FT, equation_of_state=eos, constant_salinity=true)

κh = νh = scale * 5.0   # Horizontal diffusivity and viscosity [m²/s]
κv = νv = 0.02  # Vertical diffusivity and viscosity [m²/s]
#closure = ConstantAnisotropicDiffusivity(FT, νh=νh, νv=νv, κh=κh, κv=κv)
closure = AnisotropicMinimumDissipation(FT)

bc_params = (
    Ly = Ly,
    B½ = 1.96e-7 / 10,    # Buoyancy flux at midchannel [m²/s³]
    Lᶠ = 10kilometer, # Characteristic length scale of the forcing [m]
	τ = 0.1, # [N m⁻²] Zonal stress
	ρ = 1024 # [kg / m³]
)

@inline wind_stress(x, y, t, p) = p.τ / p.ρ * sin( π*y / p.Ly)
@inline buoyancy_flux(x, y, t, p) = p.B½ * (tanh(2 * (y - p.Ly/2) / p.Lᶠ) + 0.0)  # Surface buoyancy flux [m²/s³], + 1.0 for HM

# Buoyancy
buoyancy_flux_bf = BoundaryFunction{:z, Cell, Cell}(buoyancy_flux, bc_params)
top_b_bc = FluxBoundaryCondition(buoyancy_flux_bf)
b_bcs = TracerBoundaryConditions(grid, top=top_b_bc)

# Zonal Velocity
v_velocity_flux_bf = BoundaryFunction{:z, Cell, Face}(wind_stress, bc_params)
top_v_bc = FluxBoundaryCondition(v_velocity_flux_bf)
bottom_v_bc = ValueBoundaryCondition(-0.0)
v_bcs = VVelocityBoundaryConditions(grid, top = top_v_bc, bottom = bottom_v_bc)

# Meridional Velocity
bottom_u_bc = ValueBoundaryCondition(-0.0)
u_bcs = UVelocityBoundaryConditions(grid, bottom = bottom_u_bc)

top_C_bc = ValueBoundaryCondition(1.0)
C_bcs = TracerBoundaryConditions(grid, top=top_C_bc)

model = IncompressibleModel(
           architecture = arch,
             float_type = FT,
                   grid = grid,
               coriolis = coriolis,
               buoyancy = buoyancy,
                closure = closure,
                tracers = (:b,),
    boundary_conditions = (b = b_bcs, v = v_bcs, u = u_bcs)
)

T_s  = 12.0    # Surface temperature [°C]
N_s = 8.37e-4  # Uniform vertical stratification [s⁻¹]
ε(σ) = σ * randn()
B₀(x, y, z) = N_s^2 * z + ε(1e-8)

set!(model, b=B₀)
# set!(model, b=B₀, C=0.0)

fields = Dict(
    "u" => model.velocities.u,
    "v" => model.velocities.v,
    "w" => model.velocities.w,
    "b" => model.tracers.b
)



surface_output_writer =
    NetCDFOutputWriter(model, fields, filename= filename_1 * "_surface.nc",
			           interval=1hour, zC=Nz, zF=Nz)

middepth_output_writer =
    NetCDFOutputWriter(model, fields, filename= filename_1 * "_middepth.nc",
                       interval=1hour, zC=Int(Nz/2), zF=Int(Nz/2))

zonal_output_writer =
    NetCDFOutputWriter(model, fields, filename= filename_1 * "_zonal.nc",
                       interval=1hour, yC=Int(Ny/2), yF=Int(Ny/2))

meridional_output_writer =
    NetCDFOutputWriter(model, fields, filename= filename_1 * "_meridional.nc",
                       interval=1hour, xC=Int(Nx/2), xF=Int(Nx/2))
###

#bouyancy profile
Uz = ZonalAverage(model.velocities.u; return_type=Array)
Bz = ZonalAverage(model.tracers.b; return_type=Array)
# Create output writer that writes vertical profiles to JLD2 output files.
zonal_averages = Dict(
	"Uz" => model -> Uz(model),
	"Bz" => model -> Bz(model),
)

output_attributes = Dict(
    "Uz" => Dict("longname" => "Zonal Average Velocity in the x-direction", "units" => "m/s"),
    "Bz" => Dict("longname" => "Zonal Average Buoyancy", "units" => "m/s²"),
)
# Should probably output error if this is not supplied for nonfield objects
dimensions = Dict(
	"Uz" => ("xF", "yC", "zC"),
	"Bz" => ("xC", "yC", "zC"),
)

zonal_average_output_writer = NetCDFOutputWriter(model, zonal_averages, filename =  filename_1 * "_zonal_average.nc", interval=1hour, output_attributes=output_attributes, dimensions = dimensions)
###
Δt_wizard = TimeStepWizard(cfl=0.3, Δt=10.0, max_change=1.2, max_Δt= 2 * 600.0)
cfl = AdvectiveCFL(Δt_wizard)


# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 10

function print_progress(simulation)
    model = simulation.model
    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    @printf("[%05.2f%%] i: %d, t: %.2e days, umax: (%6.3e, %6.3e, %6.3e) m/s, CFL: %6.4e, next Δt: %.1e s\n",
    	    progress, i, t / day, umax, vmax, wmax, cfl(model), Δt_wizard.Δt)
end

end_time = 10day
simulation = Simulation(model, Δt=Δt_wizard, stop_time=end_time, progress=print_progress, progress_frequency=Ni)

simulation.output_writers[:surface] = surface_output_writer
simulation.output_writers[:middepth] = middepth_output_writer
simulation.output_writers[:zonal] = zonal_output_writer
simulation.output_writers[:meridional] = meridional_output_writer
simulation.output_writers[:zonal_average] = zonal_average_output_writer
###
run!(simulation)


###
# Define horizontal average diagnostics.
Up = HorizontalAverage(model.velocities.u;       return_type=Array)
Vp = HorizontalAverage(model.velocities.v;       return_type=Array)
Wp = HorizontalAverage(model.velocities.w;       return_type=Array)

u = model.velocities.u
v = model.velocities.v
w = model.velocities.w

uu = HorizontalAverage(u*u, model; return_type=Array)
vv = HorizontalAverage(v*v, model; return_type=Array)
ww = HorizontalAverage(w*w, model; return_type=Array)
uv = HorizontalAverage(u*v, model; return_type=Array)
uw = HorizontalAverage(u*w, model; return_type=Array)
vw = HorizontalAverage(v*w, model; return_type=Array)
# Create output writer that writes vertical profiles to JLD2 output files.
profiles = Dict(
    :vv => model -> vv(model),
)

profile_writer = NetCDFOutputWriter(model, profiles, filename= filename_1 * "_test.nc", interval=1hour)


###
using Oceananigans.Fields
using Oceananigans.Utils: validate_interval
using Oceananigans.Grids: topology, interior_x_indices, interior_y_indices, interior_z_indices
using Oceananigans.Fields: cpudata
using NCDatasets
model = model
outputs = zonal_averages
filename =filename_1 * "_zonal_average_9.nc"

interval = nothing
frequency = nothing
global_attributes = Dict()
# utput_attributes = Dict()
dimensions = Dict()
const zonal_dims = size(Uz(model))
dimensions = Dict(
	"Uz" => ("xC", "yC", "zC"),
	"Bz" => ("xC", "yC", "zC"),
)
clobber = true
compression = 0
xC = interior_x_indices(Cell, model.grid)
xF = interior_x_indices(Face, model.grid)
yC = interior_y_indices(Cell, model.grid)
yF = interior_y_indices(Face, model.grid)
zC = interior_z_indices(Cell, model.grid)
zF = interior_z_indices(Face, model.grid)
###
mode = clobber ? "c" : "a"
validate_interval(interval, frequency)

# Generates a dictionary with keys "xC", "xF", etc, whose values give the slices to be saved.
slice_keywords = Dict(name => a for (name, a) in zip(("xC", "yC", "zC", "xF", "yF", "zF"),
                                                         ( xC,   yC,   zC,   xF,   yF,   zF )))

# Initiates the output file with dimensions
write_grid_and_attributes(model; filename=filename, compression=compression,
                              attributes=global_attributes, mode=mode,
                              xC=xC, yC=yC, zC=zC, xF=xF, yF=yF, zF=zF)

# Opens the same output file for writing fields from the user-supplied variable outputs
dataset = Dataset(filename, "a")

    # Creates an unliimited dimension "time"
defDim(dataset, "time", Inf)
defVar(dataset, "time", Float64, ("time",))
sync(dataset)
###
# Ensure we have an attribute for every output. Use reasonable defaults if
# none were specified by the user.
for c in keys(outputs)
    if !haskey(output_attributes, c)
        output_attributes[c] = default_output_attributes[c]
    end
end

# Initiates empty variables for fields from the user-supplied variable outputs
for (name, output) in outputs
	println(name)
	println(output)
    if output isa Field
        FT = eltype(output.grid)
        defVar(dataset, name, FT, (netcdf_spatial_dimensions(output)..., "time"),
                   compression=compression, attrib=output_attributes[name])
    else
        defVar(dataset, name, Float64, (dimensions[name]..., "time"),
                   compression=compression, attrib=output_attributes[name])
    end
end
sync(dataset)

field_outputs = filter(o -> o.second isa Field, outputs) # extract outputs whose values are Fields

# Store a slice specification for each field.
slices = Dict(name => slice_indices(field; xC=xC, yC=yC, zC=zC, xF=xF, yF=yF, zF=zF) for (name, field) in field_outputs)

NetCDFOutputWriter(filename, dataset, outputs, interval, frequency, clobber, slices, 0.0)


###
netcdf_spatial_dimensions(::Field{LX, LY, LZ}) where {LX, LY, LZ} = xdim(LX), ydim(LY), zdim(LZ)

xdim(::Type{Face}) = "xF"
ydim(::Type{Face}) = "yF"
zdim(::Type{Face}) = "zF"

xdim(::Type{Cell}) = "xC"
ydim(::Type{Cell}) = "yC"
zdim(::Type{Cell}) = "zC"
