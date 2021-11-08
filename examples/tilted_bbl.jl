using Oceananigans
using Oceananigans.Units
using Printf
using CUDA

params = (f₀ = 1e-4, #1/s
          V∞ = 0.1, # m/s
          N²∞ = 1e-5, # 1/s²
          θ_rad = 0.05,
          Lx = 1000, # m
          Lz = 100, # m
          Nx = 64,
          Nz = 64,
          ν = 5e-4, # m²/s
          sponge_frac = 1/5,
          sponge_rate = √1e-5, # 1/s
          z_0 = 0.1, # m (roughness length)
          )

arch = CPU()
ĝ = [sin(params.θ_rad), 0, cos(params.θ_rad)]



#++++ Grid
S = 1.3
z_faces(k) = params.Lz*(1 + tanh(S * ( (k - 1) / params.Nz- 1)) / tanh(S))

topo = (Periodic, Flat, Bounded)
grid_reg = RegularRectilinearGrid(topology=topo,
                              size=(params.Nx, params.Nz),
                              x=(0, params.Lx), z=(0, params.Lz),
                              halo=(3,3),
                              )
                          
S = 1.3
z_faces(k) = params.Lz*(1 + tanh(S * ( (k - 1) / (params.Nz) - 1)) / tanh(S))
grid_str = VerticallyStretchedRectilinearGrid(topology=topo,
                                              architecture = arch,
                                              size=(params.Nx, params.Nz),
                                              x=(0, params.Lx), z_faces=z_faces,
                                              halo=(3,3),
                                              )
grid = grid_str
@info grid
#----




#++++ Buoyancy model and background
buoyancy = Buoyancy(model=BuoyancyTracer(), vertical_unit_vector=ĝ)

b∞(x, y, z, t, p) = p.N²∞ * (x * sin(p.θ_rad) + z * cos(p.θ_rad))
B_field = BackgroundField(b∞, parameters=(; params.N²∞, params.θ_rad))

db∞dz = params.N²∞ * ĝ[3]
grad_bc_b = GradientBoundaryCondition(-db∞dz) # db/dz + db∞/dz = 0 @ z=0
b_bcs = FieldBoundaryConditions(bottom = grad_bc_b)
#----


#+++++ Boundary Conditions
#+++++ Bottom Drag (Implemented as in https://doi.org/10.1029/2005WR004685)
const κ = 0.4 # von Karman constant
z₁ = CUDA.@allowscalar znodes(Center, grid)[1]
cᴰ = (κ / log(z₁/params.z_0))^2 # quadratic drag coefficient

@info "Adding drag"
@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)

@info "Defining drag BC"
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(cᴰ=cᴰ, V∞=params.V∞))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(cᴰ=cᴰ, V∞=params.V∞))

@info "Defining u, v BCs"
u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

V_bg(x, y, z, t, p) = p.V∞
V_field = BackgroundField(V_bg, parameters=(; V∞=params.V∞))
#-----

bcs = (u=u_bcs,
       v=v_bcs,
       b=b_bcs,
       )
#-----


#++++ Sponge layer definition
@info "Defining sponge layer"
@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))

const sp_frac = params.sponge_frac
const Lz = params.Lz

function top_mask_2nd(x, y, z)
    z₁ = +Lz; z₀ = z₁ - Lz * sp_frac 
    return heaviside((z - z₀)/(z₁ - z₀)) * ((z - z₀)/(z₁ - z₀))^2
end

full_sponge_0 = Relaxation(rate=params.sponge_rate, mask=top_mask_2nd, target=0)
forcing = (u=full_sponge_0, v=full_sponge_0, w=full_sponge_0,)
#----


#++++ Turbulence closure
closure = IsotropicDiffusivity(ν=params.ν, κ=params.ν)
#----


#++++ Model and ICs
@info "Creating model"
model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            architecture = arch,
                            advection = UpwindBiasedFifthOrder(),
                            buoyancy = buoyancy,
                            coriolis = ConstantCartesianCoriolis(f=params.f₀, rotation_axis=ĝ),
                            tracers = :b,
                            closure = closure,
                            boundary_conditions = bcs,
                            background_fields = (b=B_field, v=V_field,),
                            forcing=forcing,
                           )
@info "" model
if has_cuda() run(`nvidia-smi`) end

noise(z, kick) = kick * randn() * exp(-z / (params.Lz/5))
u_ic(x, y, z) = noise(z, 1e-3)
set!(model, b=0, u=u_ic, w=u_ic)

ū = sum(model.velocities.u.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
v̄ = sum(model.velocities.v.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
w̄ = sum(model.velocities.w.data.parent) / (grid.Nx * grid.Ny * grid.Nz)

model.velocities.u.data.parent .-= ū
model.velocities.v.data.parent .-= v̄
model.velocities.w.data.parent .-= w̄
#----


#++++ Create simulation
using Oceananigans.Grids: min_Δz
if ndims==3
    cfl=0.85
else
    cfl=0.5
end
wizard = TimeStepWizard(Δt=0.5*min_Δz(grid)/params.V∞, max_change=1.03, cfl=cfl)

# Print a progress message
start_time = 1e-9*time_ns()
progress_message(sim) =
    @printf("i: %04d, t: %s, Δt: %s, wmax = %.1e ms⁻¹, wall time: %s\n",
            sim.model.clock.iteration, prettytime(model.clock.time),
            prettytime(wizard.Δt), maximum(abs, sim.model.velocities.w),
            prettytime((time_ns() - start_time) * 1e-9))

simulation = Simulation(model, Δt=wizard, 
                        stop_time=3days, 
                        wall_time_limit=23.5hours,
                        progress=progress_message,
                        iteration_interval=2,
                        stop_iteration=Inf,
                       )
#----


#++++ Outputs
u, v, w = model.velocities
b_tot = ComputedField(model.tracers.b + model.background_fields.tracers.b)
v_tot = ComputedField(v + model.background_fields.velocities.v)
ω_y = ComputedField(∂z(u)-∂x(w))
fields = merge((; u, v_tot, w, b_tot, ω_y))
simulation.output_writers[:fields] =
NetCDFOutputWriter(model, fields, filepath = "out.tilted_bbl.nc",
                   schedule = TimeInterval(20minutes),
                   mode = "c")
#----
#

@info "Starting simulation"
start_time = time_ns() # so we can print the total elapsed wall time
run!(simulation)
