using Oceananigans
using Oceananigans.Units

N = 64
L = 2000
topo = (Periodic, Bounded, Bounded)
grid = RegularRectilinearGrid(topology=topo, size=(1, N, N), extent=(L, L, L))

use_buoyancy_tracer = false
θ = 60
N² = 1e-5

g̃ = (0, sind(θ), cosd(θ))
if use_buoyancy_tracer
    buoyancy = BuoyancyModel(model=BuoyancyTracer(), gravitational_unit_vector=g̃)
    tracers = :b

    ybc = GradientBoundaryCondition(N²*g̃[2])
    zbc = GradientBoundaryCondition(N²*g̃[3])
    bcs = (b=TracerBoundaryConditions(grid, 
                                      bottom=zbc, top=zbc,
                                      south=ybc, north=ybc,
                                      ),)
    

else
    buoyancy = BuoyancyModel(model=SeawaterBuoyancy(), gravitational_unit_vector=g̃)
    tracers = :T, :S

    α = 2e-4; g₀ = 9.81
    dTdz = N² / (g₀ * α)

    ybc = GradientBoundaryCondition(dTdz*g̃[2])
    zbc = GradientBoundaryCondition(dTdz*g̃[3])
    bcs = (T=TracerBoundaryConditions(grid, 
                                      bottom=zbc, top=zbc,
                                      south=ybc, north=ybc,
                                      ),)
end



model = IncompressibleModel(
           grid = grid,
      advection = WENO5(),
    timestepper = :RungeKutta3,
       buoyancy = buoyancy,
        tracers = tracers,
        closure = IsotropicDiffusivity(ν=0, κ=0),
        boundary_conditions = bcs,
)


if use_buoyancy_tracer
    b₀(x, y, z) = N² * (x*g̃[1] + y*g̃[2] + z*g̃[3])
    set!(model, b=b₀)
else
    T₀(x, y, z) = dTdz * (x*g̃[1] + y*g̃[2] + z*g̃[3])
    set!(model, T=T₀)
end

print_progress(sim) = @info "iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=10seconds, stop_time=4hours, progress=print_progress, iteration_interval=10)

fields = merge(model.velocities, model.tracers)
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, filepath = "tilted_gravity_plume.nc",
                       schedule = TimeInterval(5minutes),
                       mode = "c")

run!(simulation)


using Statistics: mean
if use_buoyancy_tracer
    b_y = ComputedField(∂y(model.tracers.b))
    b_z = ComputedField(∂z(model.tracers.b))
    compute!(b_y)
    compute!(b_z)
    mean(b_y, dims=(1, 2, 3))
    mean(b_z, dims=(1, 2, 3))
    
    println(mean(b_y, dims=(1, 2, 3)))
    println(N² * g̃[2])
    println(N² * g̃[2] ≈ mean(b_y, dims=(1, 2, 3))[1])
    println()
    println(mean(b_z, dims=(1, 2, 3)))
    println(N² * g̃[3])
    println(N² * g̃[3] ≈ mean(b_z, dims=(1, 2, 3))[1])
else
    T_y = ComputedField(∂y(model.tracers.T))
    T_z = ComputedField(∂z(model.tracers.T))
    compute!(T_y)
    compute!(T_z)
    mean(T_y, dims=(1, 2, 3))
    mean(T_z, dims=(1, 2, 3))
    
    println(mean(T_y, dims=(1, 2, 3)))
    println(dTdz * g̃[2])
    println(dTdz * g̃[2] ≈ mean(T_y, dims=(1, 2, 3))[1])
    println()
    println(mean(T_z, dims=(1, 2, 3)))
    println(dTdz * g̃[3])
    println(dTdz * g̃[3] ≈ mean(T_z, dims=(1, 2, 3))[1])

end 
