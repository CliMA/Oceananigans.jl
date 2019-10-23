using Oceananigans, Printf

####
#### Model set-up
####

Nx, Ny, Nz = 16, 16, 16       # No. of grid points in x, y, and z, respectively.
Lx, Ly, Lz = 100, 100, 100    # Length of the domain in x, y, and z, respectively (m).
tf = 5000                     # Length of the simulation (s)

model = Model(grid=RegularCartesianGrid(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz)),
              closure=ConstantIsotropicDiffusivity())

# Add a cube-shaped warm temperature anomaly that takes up the middle 50%
# of the domain volume.
i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

####
#### Set up output
####

write_grid(model)

outputs =      Dict("u" => model.velocities.u,
                    "v" => model.velocities.v,
                    "w" => model.velocities.w,
                    "T" => model.tracers.T,
                    "S" => model.tracers.S)

outputattrib = Dict("u" => ["longname" => "Velocity in the x-direction", "units" => "m/s"],
                    "v" => ["longname" => "Velocity in the y-direction", "units" => "m/s"],
                    "w" => ["longname" => "Velocity in the z-direction", "units" => "m/s"],
                    "T" => ["longname" => "Temperature", "units" => "K"],
                    "S" => ["longname" => "Salinity", "units" => "g/kg"])

globalattrib = Dict("f" => 1e-4, "name" => "Thermal bubble expt 1")

# The following writer saves a yz plane at xC[5] for all fields that
# have xC as their dimension and at xF[6] for all fields that have xF
# as their dimension. Ranges also can be specified (e.g. xC=2:10)
subsetwriter = NetCDFOutputWriter(model, outputs;
                                  interval=10, filename="dump_subset.nc",
                                  outputattrib=outputattrib,
                                  globalattrib=globalattrib,
                                  xC=5, xF=6)
push!(model.output_writers, subsetwriter)

# The following writer saves a data from the entire domain
globalwriter = NetCDFOutputWriter(model, outputs, interval=10,
                                  filename="dump_global.nc")
push!(model.output_writers, globalwriter)

####
#### Run the simulation
####

function terse_message(model, walltime, Δt)
    cfl = Δt / Oceananigans.cell_advection_timescale(model)
    return @sprintf("i: %d, t: %.4f s, Δt: %.1f s, cfl: %.3f, wall time: %s\n",
                    model.clock.iteration, model.clock.time, Δt, cfl, prettytime(walltime))
end

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.2, Δt=10.0, max_change=1.1, max_Δt=50.0)

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 10, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)
end

# Close the NetCDFOutputWriter
OWClose(subsetwriter)
OWClose(globalwriter)
