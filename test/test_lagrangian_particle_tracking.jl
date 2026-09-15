include("dependencies_for_runtests.jl")

using NCDatasets
using StructArrays
using Oceananigans.Architectures: architecture, on_architecture

using Oceananigans.Models.LagrangianParticleTracking: no_dynamics

struct TestParticle{T}
    x::T
    y::T
    z::T
    u::T
    v::T
    w::T
    s::T
end

function particle_tracking_simulation(; grid, particles, timestepper=:RungeKutta3, velocities=nothing)
    Arch = typeof(architecture(grid))

    if grid isa RectilinearGrid
        model = NonhydrostaticModel(grid; timestepper, velocities, particles)
        set!(model, u=1, v=1)
    else
        set!(velocities.u, 1)
        set!(velocities.v, 1)
        model = HydrostaticFreeSurfaceModel(grid; velocities=PrescribedVelocityFields(; velocities...), particles)
    end

    simulation = Simulation(model, Δt=1e-2, stop_iteration=1)

    jld2_filepath = "test_particles_$Arch.jld2"
    simulation.output_writers[:particles_jld2] = JLD2Writer(model, (; particles=model.particles),
                                                            filename = jld2_filepath,
                                                            schedule = IterationInterval(1))

    nc_filepath = "test_particles_$Arch.nc"
    simulation.output_writers[:particles_nc] = NetCDFWriter(model, (; particles=model.particles),
                                                            filename = nc_filepath,
                                                            schedule = IterationInterval(1))

    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(1),
                                                            dir=".", prefix="particles_checkpoint_$Arch")

    return simulation, jld2_filepath, nc_filepath
end

function run_particle_tracking_tests(grid, dynamics, timestepper=:QuasiAdamsBashforth)
    arch = architecture(grid)
    Arch = typeof(arch)
    P = 10

    #####
    ##### Test default particle
    #####

    xs = on_architecture(arch, 0.6 * ones(P))
    ys = on_architecture(arch, 0.58 * ones(P))
    zs = on_architecture(arch, 0.8 * ones(P))

    particles = LagrangianParticles(; x=xs, y=ys, z=zs, dynamics)
    @test particles isa LagrangianParticles

    if grid isa RectilinearGrid
        simulation, jld2_filepath, nc_filepath = particle_tracking_simulation(; grid, particles, timestepper)
        model = simulation.model
        run!(simulation)

        # Just test we run without errors
        @test length(model.particles) == P
        @test propertynames(model.particles.properties) == (:x, :y, :z)

        rm(jld2_filepath)
        rm(nc_filepath)
        rm("particles_checkpoint_$(Arch)_iteration0.jld2")
    end

    #####
    ##### Test Boundary restitution
    #####

    # The particle bounces off the top of the fluid: the top of the domain, or, on the immersed
    # grid whose top cell is immersed, the bottom face of that cell.
    @allowscalar begin
        initial_z    = znode(1, 1, grid.Nz-1, grid, Center(), Center(), Center())
        top_boundary = grid isa ImmersedBoundaryGrid ? znode(1, 1, grid.Nz,   grid, Center(), Center(), Face()) :
                                                       znode(1, 1, grid.Nz+1, grid, Center(), Center(), Face())
    end

    x, y, z = on_architecture.(Ref(arch), ([0.0], [0.0], [initial_z]))
    particles = LagrangianParticles(; x, y, z, dynamics)
    u, v, w = VelocityFields(grid)

    Δt = 0.01
    interior(w, :, :, grid.Nz) .= (0.1 + top_boundary - initial_z) / Δt
    interior(w, :, :, grid.Nz - 1) .= (0.2 + top_boundary - initial_z) / Δt

    velocities = PrescribedVelocityFields(; u, v, w)
    model = HydrostaticFreeSurfaceModel(grid; particles, velocities, buoyancy=nothing, tracers=())
    time_step!(model, Δt)

    if dynamics == no_dynamics
        zᶠ = convert(array_type(arch), model.particles.properties.z)
        @test all(zᶠ .≈ (top_boundary - 0.15))
    end

    #####
    ##### Test custom particle "TestParticle"
    #####

    xs = on_architecture(arch, zeros(P))
    ys = on_architecture(arch, zeros(P))
    zs = on_architecture(arch, 0.5 * ones(P))
    us = on_architecture(arch, zeros(P))
    vs = on_architecture(arch, zeros(P))
    ws = on_architecture(arch, zeros(P))
    ss = on_architecture(arch, zeros(P))

    # Test custom constructor
    particles = StructArray{TestParticle}((xs, ys, zs, us, vs, ws, ss))

    u, v, w = velocities = VelocityFields(grid)
    speed = Field(√(u * u + v * v))
    tracked_fields = merge(velocities, (; s=speed))

    # applying v component of advection with background field to ensure it is included
    background_v = VelocityFields(grid).v
    background_v .= 1

    # Test second constructor
    lagrangian_particles = LagrangianParticles(particles; tracked_fields, dynamics)
    @test lagrangian_particles isa LagrangianParticles

    if grid isa RectilinearGrid
        model = NonhydrostaticModel(grid; timestepper,
                                      velocities, particles=lagrangian_particles,
                                      background_fields=(v=background_v,))

        set!(model, u=1)

        simulation = Simulation(model, Δt=1e-2, stop_iteration=1)

        jld2_filepath = "test_particles_$Arch.jld2"
        jld2_ow = JLD2Writer(model, (; particles=model.particles),
                             filename = jld2_filepath,
                             schedule = IterationInterval(1))

        Oceananigans.Simulations.initialize!(jld2_ow, model)
        simulation.output_writers[:particles_jld2] = jld2_ow

        nc_filepath = "test_particles_$Arch.nc"
        nc_ow = NetCDFWriter(model, (; particles = model.particles),
                             filename = nc_filepath,
                             schedule = IterationInterval(1))

        Oceananigans.Simulations.initialize!(nc_ow, model)
        simulation.output_writers[:particles_nc] = nc_ow

        checkpointer_ow = Checkpointer(model, schedule=IterationInterval(1),
                                       dir=".", prefix="particles_checkpoint_$Arch")

        simulation.output_writers[:checkpointer] = checkpointer_ow

        rm(jld2_filepath)
        rm(nc_filepath)
        rm("particles_checkpoint_$(Arch)_iteration1.jld2")
    end

    simulation, jld2_filepath, nc_filepath = particle_tracking_simulation(; grid, particles=lagrangian_particles, timestepper, velocities)
    model = simulation.model
    run!(simulation)

    @test length(model.particles) == P
    @test size(model.particles) == tuple(P)
    @test propertynames(model.particles.properties) == (:x, :y, :z, :u, :v, :w, :s)

    x = convert(array_type(arch), model.particles.properties.x)
    y = convert(array_type(arch), model.particles.properties.y)
    z = convert(array_type(arch), model.particles.properties.z)
    u = convert(array_type(arch), model.particles.properties.u)
    v = convert(array_type(arch), model.particles.properties.v)
    w = convert(array_type(arch), model.particles.properties.w)
    s = convert(array_type(arch), model.particles.properties.s)

    @test size(x) == tuple(P)
    @test size(y) == tuple(P)
    @test size(z) == tuple(P)
    @test size(u) == tuple(P)
    @test size(v) == tuple(P)
    @test size(w) == tuple(P)
    @test size(s) == tuple(P)

    if grid isa RectilinearGrid
        @test all(x .≈ 0.01)
        @test all(y .≈ 0.01)
    end
    @test all(z .≈ 0.5)
    @test all(u .≈ 1)
    @test all(v .≈ 1)
    @test all(w .≈ 0)
    @test all(s .≈ √2)

    # Test NetCDF output is correct.
    ds = NCDataset(nc_filepath)
    x, y, z = ds["x"], ds["y"], ds["z"]
    u, v, w, s = ds["u"], ds["v"], ds["w"], ds["s"]

    @test size(x) == (P, 2)
    @test size(y) == (P, 2)
    @test size(z) == (P, 2)
    @test size(u) == (P, 2)
    @test size(v) == (P, 2)
    @test size(w) == (P, 2)
    @test size(s) == (P, 2)

    if grid isa RectilinearGrid
        @test all(x[:, end] .≈ 0.01)
        @test all(y[:, end] .≈ 0.01)
    end
    @test all(z[:, end] .≈ 0.5)
    @test all(u[:, end] .≈ 1)
    @test all(v[:, end] .≈ 1)
    @test all(w[:, end] .≈ 0)
    @test all(s[:, end] .≈ √2)

    close(ds)
    rm(nc_filepath)

    # Test JLD2 output is correct
    file = jldopen(jld2_filepath)
    @test haskey(file["timeseries"], "particles")
    @test haskey(file["timeseries/particles"], "0")
    @test haskey(file["timeseries/particles"], "0")

    @test size(file["timeseries/particles/1"].x) == tuple(P)
    @test size(file["timeseries/particles/1"].y) == tuple(P)
    @test size(file["timeseries/particles/1"].z) == tuple(P)
    @test size(file["timeseries/particles/1"].u) == tuple(P)
    @test size(file["timeseries/particles/1"].v) == tuple(P)
    @test size(file["timeseries/particles/1"].w) == tuple(P)
    @test size(file["timeseries/particles/1"].s) == tuple(P)

    if grid isa RectilinearGrid
        @test all(file["timeseries/particles/1"].x .≈ 0.01)
        @test all(file["timeseries/particles/1"].y .≈ 0.01)
    end
    @test all(file["timeseries/particles/1"].z .≈ 0.5)
    @test all(file["timeseries/particles/1"].u .≈ 1)
    @test all(file["timeseries/particles/1"].v .≈ 1)
    @test all(file["timeseries/particles/1"].w .≈ 0)
    @test all(file["timeseries/particles/1"].s .≈ √2)

    close(file)
    rm(jld2_filepath)

    # Test checkpoint of particle properties
    model.particles.properties.x .= 0
    model.particles.properties.y .= 0
    model.particles.properties.z .= 0
    model.particles.properties.u .= 0
    model.particles.properties.v .= 0
    model.particles.properties.w .= 0
    model.particles.properties.s .= 0

    set!(simulation; checkpoint="particles_checkpoint_$(Arch)_iteration1.jld2")

    x = convert(array_type(arch), model.particles.properties.x)
    y = convert(array_type(arch), model.particles.properties.y)
    z = convert(array_type(arch), model.particles.properties.z)
    u = convert(array_type(arch), model.particles.properties.u)
    v = convert(array_type(arch), model.particles.properties.v)
    w = convert(array_type(arch), model.particles.properties.w)
    s = convert(array_type(arch), model.particles.properties.s)

    @test model.particles.properties isa StructArray

    @test size(x) == tuple(P)
    @test size(y) == tuple(P)
    @test size(z) == tuple(P)
    @test size(u) == tuple(P)
    @test size(v) == tuple(P)
    @test size(w) == tuple(P)
    @test size(s) == tuple(P)

    if grid isa RectilinearGrid
        @test all(x .≈ 0.01)
        @test all(y .≈ 0.01)
    end
    @test all(z .≈ 0.5)
    @test all(u .≈ 1)
    @test all(v .≈ 1)
    @test all(w .≈ 0)
    @test all(s .≈ √2)

    rm("particles_checkpoint_$(Arch)_iteration0.jld2")
    rm("particles_checkpoint_$(Arch)_iteration1.jld2")

    return nothing
end

lagrangian_particle_test_grid(arch, ::Periodic, z) =
    RectilinearGrid(arch; topology=(Periodic, Periodic, Bounded), size=(5, 5, 5), x=(-1, 1), y=(-1, 1), z)
lagrangian_particle_test_grid(arch, ::Flat, z) =
    RectilinearGrid(arch; topology=(Periodic, Flat, Bounded), size=(5, 5), x=(-1, 1), z)

lagrangian_particle_test_grid_expanded(arch, ::Periodic, z) =
    RectilinearGrid(arch; topology=(Periodic, Periodic, Bounded), size=(5, 5, 5), x=(-1, 1), y=(-1, 1), z = 2 .*z)
lagrangian_particle_test_grid_expanded(arch, ::Flat, z) =
    RectilinearGrid(arch; topology=(Periodic, Flat, Bounded), size=(5, 5), x=(-1, 1), z = 2 .*z)

function lagrangian_particle_test_immersed_grid(arch, y_topo, z)
    underlying_grid = lagrangian_particle_test_grid_expanded(arch, y_topo, z)
    z_immersed_boundary(x, z) = z < -1 || z > 1
    z_immersed_boundary(x, y, z) = z < -1 || z > 1
    GFB = GridFittedBoundary(z_immersed_boundary)
    return ImmersedBoundaryGrid(underlying_grid, GFB)
end

lagrangian_particle_test_curvilinear_grid(arch, z) =
    LatitudeLongitudeGrid(arch; size=(5, 5, 5), longitude=(-1, 1), latitude=(-1, 1), z, precompute_metrics=true)

function run_immersed_boundary_bounce_tests(arch)
    #####
    ##### Particles that are advected into an immersed cell bounce off its face
    #####

    # A solid block filling cells 2:4 in every direction, surrounded by a one-cell shell of fluid
    underlying_grid = RectilinearGrid(arch; size=(5, 5, 5), x=(0, 5), y=(0, 5), z=(0, 5),
                                      topology=(Bounded, Bounded, Bounded))
    block(x, y, z) = 1 < x < 4 && 1 < y < 4 && 1 < z < 4
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(block))

    # One particle facing each face of the block, half a cell away from it.
    # Each velocity component points towards the block and vanishes on the plane through its centre,
    # so every particle is carried 0.8 cell widths towards the block, 0.3 of them past its face.
    x₀ = [0.5, 4.5, 2.5, 2.5, 2.5, 2.5]
    y₀ = [2.5, 2.5, 0.5, 4.5, 2.5, 2.5]
    z₀ = [2.5, 2.5, 2.5, 2.5, 0.5, 4.5]

    u = (x, y, z, t) -> 0.8 * sign(2.5 - x)
    v = (x, y, z, t) -> 0.8 * sign(2.5 - y)
    w = (x, y, z, t) -> 0.8 * sign(2.5 - z)
    velocities = PrescribedVelocityFields(; u, v, w)

    for restitution in (1.0, 0.5)
        @info "  Testing Lagrangian particles bouncing off an immersed boundary [$(typeof(arch))] with restitution $restitution ..."

        x, y, z = on_architecture.(Ref(arch), (copy(x₀), copy(y₀), copy(z₀)))
        particles = LagrangianParticles(; x, y, z, restitution)
        model = HydrostaticFreeSurfaceModel(grid; particles, velocities, buoyancy=nothing, tracers=())
        time_step!(model, 1)

        x = Array(model.particles.properties.x)
        y = Array(model.particles.properties.y)
        z = Array(model.particles.properties.z)

        # Reflected off the face of the block, back into the fluid shell
        near, far = 1 - 0.3 * restitution, 4 + 0.3 * restitution
        @test x ≈ [near, far, 2.5, 2.5, 2.5, 2.5]
        @test y ≈ [2.5, 2.5, near, far, 2.5, 2.5]
        @test z ≈ [2.5, 2.5, 2.5, 2.5, near, far]
    end

    #####
    ##### Particles that cross a periodic boundary into an immersed cell bounce off the face they crossed
    #####

    @info "  Testing Lagrangian particles bouncing off an immersed boundary across a periodic boundary [$(typeof(arch))] ..."

    underlying_grid = RectilinearGrid(arch; size=(5, 5), x=(0, 5), z=(0, 5), topology=(Periodic, Flat, Bounded))

    # A solid column against the left (right) periodic boundary; one particle approaches it across the
    # periodic boundary and another approaches it from within the domain.
    left_column(x, z) = x < 1
    right_column(x, z) = x > 4

    for (column, x₀, u, expected) in ((left_column,  [4.5, 1.5], (x, z, t) -> 0.8 * sign(x - 3), [4.7, 1.3]),
                                      (right_column, [0.5, 3.5], (x, z, t) -> 0.8 * sign(x - 2), [0.3, 3.7]))

        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(column))
        velocities = PrescribedVelocityFields(; u)

        x, y, z = on_architecture.(Ref(arch), (copy(x₀), zeros(2), [2.5, 2.5]))
        particles = LagrangianParticles(; x, y, z)
        model = HydrostaticFreeSurfaceModel(grid; particles, velocities, buoyancy=nothing, tracers=())
        time_step!(model, 1)

        @test Array(model.particles.properties.x) ≈ expected
        @test Array(model.particles.properties.z) ≈ [2.5, 2.5]
    end

    return nothing
end

@testset "Lagrangian particle tracking" begin
    timesteppers = (:QuasiAdamsBashforth2, :RungeKutta3)
    y_topologies = (Periodic(), Flat())
    vertical_grids = (uniform=(-1, 1), stretched=[-1, -0.5, 0.0, 0.4, 0.7, 1])
    particle_dynamics = (no_dynamics, DroguedParticleDynamics)

    for arch in archs, timestepper in timesteppers, y_topo in y_topologies, (z_grid_type, z) in pairs(vertical_grids), dynamics in particle_dynamics
        A = typeof(arch)
        Y = typeof(y_topo)
        Z = typeof(z_grid_type)
        @info "  Testing Lagrangian particle tracking [$A, $timestepper] with y $Y on vertically $Z grid and $dynamics ..."
        if dynamics == DroguedParticleDynamics
            dynamics = dynamics(on_architecture(arch, [-1:0.1:0;]))
        end

        grid = lagrangian_particle_test_grid(arch, y_topo, z)
        run_particle_tracking_tests(grid, dynamics, timestepper)

        if z isa NTuple{2} # Test immersed regular grids
            @info "  Testing Lagrangian particle tracking [$(typeof(arch)), $timestepper] with y $(typeof(y_topo)) on vertically $z_grid_type immersed grid and $(dynamics) ..."
            grid = lagrangian_particle_test_immersed_grid(arch, y_topo, z)
            run_particle_tracking_tests(grid, dynamics, timestepper)
        end
    end

    for arch in archs, (z_grid_type, z) in pairs(vertical_grids), dynamics in particle_dynamics
        @info "  Testing Lagrangian particle tracking [$(typeof(arch))] with a LatitudeLongitudeGrid with vertically $z_grid_type z coordinate ..."
        if dynamics == DroguedParticleDynamics
            dynamics = dynamics(on_architecture(arch, [-1:0.1:0;]))
        end

        grid = lagrangian_particle_test_curvilinear_grid(arch, z)
        run_particle_tracking_tests(grid, dynamics)
    end

    for arch in archs
        run_immersed_boundary_bounce_tests(arch)
    end

    for arch in archs
        @info "  Testing Lagrangian particle tracking [$(typeof(arch))] with 0 particles ..."
        xp = Array{Float64}(undef, 0)
        yp = Array{Float64}(undef, 0)
        zp = Array{Float64}(undef, 0)

        xp = on_architecture(arch, xp)
        yp = on_architecture(arch, yp)
        zp = on_architecture(arch, zp)

        grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
        particles = LagrangianParticles(x=xp, y=yp, z=zp)
        model = NonhydrostaticModel(grid; particles)
        time_step!(model, 1)
        @test model.particles isa LagrangianParticles
    end
end
