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
        model = NonhydrostaticModel(; grid, timestepper, velocities, particles)
        set!(model, u=1, v=1)
    else
        set!(velocities.u, 1)
        set!(velocities.v, 1)
        model = HydrostaticFreeSurfaceModel(; grid, velocities=PrescribedVelocityFields(; velocities...), particles)
    end
    sim = Simulation(model, Δt=1e-2, stop_iteration=1)

    jld2_filepath = "test_particles_$Arch.jld2"
    sim.output_writers[:particles_jld2] = JLD2Writer(model, (; particles=model.particles),
                                                     filename=jld2_filepath, schedule=IterationInterval(1))

    nc_filepath = "test_particles_$Arch.nc"
    sim.output_writers[:particles_nc] = NetCDFWriter(model,
                                                     (; model.particles),
                                                     filename = nc_filepath,
                                                     schedule = IterationInterval(1))

    sim.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(1),
                                                     dir=".", prefix="particles_checkpoint_$Arch")

    return sim, jld2_filepath, nc_filepath
end

function run_simple_particle_tracking_tests(grid, dynamics, timestepper=:QuasiAdamsBashforth)

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
        sim, jld2_filepath, nc_filepath = particle_tracking_simulation(; grid, particles, timestepper)
        model = sim.model
        run!(sim)

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

    initial_z = @allowscalar grid.z.cᵃᵃᶜ[grid.Nz-1]
    top_boundary = @allowscalar grid.z.cᵃᵃᶠ[grid.Nz+1]

    x, y, z = on_architecture.(Ref(arch), ([0.0], [0.0], [initial_z]))

    particles = LagrangianParticles(; x, y, z, dynamics)
    u, v, w = VelocityFields(grid)

    Δt = 0.01
    interior(w, :, :, grid.Nz) .= (0.1 + top_boundary - initial_z) / Δt
    interior(w, :, :, grid.Nz - 1) .= (0.2 + top_boundary - initial_z) / Δt

    velocities = PrescribedVelocityFields(; u, v, w)

    model = HydrostaticFreeSurfaceModel(; grid, particles, velocities, buoyancy=nothing, tracers=())

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
        model = NonhydrostaticModel(; grid, timestepper,
                                      velocities, particles=lagrangian_particles,
                                      background_fields=(v=background_v,))

        set!(model, u=1)

        sim = Simulation(model, Δt=1e-2, stop_iteration=1)

        jld2_filepath = "test_particles_$Arch.jld2"
        sim.output_writers[:particles_jld2] = JLD2Writer(model, (; particles=model.particles),
                                                         filename=jld2_filepath, schedule=IterationInterval(1))

        nc_filepath = "test_particles_$Arch.nc"
        sim.output_writers[:particles_nc] = NetCDFWriter(model,
                                                         (; particles = model.particles),
                                                         filename = nc_filepath,
                                                         schedule = IterationInterval(1))

        sim.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(1),
                                                         dir=".", prefix="particles_checkpoint_$Arch")

        rm(jld2_filepath)
        rm(nc_filepath)
        rm("particles_checkpoint_$(Arch)_iteration1.jld2")
    end

    sim, jld2_filepath, nc_filepath = particle_tracking_simulation(; grid, particles=lagrangian_particles, timestepper, velocities)
    model = sim.model
    run!(sim)

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

    set!(model, "particles_checkpoint_$(Arch)_iteration1.jld2")

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
    z_immersed_boundary(x, z) = ifelse(z < -1, true, ifelse(z > 1, true, false))
    z_immersed_boundary(x, y, z) = z_immersed_boundary(x, z)
    GFB = GridFittedBoundary(z_immersed_boundary)
    return ImmersedBoundaryGrid(underlying_grid, GFB)
end

lagrangian_particle_test_curvilinear_grid(arch, z) =
    LatitudeLongitudeGrid(arch; size=(5, 5, 5), longitude=(-1, 1), latitude=(-1, 1), z, precompute_metrics=true)

@testset "Lagrangian particle tracking" begin
    timesteppers = (:QuasiAdamsBashforth2, :RungeKutta3)
    y_topologies = (Periodic(), Flat())
    vertical_grids = (uniform=(-1, 1), stretched=[-1, -0.5, 0.0, 0.4, 0.7, 1])
    particle_dynamics = (no_dynamics, DroguedParticleDynamics)

    for arch in archs, timestepper in timesteppers, y_topo in y_topologies, (z_grid_type, z) in pairs(vertical_grids), dynamics in particle_dynamics
        @info "  Testing Lagrangian particle tracking [$(typeof(arch)), $timestepper] with y $(typeof(y_topo)) on vertically $z_grid_type grid and $(dynamics) ..."
        if dynamics == DroguedParticleDynamics
            dynamics = dynamics(on_architecture(arch, [-1:0.1:0;]))
        end

        grid = lagrangian_particle_test_grid(arch, y_topo, z)
        run_simple_particle_tracking_tests(grid, dynamics, timestepper)

        if z isa NTuple{2} # Test immersed regular grids
            @info "  Testing Lagrangian particle tracking [$(typeof(arch)), $timestepper] with y $(typeof(y_topo)) on vertically $z_grid_type immersed grid and $(dynamics) ..."
            grid = lagrangian_particle_test_immersed_grid(arch, y_topo, z)
            run_simple_particle_tracking_tests(grid, dynamics, timestepper)
        end
    end

    for arch in archs, (z_grid_type, z) in pairs(vertical_grids), dynamics in particle_dynamics
        @info "  Testing Lagrangian particle tracking [$(typeof(arch))] with a LatitudeLongitudeGrid with vertically $z_grid_type z coordinate ..."
        if dynamics == DroguedParticleDynamics
            dynamics = dynamics(on_architecture(arch, [-1:0.1:0;]))
        end

        grid = lagrangian_particle_test_curvilinear_grid(arch, z)
        run_simple_particle_tracking_tests(grid, dynamics)
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
        model = NonhydrostaticModel(; grid, particles)
        time_step!(model, 1)
        @test model.particles isa LagrangianParticles
    end
end
