include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using StructArrays
using Oceananigans.Architectures: on_architecture

function labeled_particles(arch, FT, x, y, z)
    values = (x = FT.(x), y = FT.(y), z = FT.(z), label = collect(1:length(x)))
    return StructArray(on_architecture(arch, values))
end

particle_state(particles) = map(Array, StructArrays.components(on_architecture(CPU(), particles.properties)))

# Advect particles for one time step of Δt = 1 with prescribed velocities
function advect_once(grid, particles; u=0, v=0, w=0)
    velocities = PrescribedVelocityFields(u = (x, y, z, t) -> u, v = (x, y, z, t) -> v, w = (x, y, z, t) -> w)
    model = HydrostaticFreeSurfaceModel(grid; particles, velocities, buoyancy=nothing, tracers=())
    time_step!(model, 1)
    return particle_state(particles)
end

@testset "ParticleBoundaryConditions" begin
    default = ParticleBoundaryConditions()
    @test all(side -> getproperty(default, side) == ReflectingBoundary(1.0),
              (:west, :east, :south, :north, :bottom, :top, :immersed))
    @test summary(default) == "ParticleBoundaryConditions(ReflectingBoundary(1.0))"

    bcs = ParticleBoundaryConditions(ReflectingBoundary(0.5); bottom=AbsorbingBoundary(), immersed=AbsorbingBoundary())
    @test bcs.west == ReflectingBoundary(0.5)
    @test bcs.bottom == AbsorbingBoundary()
    @test summary(bcs) == "ParticleBoundaryConditions(ReflectingBoundary(0.5); bottom=AbsorbingBoundary(), immersed=AbsorbingBoundary())"
    @test occursin("├── bottom: AbsorbingBoundary()", sprint(show, bcs))

    particles = LagrangianParticles(x=[1.0], y=[1.0], z=[1.0], boundary_conditions=bcs)
    @test occursin("boundary conditions: $(summary(bcs))", sprint(show, particles))
    @test isnothing(LagrangianParticles(x=[1.0], y=[1.0], z=[1.0]).transfers)
    @test !isnothing(particles.transfers)

    # Drogued particles store depths outside particle properties, which reordering would misalign
    dynamics = DroguedParticleDynamics([0.0])
    @test_throws ArgumentError LagrangianParticles(x=[1.0], y=[1.0], z=[1.0]; dynamics, sorting=CountingSort())
    @test_throws ArgumentError LagrangianParticles(x=[1.0], y=[1.0], z=[1.0]; dynamics, boundary_conditions=bcs)
end

@testset "Lagrangian particle boundary conditions [$arch, $FT]" for arch in archs, FT in float_types
    grid = RectilinearGrid(arch, FT; size=(5, 5, 5), x=(0, 5), y=(0, 5), z=(0, 5),
                           topology=(Bounded, Periodic, Bounded))

    @testset "Reflecting boundaries with restitution $restitution" for restitution in (1, 0.5)
        boundary_conditions = ParticleBoundaryConditions(ReflectingBoundary(restitution))
        properties = labeled_particles(arch, FT, [0.5, 4.5], [2.5, 2.5], [0.5, 4.5])
        particles = LagrangianParticles(properties; boundary_conditions)
        state = advect_once(grid, particles; u=0.8, w=-0.8)

        # The first particle crosses the bottom and the second particle crosses the east boundary by 0.3
        @test state.label == [1, 2]
        @test state.x ≈ FT[1.3, 5 - 0.3restitution]
        @test state.z ≈ FT[0.3restitution, 3.7]
    end

    @testset "Absorbing bottom with sorting = $sorting" for sorting in (nothing, CountingSort())
        boundary_conditions = ParticleBoundaryConditions(bottom=AbsorbingBoundary())
        properties = labeled_particles(arch, FT, [4.5, 0.5, 2.5, 1.5], [2.5, 2.5, 2.5, 2.5], [2.5, 0.5, 4.5, 0.9])
        particles = LagrangianParticles(properties; boundary_conditions, sorting)
        state = advect_once(grid, particles; w=-1)

        # Particles 2 and 4 cross the bottom and are removed; the others keep their order
        @test length(particles) == 2
        if isnothing(sorting)
            @test state.label == [1, 3]
        else
            @test sort(state.label) == [1, 3]
        end
        @test state.z[sortperm(state.label)] ≈ FT[1.5, 3.5]

        # Removing all particles
        state = advect_once(grid, particles; w=-5)
        @test length(particles) == 0

        # Advecting no particles
        state = advect_once(grid, particles; w=-5)
        @test length(particles) == 0
    end

    @testset "Absorbing west, reflecting east" begin
        boundary_conditions = ParticleBoundaryConditions(west=AbsorbingBoundary())
        properties = labeled_particles(arch, FT, [0.5, 4.5], [2.5, 2.5], [2.5, 2.5])
        particles = LagrangianParticles(properties; boundary_conditions)
        state = advect_once(grid, particles; u=-0.8)
        @test state.label == [2]
        @test state.x ≈ FT[3.7]

        properties = labeled_particles(arch, FT, [0.5, 4.5], [2.5, 2.5], [2.5, 2.5])
        particles = LagrangianParticles(properties; boundary_conditions)
        state = advect_once(grid, particles; u=0.8)
        @test state.label == [1, 2]
        @test state.x ≈ FT[1.3, 4.7]
    end

    @testset "Many particles absorbed at the bottom" begin
        Np = 10000
        rng = Xoshiro(1)
        x, y, z = 5rand(rng, Np), 5rand(rng, Np), 5rand(rng, Np)
        properties = labeled_particles(arch, FT, x, y, z)
        particles = LagrangianParticles(properties; boundary_conditions=ParticleBoundaryConditions(bottom=AbsorbingBoundary()))
        state = advect_once(grid, particles; w=-1)

        expected = findall(z -> z - 1 ≥ 0, FT.(z))
        @test state.label == expected
        @test state.z ≈ FT.(z[expected]) .- 1
    end
end

@testset "Absorbing immersed boundaries [$arch]" for arch in archs
    underlying_grid = RectilinearGrid(arch; size=(5, 5, 5), x=(0, 5), y=(0, 5), z=(0, 5),
                                      topology=(Bounded, Bounded, Bounded))
    block(x, y, z) = (1 < x < 4) & (1 < y < 4) & (1 < z < 4)
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(block))

    boundary_conditions = ParticleBoundaryConditions(immersed=AbsorbingBoundary())
    properties = labeled_particles(arch, Float64, [0.5, 0.5], [2.5, 0.5], [2.5, 2.5])
    particles = LagrangianParticles(properties; boundary_conditions)

    # The first particle enters the block and is removed; the second particle passes beside it
    state = advect_once(grid, particles; u=0.8)
    @test state.label == [2]
    @test state.x ≈ [1.3]
end
