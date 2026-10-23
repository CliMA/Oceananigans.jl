include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using NCDatasets
using StructArrays
using Oceananigans.Architectures: on_architecture
using Oceananigans.OutputWriters: fetch_output
using Oceananigans.Models.LagrangianParticleTracking: LagrangianParticleTracking, particle_species, species_index,
                                                      sort_lagrangian_particles!, cell_particle_indices,
                                                      transfer_particles!, resize_transfer_flags!,
                                                      launch_species!, SuperrangeLaunch, PerSpeciesLaunch,
                                                      advect_particle!

#####
##### Particle species with different element types
#####

struct LiquidDroplet{T, L}
         x :: T
         y :: T
         z :: T
    radius :: T
     label :: L
end

struct IceCrystal{T, L}
               x :: T
               y :: T
               z :: T
          radius :: T
    aspect_ratio :: T
           label :: L
end

struct Aerosol{T}
       x :: T
       y :: T
       z :: T
    mass :: T
end

# Freezing liquid droplets become spherical ice crystals
@inline LagrangianParticleTracking.convert_particle(::Type{<:IceCrystal}, p::LiquidDroplet) =
    (; x = p.x, y = p.y, z = p.z, radius = p.radius, aspect_ratio = one(p.radius), label = p.label)

# Copy positions, which `on_architecture` does not copy on the CPU and the particles modify in place
function liquid_droplets(arch, x, y, z; first_label=1)
    N = length(x)
    columns = (copy(x), copy(y), copy(z), collect(range(1e-5, 2e-5, length=N)), collect(first_label:first_label+N-1))
    return StructArray{LiquidDroplet}(map(column -> on_architecture(arch, column), columns))
end

function ice_crystals(arch, x, y, z; first_label=1)
    N = length(x)
    columns = (copy(x), copy(y), copy(z), collect(range(3e-5, 4e-5, length=N)), collect(range(0.5, 2, length=N)),
               collect(first_label:first_label+N-1))
    return StructArray{IceCrystal}(map(column -> on_architecture(arch, column), columns))
end

host_columns(species) = map(Array, StructArrays.components(on_architecture(CPU(), species)))
host_properties(particles) = map(host_columns, particles.properties)

function random_positions(rng, N; x=(-0.8, 0.8), z=(-0.8, 0.8))
    return (x[1] .+ (x[2] - x[1]) .* rand(rng, N), rand(rng, N), z[1] .+ (z[2] - z[1]) .* rand(rng, N))
end

#####
##### Rotation in the x-z plane, so that particles cross the top and bottom boundaries
#####

@inline species_rotation_rate(t) = 1 + sin(t) / 2
@inline u_species_rotation(x, y, z, t) = - species_rotation_rate(t) * z
@inline w_species_rotation(x, y, z, t) = + species_rotation_rate(t) * x

function species_rotation_model(arch, particles)
    grid = RectilinearGrid(arch; size=(8, 4, 8), x=(-2, 2), y=(0, 1), z=(-1, 1), topology=(Periodic, Periodic, Bounded))
    background_fields = (u = BackgroundField(u_species_rotation), w = BackgroundField(w_species_rotation))
    return NonhydrostaticModel(grid; advection=nothing, background_fields, particles)
end

function run_species_rotation(arch, properties; Nt=40, Δt=0.05, kw...)
    particles = LagrangianParticles(properties; kw...)
    model = species_rotation_model(arch, particles)
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    return model
end

by_label(columns) = map(column -> column[sortperm(columns.label)], columns)

@testset "Constructing LagrangianParticles with multiple species" begin
    liquid = liquid_droplets(CPU(), zeros(2), zeros(2), zeros(2))
    ice = ice_crystals(CPU(), zeros(3), zeros(3), zeros(3))
    particles = LagrangianParticles((; liquid, ice); sorting=CountingSort())

    @test length(particles) == 5
    @test size(particles) == (5,)
    @test particles.properties.ice === ice
    @test particle_species(particles) === (liquid, ice)
    @test species_index(particles, :ice) == 2
    @test_throws ArgumentError species_index(particles, :snow)
    @test summary(particles) == "5 LagrangianParticles in 2 species: liquid (2 LiquidDroplet), ice (3 IceCrystal)"
    @test sprint(show, particles) == """
        5 LagrangianParticles in 2 species:
        ├── liquid: 2 LiquidDroplet with 5 properties (:x, :y, :z, :radius, :label)
        ├── ice: 3 IceCrystal with 6 properties (:x, :y, :z, :radius, :aspect_ratio, :label)
        ├── boundary conditions: ParticleBoundaryConditions(ReflectingBoundary(1.0))
        ├── 0 tracked fields: ()
        ├── sorting: CellSorting(CountingSort())
        └── dynamics: no_dynamics"""

    # Workspaces mirror the species
    @test keys(particles.sorting.permutation) == (:liquid, :ice)
    @test keys(particles.transfers.destinations) == (:liquid, :ice)

    # A single species needs no transfer workspace without absorbing boundaries
    @test isnothing(LagrangianParticles(liquid).transfers)
    @test !isnothing(LagrangianParticles((; liquid)).transfers)

    # Validation
    @test_throws ArgumentError LagrangianParticles((; liquid, ice = (x=zeros(2), y=zeros(2), z=zeros(2))))
    @test_throws ArgumentError LagrangianParticles((; liquid, ice = StructArray((x=zeros(2), y=zeros(2)))))
    @test_throws ArgumentError LagrangianParticles((; liquid, ice = StructArray((x=zeros(Float32, 2), y=zeros(Float32, 2), z=zeros(Float32, 2)))))
    @test_throws ArgumentError LagrangianParticles(NamedTuple())

    grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
    c = CenterField(grid)
    @test LagrangianParticles((; liquid, ice); tracked_fields=(; aspect_ratio=c)) isa LagrangianParticles
    @test_throws ArgumentError LagrangianParticles((; liquid, ice); tracked_fields=(; mass=c))
    @test_throws ArgumentError LagrangianParticles((; liquid, ice); dynamics=DroguedParticleDynamics([0.0]))
end

@testset "Superrange and per-species launches [$(typeof(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1))
    velocities = (u = CenterField(grid), v = CenterField(grid), w = CenterField(grid))
    set!(velocities.u, (x, y, z) -> x * y)
    set!(velocities.w, (x, y, z) -> -z)

    rng = Xoshiro(7)
    x, y, z = random_positions(rng, 50; x=(0, 1), z=(0, 1))
    states = map((SuperrangeLaunch(), PerSpeciesLaunch())) do strategy
        species = (liquid_droplets(arch, x[1:20], y[1:20], z[1:20]),
                   liquid_droplets(arch, Float64[], Float64[], Float64[]),
                   ice_crystals(arch, x[21:50], y[21:50], z[21:50]))
        species_args = map(species -> (species, nothing), species)
        shared_args = (ParticleBoundaryConditions(), grid, 0.1, velocities)
        launch_species!(arch, advect_particle!, map(length, species), species_args, shared_args, strategy)
        map(host_columns, species)
    end

    @test states[1] == states[2]
    @test states[1][1].x != x[1:20]
end

@testset "A single species in a NamedTuple [$(typeof(arch))]" for arch in archs

    rng = Xoshiro(42)
    x, y, z = random_positions(rng, 300)
    boundary_conditions = ParticleBoundaryConditions(AbsorbingBoundary())

    for sorting in (nothing, CountingSort())
        plain = run_species_rotation(arch, liquid_droplets(arch, x, y, z); boundary_conditions, sorting)
        named = run_species_rotation(arch, (; liquid = liquid_droplets(arch, x, y, z)); boundary_conditions, sorting)

        # Bitwise identical
        @test 0 < length(plain.particles) < 300
        @test host_properties(named.particles).liquid == host_columns(plain.particles.properties)
    end
end

@testset "Advecting two species [$(typeof(arch))]" for arch in archs

    rng = Xoshiro(43)
    xˡ, yˡ, zˡ = random_positions(rng, 200)
    xⁱ, yⁱ, zⁱ = random_positions(rng, 100)
    boundary_conditions = ParticleBoundaryConditions(AbsorbingBoundary())

    for sorting in (nothing, CountingSort(), ComparisonSort())
        properties = (; liquid = liquid_droplets(arch, xˡ, yˡ, zˡ), ice = ice_crystals(arch, xⁱ, yⁱ, zⁱ))
        together = run_species_rotation(arch, properties; boundary_conditions, sorting)

        liquid = run_species_rotation(arch, liquid_droplets(arch, xˡ, yˡ, zˡ); boundary_conditions, sorting)
        ice = run_species_rotation(arch, ice_crystals(arch, xⁱ, yⁱ, zⁱ); boundary_conditions, sorting)

        # Advecting species together gives the same trajectories as advecting them separately
        state = host_properties(together.particles)
        @test state.liquid == host_columns(liquid.particles.properties)
        @test state.ice == host_columns(ice.particles.properties)
        @test 0 < length(state.liquid.x) < 200
        @test 0 < length(state.ice.x) < 100

        @test length(together.particles.transfers.destinations.ice) == length(state.ice.x)

        if !isnothing(sorting)
            @test issorted(Array(together.particles.sorting.cell_indices.liquid))
            @test issorted(Array(together.particles.sorting.cell_indices.ice))
        end
    end
end

#####
##### Joint sorting of species
#####

reference_cell_index(grid, x, y, z) = begin
    Nx, Ny, Nz = size(grid)
    xᶠ, yᶠ, zᶠ = map(Array, nodes(on_architecture(CPU(), grid), Face(), Face(), Face()))
    i = clamp(searchsortedlast(xᶠ, x), 1, Nx)
    j = clamp(searchsortedlast(yᶠ, y), 1, Ny)
    k = clamp(searchsortedlast(zᶠ, z), 1, Nz)
    i + Nx * (j - 1 + Ny * (k - 1))
end

function test_jointly_sorted(grid, particles, originals)
    sorting = particles.sorting
    Nx, Ny, Nz = size(grid)
    Nc = Nx * Ny * Nz
    offsets = Array(sorting.cell_offsets)
    S = length(originals)
    @test length(offsets) == S * (Nc + 1)

    for (s, name) in enumerate(keys(originals))
        state = host_columns(particles.properties[name])
        original = originals[name]
        N = length(state.x)
        cell_indices = Array(sorting.cell_indices[name])

        # Properties are permuted together, and cell indices are correct and sorted
        @test sort(state.label) == sort(original.label)
        order = indexin(state.label, original.label)
        @test state.x == original.x[order]
        @test state.z == original.z[order]
        @test cell_indices == reference_cell_index.(Ref(grid), state.x, state.y, state.z)
        @test issorted(cell_indices)

        # Offsets of species `s` delimit its particles in each cell, and particles keep their order within cells
        ranges = [cell_particle_indices(i, j, k, grid, sorting, s) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        @test sum(length, ranges) == N
        @test offsets[(s - 1) * (Nc + 1) + 1] == 0
        @test offsets[s * (Nc + 1)] == N
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            r = ranges[i, j, k]
            c = i + Nx * (j - 1 + Ny * (k - 1))
            @test all(==(c), cell_indices[r])
            @test issorted(order[r])
        end
    end

    return map(host_columns, particles.properties)
end

@testset "Sorting particle species jointly [$(typeof(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size=(3, 4, 5), x=(-1, 1), y=(0, 1), z=(-1, 1), topology=(Periodic, Periodic, Bounded))
    rng = Xoshiro(44)
    xˡ, yˡ, zˡ = random_positions(rng, 500; x=(-1, 1), z=(-1, 1))
    xⁱ, yⁱ, zⁱ = random_positions(rng, 300; x=(-1, 1), z=(-1, 1))

    results = map((CountingSort(), ComparisonSort())) do algorithm
        properties = (; liquid = liquid_droplets(arch, xˡ, yˡ, zˡ),
                        empty = ice_crystals(arch, Float64[], Float64[], Float64[]),
                        ice = ice_crystals(arch, xⁱ, yⁱ, zⁱ))
        originals = map(host_columns, properties)
        particles = LagrangianParticles(properties; sorting=algorithm)
        sort_lagrangian_particles!(particles, grid)
        sorted = test_jointly_sorted(grid, particles, originals)

        # Sorting sorted particles changes nothing
        sort_lagrangian_particles!(particles, grid)
        @test map(host_columns, particles.properties) == sorted

        # Resizing species
        resize!(properties.liquid, 100)
        resize!(properties.empty, 0)
        resize!(properties.ice, 0)
        originals = map(host_columns, properties)
        sort_lagrangian_particles!(particles, grid)
        test_jointly_sorted(grid, particles, originals)

        # No particles
        resize!(properties.liquid, 0)
        sort_lagrangian_particles!(particles, grid)
        @test all(iszero, Array(particles.sorting.cell_offsets))

        sorted
    end

    # Both algorithms produce identical results
    @test results[1] == results[2]

    # Single-species signature
    properties = liquid_droplets(arch, xˡ, yˡ, zˡ)
    particles = LagrangianParticles(properties; sorting=CountingSort())
    sort_lagrangian_particles!(particles, grid)
    @test cell_particle_indices(2, 3, 4, grid, particles.sorting) == cell_particle_indices(2, 3, 4, grid, particles.sorting, 1)
    @test host_columns(properties) == results[1].liquid
end

#####
##### Transfers between species
#####

function transfer_test_model(arch, liquid, ice; kw...)
    grid = RectilinearGrid(arch; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1))
    particles = LagrangianParticles((; liquid, ice); kw...)
    return NonhydrostaticModel(grid; particles)
end

set_flags!(flags, values) = copyto!(flags, on_architecture(architecture(flags), Int8.(values)))

# Transfer flags are resized with the particles and reset after transfers
function check_flags(particles)
    for (species, flags) in zip(particle_species(particles), particle_species(particles.transfers.destinations))
        @test length(flags) == length(species)
        @test all(iszero, Array(flags))
    end
end

@testset "Transferring particles between species [$(typeof(arch))]" for arch in archs
    rng = Xoshiro(45)
    xˡ, yˡ, zˡ = random_positions(rng, 10; x=(0, 1), z=(0, 1))
    xⁱ, yⁱ, zⁱ = random_positions(rng, 4; x=(0, 1), z=(0, 1))

    @testset "Freezing liquid droplets" begin
        liquid = liquid_droplets(arch, xˡ, yˡ, zˡ)
        ice = ice_crystals(arch, xⁱ, yⁱ, zⁱ; first_label=101)
        model = transfer_test_model(arch, liquid, ice)
        particles = model.particles
        original = host_properties(particles)

        d = species_index(particles, :ice)
        set_flags!(particles.transfers.destinations.liquid, [0, d, 0, 0, d, -1, 1, d, 0, 0])
        transfer_particles!(particles, model.grid)

        state = host_properties(particles)
        @test state.liquid.label == [1, 3, 4, 7, 9, 10]
        @test state.liquid.x == original.liquid.x[[1, 3, 4, 7, 9, 10]]
        @test state.liquid.radius == original.liquid.radius[[1, 3, 4, 7, 9, 10]]

        # Frozen droplets are appended to the ice in order
        @test state.ice.label == [101, 102, 103, 104, 2, 5, 8]
        @test state.ice.x == vcat(original.ice.x, original.liquid.x[[2, 5, 8]])
        @test state.ice.z == vcat(original.ice.z, original.liquid.z[[2, 5, 8]])
        @test state.ice.radius == vcat(original.ice.radius, original.liquid.radius[[2, 5, 8]])
        @test state.ice.aspect_ratio == vcat(original.ice.aspect_ratio, ones(3))

        check_flags(particles)

        # Nothing to transfer
        transfer_particles!(particles, model.grid)
        @test host_properties(particles) == state

        # Transfers in both directions: ice becomes liquid with the default conversion
        l = species_index(particles, :liquid)
        set_flags!(particles.transfers.destinations.liquid, [d, 0, 0, 0, 0, d])
        set_flags!(particles.transfers.destinations.ice, [0, l, -3, 0, l, 0, 0])
        transfer_particles!(particles, model.grid)
        swapped = host_properties(particles)
        @test swapped.liquid.label == [3, 4, 7, 9, 102, 2]
        @test swapped.liquid.radius == vcat(state.liquid.radius[2:5], state.ice.radius[[2, 5]])
        @test swapped.ice.label == [101, 104, 5, 8, 1, 10]
        @test swapped.ice.aspect_ratio == vcat(state.ice.aspect_ratio[[1, 4, 6, 7]], ones(2))
        check_flags(particles)

        # Freezing all droplets
        set_flags!(particles.transfers.destinations.liquid, fill(d, 6))
        transfer_particles!(particles, model.grid)
        @test length(particles.properties.liquid) == 0
        @test Array(particles.properties.ice.label) == [101, 104, 5, 8, 1, 10, 3, 4, 7, 9, 102, 2]
        check_flags(particles)

        # Transfers from an empty species
        transfer_particles!(particles, model.grid)
        @test length(particles) == 12

        # Melting into an empty species
        set_flags!(particles.transfers.destinations.ice, [l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l])
        transfer_particles!(particles, model.grid)
        @test Array(particles.properties.liquid.label) == [101, 2]
        check_flags(particles)
    end

    @testset "Conversions are validated" begin
        liquid = liquid_droplets(arch, xˡ, yˡ, zˡ)
        aerosol = StructArray{Aerosol}(map(c -> on_architecture(arch, c), (zeros(2), zeros(2), zeros(2), ones(2))))
        grid = RectilinearGrid(arch; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1))
        particles = LagrangianParticles((; liquid, aerosol))
        set_flags!(particles.transfers.destinations.liquid, [2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        @test_throws ArgumentError transfer_particles!(particles, grid)
    end

    @testset "Freezing during time steps" begin
        # Droplets that are below z = 0 before they are advected freeze after they are advected
        function freeze!(particles, model, Δt)
            liquid = particles.properties.liquid
            flags = particles.transfers.destinations.liquid
            d = species_index(particles, :ice)
            flags .= ifelse.(liquid.z .< 0, Int8(d), Int8(0))
            return nothing
        end

        rng = Xoshiro(46)
        x, y, z = random_positions(rng, 200)
        properties = (; liquid = liquid_droplets(arch, x, y, z), ice = ice_crystals(arch, Float64[], Float64[], Float64[]))
        boundary_conditions = ParticleBoundaryConditions(AbsorbingBoundary())
        model = run_species_rotation(arch, properties; boundary_conditions, dynamics=freeze!, sorting=CountingSort())

        # Transfers do not change trajectories
        reference = run_species_rotation(arch, liquid_droplets(arch, x, y, z); boundary_conditions)
        reference_state = by_label(host_columns(reference.particles.properties))
        state = host_properties(model.particles)

        labels = vcat(state.liquid.label, state.ice.label)
        @test sort(labels) == reference_state.label
        @test length(state.ice.x) > 0
        @test all(==(1), state.ice.aspect_ratio)
        survivors = indexin(state.ice.label, reference_state.label)
        @test state.ice.x ≈ reference_state.x[survivors]
        @test state.ice.z ≈ reference_state.z[survivors]
        survivors = indexin(state.liquid.label, reference_state.label)
        @test state.liquid.x ≈ reference_state.x[survivors]
        @test state.liquid.z ≈ reference_state.z[survivors]
    end
end

#####
##### Tracked fields, output, and checkpointing
#####

@testset "Tracked fields and output for multiple species [$(typeof(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1))
    aspect_ratio = CenterField(grid)
    set!(aspect_ratio, 3)
    fill_halo_regions!(aspect_ratio)

    rng = Xoshiro(47)
    liquid = liquid_droplets(arch, random_positions(rng, 5; x=(0, 1), z=(0, 1))...)
    ice = ice_crystals(arch, random_positions(rng, 3; x=(0, 1), z=(0, 1))...)
    particles = LagrangianParticles((; liquid, ice); tracked_fields=(; aspect_ratio))
    model = NonhydrostaticModel(grid; particles)
    time_step!(model, 0.01)
    @test all(≈(3), Array(ice.aspect_ratio))

    output = on_architecture(CPU(), fetch_output(particles, model))
    @test keys(output) == (:liquid, :ice)
    @test output.ice.aspect_ratio == Array(ice.aspect_ratio)
    @test output.liquid.radius == Array(liquid.radius)

    filename = "multi_species_particles_$(typeof(arch)).jld2"
    simulation = Simulation(model; Δt=0.01, stop_iteration=3)
    simulation.output_writers[:particles] = JLD2Writer(model, (; particles), filename=filename,
                                                       schedule=IterationInterval(1), overwrite_files=true, array_type=Array{Float64})
    run!(simulation)

    jldopen(filename) do file
        iterations = keys(file["timeseries/particles"])
        saved = file["timeseries/particles/3"]
        @test saved.ice.label == Array(ice.label)
        @test saved.liquid.x == Array(liquid.x)
    end
    rm(filename, force=true)

    @test_throws ArgumentError NetCDFWriter(model, (; particles), filename="multi_species_particles.nc",
                                            schedule=IterationInterval(1), overwrite_files=true)
    rm("multi_species_particles.nc", force=true)
end

@testset "Checkpointing multiple species [$(typeof(arch))]" for arch in archs

    # Freeze the droplet closest to z = 0 at iteration 3
    function freeze_central!(particles, model, Δt)
        if model.clock.iteration == 3
            flags = particles.transfers.destinations.liquid
            central = argmin(abs.(Array(particles.properties.liquid.z)))
            @allowscalar flags[central] = species_index(particles, :ice)
        end
        return nothing
    end

    function make_model()
        rng = Xoshiro(48)
        x, y, z = random_positions(rng, 12)
        properties = (; liquid = liquid_droplets(arch, x[1:8], y[1:8], z[1:8]),
                        ice = ice_crystals(arch, x[9:12], y[9:12], z[9:12]; first_label=9))
        particles = LagrangianParticles(properties; dynamics=freeze_central!, sorting=CountingSort(),
                                        boundary_conditions=ParticleBoundaryConditions(bottom=AbsorbingBoundary()))
        return species_rotation_model(arch, particles)
    end

    Δt = 0.05
    reference = make_model()
    for _ in 1:10
        time_step!(reference, Δt)
    end

    prefix = "multi_species_checkpoint_$(typeof(arch))"
    model = make_model()
    simulation = Simulation(model; Δt, stop_iteration=5, verbose=false)
    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(5), prefix=prefix)
    run!(simulation)
    @test any(≤(8), Array(model.particles.properties.ice.label))

    restored = make_model()
    simulation = Simulation(restored; Δt, stop_iteration=10, verbose=false)
    simulation.output_writers[:checkpointer] = Checkpointer(restored, schedule=IterationInterval(5), prefix=prefix)
    set!(simulation; checkpoint=:latest)
    @test host_properties(restored.particles) == host_properties(model.particles)
    run!(simulation)

    @test host_properties(restored.particles) == host_properties(reference.particles)

    rm.(filter(startswith(prefix), readdir()), force=true)
end
