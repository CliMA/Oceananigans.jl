include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using StructArrays
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Models.LagrangianParticleTracking: sort_lagrangian_particles!, cell_particle_indices

function labeled_particles(arch, FT, Np; seed=1234)
    rng = Xoshiro(seed)
    values = (x = rand(rng, FT, Np),
              y = rand(rng, FT, Np),
              z = rand(rng, FT, Np),
              label = collect(1:Np))
    return StructArray(on_architecture(arch, values))
end

# Index of the cell bounded by faces `faces` containing `ξ`, or 1 for Flat dimensions
reference_index(ξ, ::Nothing, N) = 1
reference_index(ξ, faces, N) = clamp(searchsortedlast(faces, ξ), 1, N)

function face_coordinates(grid)
    xᶠ = topology(grid, 1) == Flat ? nothing : Array(xnodes(grid, Face()))
    yᶠ = topology(grid, 2) == Flat ? nothing : Array(ynodes(grid, Face()))
    zᶠ = topology(grid, 3) == Flat ? nothing : Array(znodes(grid, Face()))
    return xᶠ, yᶠ, zᶠ
end

function reference_cell_indices(grid, x, y, z)
    Nx, Ny, Nz = size(grid)
    xᶠ, yᶠ, zᶠ = face_coordinates(on_architecture(CPU(), grid))
    i = reference_index.(x, Ref(xᶠ), Nx)
    j = reference_index.(y, Ref(yᶠ), Ny)
    k = reference_index.(z, Ref(zᶠ), Nz)
    return @. i + Nx * (j - 1 + Ny * (k - 1))
end

function sorted_state(particles)
    properties = on_architecture(CPU(), particles.properties)
    sorting = particles.sorting
    return (x = Array(properties.x),
            y = Array(properties.y),
            z = Array(properties.z),
            label = Array(properties.label),
            cell_indices = Array(sorting.cell_indices),
            cell_offsets = Array(sorting.cell_offsets))
end

function test_sorted_particles(grid, particles, original)
    sorted = sorted_state(particles)
    Np = length(sorted.label)
    Nc = prod(size(grid))

    # Particle properties are permuted together
    @test sort(sorted.label) == 1:Np
    @test sorted.x == original.x[sorted.label]
    @test sorted.y == original.y[sorted.label]
    @test sorted.z == original.z[sorted.label]

    # Cell indices are correct and sorted
    @test sorted.cell_indices == reference_cell_indices(grid, sorted.x, sorted.y, sorted.z)
    @test issorted(sorted.cell_indices)

    # Cell offsets delimit the particles in each cell, and particles keep their order within cells
    offsets = sorted.cell_offsets
    @test length(offsets) == Nc + 1
    @test first(offsets) == 0
    @test last(offsets) == Np
    @test all(c -> all(==(c), sorted.cell_indices[offsets[c]+1:offsets[c+1]]), 1:Nc)
    @test all(c -> issorted(sorted.label[offsets[c]+1:offsets[c+1]]), 1:Nc)

    return sorted
end

algorithms = (CountingSort(), ComparisonSort())

@testset "Sorting Lagrangian particles [$arch, $FT]" for arch in archs, FT in float_types
    z = FT[0, 0.1, 0.15, 0.3, 0.6, 0.8, 1]
    grid = RectilinearGrid(arch, FT; size=(4, 5, 6), x=(0, 1), y=(0, 1), z,
                           topology=(Periodic, Periodic, Bounded))

    Np = 1000
    results = []

    for algorithm in algorithms
        @testset "$(summary(algorithm))" begin
            properties = labeled_particles(arch, FT, Np)
            original = map(Array, StructArrays.components(on_architecture(CPU(), properties)))
            particles = LagrangianParticles(properties; sorting=algorithm)

            @test particles.sorting.algorithm == algorithm
            @test occursin("sorting: CellSorting($(summary(algorithm)))", sprint(show, particles))

            sort_lagrangian_particles!(particles, grid)
            sorted = test_sorted_particles(grid, particles, original)
            push!(results, sorted)

            # Range of particles in a cell
            sorting = particles.sorting
            ranges = [cell_particle_indices(i, j, k, grid, sorting) for i in 1:4, j in 1:5, k in 1:6]
            @test sum(length, ranges) == Np
            @test all(r -> all(==(sorted.cell_indices[first(r)]), sorted.cell_indices[r]), filter(!isempty, ranges))

            # Sorting sorted particles changes nothing
            sort_lagrangian_particles!(particles, grid)
            @test sorted_state(particles) == sorted

            # Removing particles
            resize!(particles.properties, Np ÷ 3)
            sort_lagrangian_particles!(particles, grid)
            resorted = sorted_state(particles)
            @test resorted.label == sorted.label[1:Np ÷ 3]
            @test last(resorted.cell_offsets) == Np ÷ 3

            # No particles
            resize!(particles.properties, 0)
            sort_lagrangian_particles!(particles, grid)
            @test all(iszero, Array(particles.sorting.cell_offsets))
        end
    end

    # Both algorithms produce identical results
    @test results[1] == results[2]
end

@testset "Sorting Lagrangian particles on Flat grids [$arch]" for arch in archs
    grid = RectilinearGrid(arch; size=(8, 3), x=(0, 1), z=(0, 1), topology=(Periodic, Flat, Bounded))
    properties = labeled_particles(arch, Float64, 200)
    original = map(Array, StructArrays.components(on_architecture(CPU(), properties)))

    for algorithm in algorithms
        particles = LagrangianParticles(copy(properties); sorting=algorithm)
        sort_lagrangian_particles!(particles, grid)
        test_sorted_particles(grid, particles, original)
    end
end

@testset "Time stepping sorted Lagrangian particles [$arch]" for arch in archs
    grid = RectilinearGrid(arch; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    Np = 100
    unsorted = LagrangianParticles(labeled_particles(arch, Float64, Np))
    sorted = LagrangianParticles(labeled_particles(arch, Float64, Np); sorting=CountingSort())

    states = map((unsorted, sorted)) do particles
        model = NonhydrostaticModel(grid; particles)
        set!(model, u=0.3, v=-0.2, w=0.1)
        for _ in 1:3
            time_step!(model, 0.1)
        end
        properties = on_architecture(CPU(), particles.properties)
        order = sortperm(Array(properties.label))
        (x = Array(properties.x)[order], y = Array(properties.y)[order], z = Array(properties.z)[order])
    end

    # Sorting changes the storage order but not the particle trajectories
    @test states[1].x ≈ states[2].x
    @test states[1].y ≈ states[2].y
    @test states[1].z ≈ states[2].z

    # Particles are sorted by the cells they occupied at the beginning of the last particle step
    @test issorted(Array(sorted.sorting.cell_indices))
end
