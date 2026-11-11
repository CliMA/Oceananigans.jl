include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using MPI

# Each rank fills the halos of its piece of a distributed YinYangGrid, and reduces over all ranks, exactly as the
# serial YinYangGrid with the same partition does
yin_yang_script(partition) = """
    using MPI
    MPI.Init()
    using Test
    using Oceananigans
    using Oceananigans.MultiRegion: MultiRegionPartition
    using Oceananigans.BoundaryConditions: fill_halo_regions!
    using Oceananigans.Utils: getregion
    using Oceananigans.Architectures: architecture

    partition = $partition
    arch = Distributed(CPU(); partition)
    r = arch.local_rank + 1
    serial = YinYangGrid(CPU(); size=(48, 16, 2), z=(-1, 0), partition)
    grid = YinYangGrid(arch; size=(48, 16, 2), z=(-1, 0))
    @test architecture(grid) isa Distributed

    # Only multi-region grids apply the partition to each region
    Nr = MPI.Comm_size(MPI.COMM_WORLD)
    @test_throws ArgumentError RectilinearGrid(Distributed(CPU(); partition = Partition(1)); size=(4, 4, 1), extent=(1, 1, 1))
    @test_throws ArgumentError YinYangGrid(Distributed(CPU(); partition = Partition(Nr)); size=(48, 16, 2), z=(-1, 0))

    f(λ, φ, z) = cosd(φ)^2 * sind(φ) * cosd(2λ) + cosd(φ) * sind(λ - 40) + z
    g(λ, φ, z) = sind(φ) * cosd(λ - 10) - z

    # The distributed field with the interior of the serial one, and both with their halos filled
    function filled(serial_fields, fields)
        for (s, c) in zip(serial_fields, fields)
            interior(getregion(c, 1)) .= interior(getregion(s, r))
        end
        fill_halo_regions!(length(fields) == 1 ? only(serial_fields) : serial_fields)
        fill_halo_regions!(length(fields) == 1 ? only(fields) : fields)
        return all(parent(getregion(c, 1)) == parent(getregion(s, r)) for (s, c) in zip(serial_fields, fields))
    end

    for LX in (Center, Face), LY in (Center, Face)
        cs = Field{LX, LY, Center}(serial)
        set!(cs, f)
        @test filled((cs,), (Field{LX, LY, Center}(grid),))
    end

    us, vs = XFaceField(serial), YFaceField(serial)
    set!(us, f)
    set!(vs, g)
    @test filled((us, vs), (XFaceField(grid), YFaceField(grid)))

    cs, c = CenterField(serial), CenterField(grid)
    set!(cs, (λ, φ, z) -> 2 + f(λ, φ, z))
    interior(getregion(c, 1)) .= interior(getregion(cs, r))
    @test sum(c) ≈ sum(cs)
    @test maximum(c) == maximum(cs)
    @test minimum(c) == minimum(cs)
    @test Field(Integral(c))[1, 1, 1] ≈ Field(Integral(cs))[1, 1, 1]
"""

function run_mpi_script(script, nranks)
    mktempdir() do dir
        path = joinpath(dir, "script.jl")
        write(path, script)
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 $path`)
    end
    return nothing
end

@testset "Distributed YinYangGrid with $partition on $nranks ranks" for (nranks, partition) in
        ((2, "Partition(1)"), (3, "MultiRegionPartition(Partition(1), Partition(2))"),
         (4, "Partition(2)"), (8, "Partition(2, 2)"))
    @test (run_mpi_script(yin_yang_script(partition), nranks); true)
end
