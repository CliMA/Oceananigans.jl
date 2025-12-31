include("reactant_test_utils.jl")
include("reactant_correctness_utils.jl")

using Random

@testset "Reactant correctness" begin
    @info "Testing Reactant correctness..."

    # Use a tiny grid for fast, high-coverage tests
    Nx, Ny, Nz = 3, 4, 2
    halo = (1, 1, 1)
    extent = (1.0, 1.0, 1.0)

    # Topologies to test: all combinations of Periodic and Bounded
    topologies = vec(collect(Iterators.product(
        (Periodic, Bounded),
        (Periodic, Bounded),
        (Periodic, Bounded)
    )))

    # Locations to test: all combinations of Center and Face
    locations = vec(collect(Iterators.product(
        (Center, Face),
        (Center, Face),
        (Center, Face)
    )))

    # Get vanilla architecture from TEST_ARCHITECTURE env var (set by reactant_test_utils.jl)
    vanilla_arch = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()
    reactant_arch = ReactantState()

    @testset "fill_halo_regions! matches vanilla" begin
        @info "  Testing fill_halo_regions! equivalence across topologies and locations..."

        for topo in topologies
            TX, TY, TZ = topo
            topo_name = "$(TX)×$(TY)×$(TZ)"

            for loc in locations
                LX, LY, LZ = loc
                loc_name = "$(LX)×$(LY)×$(LZ)"
                test_name = "$topo_name @ $loc_name"

                @testset "$test_name" begin
                    # Build grids
                    grid_kw = (; size=(Nx, Ny, Nz), halo, extent, topology=(TX, TY, TZ))
                    vanilla_grid = RectilinearGrid(vanilla_arch; grid_kw...)
                    reactant_grid = RectilinearGrid(reactant_arch; grid_kw...)

                    # Build fields at the specified location
                    vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                    reactant_field = Field{LX, LY, LZ}(reactant_grid)

                    # Seed randomness deterministically
                    Random.seed!(12345)
                    init_data = randn(Float64, size(vanilla_field)...)

                    # Set fields with the same initial data
                    set!(vanilla_field, init_data)
                    set!(reactant_field, init_data)

                    # Test: interiors should match before halo fill
                    @test compare_single_field("pre", vanilla_field, reactant_field; include_halos=false)

                    # Fill halos on both
                    fill_halo_regions!(vanilla_field)
                    @jit fill_halo_regions!(reactant_field)

                    # Test: interiors should still match after halo fill
                    @test compare_single_field("post-int", vanilla_field, reactant_field; include_halos=false)

                    # Test: whole arrays (with halos) should match after halo fill
                    @test compare_single_field("post-all", vanilla_field, reactant_field; include_halos=true)
                end
            end
        end
    end
end

