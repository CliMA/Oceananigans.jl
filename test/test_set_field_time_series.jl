include("dependencies_for_runtests.jl")

using Oceananigans.Utils: Time
using Oceananigans.Fields: indices

function_of_time(x, y, z, t) = t
function_of_time(x, y, t) = t
function_of_time(x, t) = t
function_of_time(t) = t

for arch in archs
    x = y = z = (0, 1)
    zero_d_grid = RectilinearGrid(arch, size=(), topology=(Flat, Flat, Flat))
    one_d_grid = RectilinearGrid(arch, size=1; x, topology=(Periodic, Flat, Flat))
    two_d_grid = RectilinearGrid(arch, size=(1, 1); x, y,topology=(Periodic, Periodic, Flat))
    three_d_grid = RectilinearGrid(arch, size=(1, 1, 1); x, y, z, topology=(Periodic, Periodic, Bounded))

    @testset "set_to_function! with FunctionField" begin
        clock = Clock(time=1)


        for grid in (zero_d_grid, one_d_grid, two_d_grid, three_d_grid)
            c = CenterField(grid)
            Oceananigans.Fields.set_to_function!(c, function_of_time, clock)
            @test all(interior(c) .== 1)
        end
    end

    @testset "FieldTimeSeries set! with function of time" begin
        for arch in archs
            A = typeof(arch)
            @info "  Testing set! with function of time [$A]..."

            for grid in (zero_d_grid, one_d_grid, two_d_grid, three_d_grid)
                times = 0:1.0:4
                fts = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times)
                set!(fts, function_of_time)
                data = on_architecture(CPU(), view(fts, 1, 1, 1, :))
                @test data == times

                array_times = on_architecture(arch, collect(times))
                fts = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, array_times)
                set!(fts, function_of_time)
                data = on_architecture(CPU(), view(fts, 1, 1, 1, :))
                @test data == times
            end
        end
    end
end

@testset "Output writing with set!(FieldTimeSeries{OnDisk})" begin
    @info "  Testing set!(FieldTimeSeries{OnDisk})..."

    grid = RectilinearGrid(size = (1, 1, 1), extent = (1, 1, 1))
    c = CenterField(grid)

    filepath = "testfile.jld2"
    f = FieldTimeSeries(instantiated_location(c), grid, 1:10; backend=OnDisk(), path=filepath, name="c")

    for i in 1:10
        set!(c, i)
        set!(f, c, i)
    end

    g = FieldTimeSeries(filepath, "c")

    @test location(g) == (Center, Center, Center)
    @test indices(g) == (:, :, :)
    @test g.grid == grid

    @test g[1, 1, 1, 1] == 1
    @test g[1, 1, 1, 10] == 10
    @test g[1, 1, 1, Time(1.6)] == 1.6

    t = g[Time(3.8)]
    @test t[1, 1, 1] == 3.8
end

@testset "Output writing with set!(FieldDataset{OnDisk})" begin
    @info "  Testing set!(FieldDataset{OnDisk})..."

    grid = RectilinearGrid(size = (1, 1, 1), extent = (1, 1, 1))
    a = CenterField(grid)
    b = Field{Face, Center, Center}(grid)

    metadata = Dict("i"=>12, "j"=>"jay")
    filepath = "testfile.jld2"
    f = FieldDataset(1:10, (; a, b); backend=OnDisk(), path=filepath, metadata)

    for i in 1:10
        set!(a, i)
        set!(b, 2i)
        set!(f, i; a, b)
    end

    g = FieldDataset(filepath)

    @test location(g.a) == (Center, Center, Center)
    @test location(g.b) == (Face, Center, Center)
    @test g.a.grid == a.grid
    @test g.b.grid == b.grid

    @test g.a[1, 1, 1, 1] == 1
    @test g.a[1, 1, 1, 10] == 10
    @test g.a[1, 1, 1, Time(1.6)] == 1.6

    @test g.b[1, 1, 1, 1] == 2
    @test g.b[1, 1, 1, 10] == 20
    @test g.b[1, 1, 1, Time(5.1)] == 10.2

    @test g.metadata["i"] == 12
    @test g.metadata["j"] == "jay"

    t = g.a[Time(3.8)]
    @test t[1, 1, 1] == 3.8

    set!(g, 2; a=-1, b=-2)
    @test g.a[1, 1, 1, 2] == -1
    @test g.b[1, 1, 1, 2] == -2
end
