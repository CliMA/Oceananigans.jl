include("dependencies_for_runtests.jl")

using Oceananigans.OutputReaders: Cyclical, Linear, Clamp, Prefetched, InMemory,
                                  new_backend, time_indices, update_field_time_series!
using Oceananigans.Units: Time

@assert Threads.nthreads() >= 2 "Prefetched FieldTimeSeries tests require JULIA_NUM_THREADS ≥ 2 (got $(Threads.nthreads())). Re-run with `julia -t 2`."

function build_test_jld2(; Nx=8, Nt=12)
    filepath = tempname() * ".jld2"
    grid = RectilinearGrid(CPU(), size=Nx, x=(0, 1), topology=(Periodic, Flat, Flat))
    times = collect(range(0, 1; length=Nt))
    fts = FieldTimeSeries{Center, Nothing, Nothing}(grid, times; backend=OnDisk(), path=filepath, name="f")
    for n in 1:Nt
        c = CenterField(grid)
        set!(c, x -> n * x)
        set!(fts, c, n)
    end
    return filepath, Nt
end

@testset "Prefetched FieldTimeSeries" begin
    filepath, Nt = build_test_jld2()
    Nm = 4

    @testset "construction handoff: prefetch=true wraps with Prefetched on multi-thread" begin
        fts = FieldTimeSeries(filepath, "f"; backend=InMemory(Nm; prefetch=true), time_indexing=Cyclical())
        @test fts.backend isa Prefetched
        @test fts.backend.base_backend isa InMemory{Int, true}  # flag preserved through new_backend
        @test fts.backend.start == 1
        @test fts.backend.length == Nm                          # forwarded via getproperty
        @test length(fts.backend) == Nm
    end

    @testset "byte-identity vs non-prefetching reference (Cyclical, four reloads)" begin
        filepath_ref, _ = build_test_jld2()
        filepath_pf,  _ = build_test_jld2()
        ref_fts = FieldTimeSeries(filepath_ref, "f"; backend=InMemory(Nm),                time_indexing=Cyclical())
        pf_fts  = FieldTimeSeries(filepath_pf,  "f"; backend=InMemory(Nm; prefetch=true), time_indexing=Cyclical())

        @test parent(pf_fts.data) == parent(ref_fts.data)       # initial-window load alignment

        for needed in (4, 7, 10, 1)
            update_field_time_series!(ref_fts, needed, needed + 1)
            update_field_time_series!(pf_fts,  needed, needed + 1)
            @test parent(pf_fts.data) == parent(ref_fts.data)
            @test pf_fts.backend.next_start == mod1(needed + Nm - 1, Nt)
        end

        rm(filepath_ref; force=true)
        rm(filepath_pf;  force=true)
    end

    @testset "tamper guard: getproperty(:buffer_fts) warns" begin
        fts = FieldTimeSeries(filepath, "f"; backend=InMemory(Nm; prefetch=true), time_indexing=Cyclical())
        @test_logs (:warn, r"buffer_fts") fts.backend.buffer_fts
    end

    @testset "Adapt strips the wrapper" begin
        fts = FieldTimeSeries(filepath, "f"; backend=InMemory(Nm; prefetch=true), time_indexing=Cyclical())
        adapted = Adapt.adapt(Array, fts.backend)
        @test adapted isa InMemory                              # not a Prefetched
        @test !(adapted isa Prefetched)
    end

    rm(filepath; force=true)
end
