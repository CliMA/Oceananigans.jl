include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

using MPI
using Oceananigans.DistributedComputations: global_index_offset

# ============================================================================
# Helper: grid variable names tested in grid reconstruction
# ============================================================================

grid_var_names = [:Δxᶠᶠᵃ, :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ,
                  :Δyᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ,
                  :Azᶠᶠᵃ, :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ]

# ============================================================================
# Testset 1: Grid Reconstruction
# UPivot + FPivot × slab + pencil = 4 configs, all in one 4-rank mpiexec
# ============================================================================

grid_recon_configs = [
    # (fold_name, fold_kw, Nx, Ny, halo)
    ("UPivot", "",                                12, 20, (2, 2, 2)),
    ("FPivot", ", fold_topology = RightFaceFolded", 12, 21, (2, 2, 2)),
]

partition_configs = [
    # (partition_str, nranks, pname)
    ("Partition(1, 4)", 4, "1x4"),
    ("Partition(2, 2)", 4, "2x2"),
]

grid_recon_script = let
    # Build a script that tests all 4 configs in one MPI session
    configs_str = repr([
        (fn, fk, Nx, Ny, halo, ps, nr, pn)
        for (fn, fk, Nx, Ny, halo) in grid_recon_configs
        for (ps, nr, pn) in partition_configs
    ])

    """
    using MPI
    MPI.Init()
    using JLD2
    include("distributed_tests_utils.jl")
    using Oceananigans.DistributedComputations: global_index_offset

    configs = $configs_str

    for (fold_name, fold_kw, Nx, Ny, halo, partition_str, nranks, pname) in configs
        partition = eval(Meta.parse(partition_str))
        arch = Distributed(CPU(); partition)
        fold_topology = isempty(fold_kw) ? RightCenterFolded : RightFaceFolded

        local_grid = TripolarGrid(arch; size = (Nx, Ny, 1), z = (-1000, 0), halo, fold_topology)
        recon = reconstruct_global_grid(local_grid)

        tag = "grid_recon_\$(fold_name)_\$(pname)"
        nx, ny, _ = size(local_grid)
        rx, ry, _ = arch.local_index .- 1
        Rx, Ry, _ = arch.ranks

        # Save per-rank local grid data
        filename = "\$(tag)_\$(arch.local_rank).jld2"
        jldopen(filename, "w") do file
            for vname in $(repr(grid_var_names))
                file["local_\$(vname)"] = getproperty(local_grid, vname)
            end
            file["nx"] = nx; file["ny"] = ny
            file["rx"] = rx; file["ry"] = ry
            file["Rx"] = Rx; file["Ry"] = Ry
        end

        # Save reconstructed global grid (rank 0 only)
        if arch.local_rank == 0
            jldopen("\$(tag)_global.jld2", "w") do file
                for vname in $(repr(grid_var_names))
                    file["recon_\$(vname)"] = getproperty(recon, vname)
                end
            end
        end

        MPI.Barrier(MPI.COMM_WORLD)
    end

    MPI.Finalize()
    """
end

@testset "Test distributed TripolarGrid grid reconstruction" begin
    scriptfile = "grid_recon_script.jl"
    write(scriptfile, grid_recon_script)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
    rm(scriptfile)

    for (fold_name, fold_kw, Nx, Ny, halo) in grid_recon_configs
        fold_topology = isempty(fold_kw) ? RightCenterFolded : RightFaceFolded
        serial_grid = TripolarGrid(; size = (Nx, Ny, 1), z = (-1000, 0), halo, fold_topology)

        for (_, nranks, pname) in partition_configs
            tag = "grid_recon_$(fold_name)_$(pname)"

            @testset "$fold_name $pname global" begin
                global_data = jldopen("$(tag)_global.jld2")
                for vname in grid_var_names
                    @test global_data["recon_$(vname)"] ≈ getproperty(serial_grid, vname)
                end
                close(global_data)
                rm("$(tag)_global.jld2")
            end

            @testset "$fold_name $pname rank $r" for r in 0:(nranks - 1)
                rank_data = jldopen("$(tag)_$(r).jld2")
                nx = rank_data["nx"]; ny = rank_data["ny"]
                rx = rank_data["rx"]; ry = rank_data["ry"]
                Rx = rank_data["Rx"]; Ry = rank_data["Ry"]

                x_offset = global_index_offset(Nx, Rx, rx + 1)
                y_offset = global_index_offset(Ny, Ry, ry + 1)
                irange = (1 + x_offset):(x_offset + nx)
                jrange = (1 + y_offset):(y_offset + ny)

                for vname in grid_var_names
                    local_arr = rank_data["local_$(vname)"]
                    serial_arr = getproperty(serial_grid, vname)
                    @test local_arr[1:nx, 1:ny] ≈ serial_arr[irange, jrange]
                end
                close(rank_data)
                rm("$(tag)_$(r).jld2")
            end
        end
    end
end

# ============================================================================
# Testset 2: Field Reconstruction
# UPivot + FPivot × slab + pencil = 4 configs, all in one 4-rank mpiexec
# ============================================================================

field_recon_configs = [
    # (fold_name, fold_kw, Nx, Ny, halo)
    ("UPivot", "",                                40, 40, (5, 5, 5)),
    ("FPivot", ", fold_topology = RightFaceFolded", 40, 41, (5, 5, 5)),
]

field_recon_script = let
    configs_str = repr([
        (fn, fk, Nx, Ny, halo, ps, nr, pn)
        for (fn, fk, Nx, Ny, halo) in field_recon_configs
        for (ps, nr, pn) in partition_configs
    ])

    """
    using MPI
    MPI.Init()
    using JLD2
    include("distributed_tests_utils.jl")
    using Oceananigans.DistributedComputations: global_index_offset

    configs = $configs_str

    for (fold_name, fold_kw, Nx, Ny, halo, partition_str, nranks, pname) in configs
        partition = eval(Meta.parse(partition_str))
        arch = Distributed(CPU(); partition)
        fold_topology = isempty(fold_kw) ? RightCenterFolded : RightFaceFolded

        local_grid = TripolarGrid(arch; size = (Nx, Ny, 1), z = (-1000, 0), halo, fold_topology)

        # For FPivot, center-y interior is Ny-1; for UPivot it's Ny
        center_Ny = fold_topology == RightFaceFolded ? Ny - 1 : Ny
        u_init = [Float64(i + 10 * j) for i in 1:Nx, j in 1:center_Ny]
        v_init = [Float64(i + 10 * j) for i in 1:Nx, j in 1:Ny]
        c_init = [Float64(i + 10 * j) for i in 1:Nx, j in 1:center_Ny]

        up = XFaceField(local_grid); set!(up, u_init)
        vp = YFaceField(local_grid); set!(vp, v_init)
        cp = CenterField(local_grid); set!(cp, c_init)
        fill_halo_regions!((up, vp, cp))

        tag = "field_recon_\$(fold_name)_\$(pname)"
        nx, ny, _ = size(local_grid)
        rx, ry, _ = arch.local_index .- 1
        Rx, Ry, _ = arch.ranks

        # Save per-rank local field data
        filename = "\$(tag)_\$(arch.local_rank).jld2"
        jldopen(filename, "w") do file
            file["local_u"] = up.data
            file["local_v"] = vp.data
            file["local_c"] = cp.data
            file["nx"] = nx; file["ny"] = ny
            file["rx"] = rx; file["ry"] = ry
            file["Rx"] = Rx; file["Ry"] = Ry
        end

        # Save reconstructed global fields (rank 0 only)
        gu = reconstruct_global_field(up)
        gv = reconstruct_global_field(vp)
        gc = reconstruct_global_field(cp)
        if arch.local_rank == 0
            jldopen("\$(tag)_global.jld2", "w") do file
                file["recon_u"] = Array(interior(gu))
                file["recon_v"] = Array(interior(gv))
                file["recon_c"] = Array(interior(gc))
            end
        end

        MPI.Barrier(MPI.COMM_WORLD)
    end

    MPI.Finalize()
    """
end

@testset "Test distributed TripolarGrid field reconstruction" begin
    scriptfile = "field_recon_script.jl"
    write(scriptfile, field_recon_script)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
    rm(scriptfile)

    for (fold_name, fold_kw, Nx, Ny, halo) in field_recon_configs
        fold_topology = isempty(fold_kw) ? RightCenterFolded : RightFaceFolded
        serial_grid = TripolarGrid(; size = (Nx, Ny, 1), z = (-1000, 0), halo, fold_topology)

        center_Ny = fold_topology == RightFaceFolded ? Ny - 1 : Ny
        u_init = [Float64(i + 10 * j) for i in 1:Nx, j in 1:center_Ny]
        v_init = [Float64(i + 10 * j) for i in 1:Nx, j in 1:Ny]
        c_init = [Float64(i + 10 * j) for i in 1:Nx, j in 1:center_Ny]

        us = XFaceField(serial_grid);  set!(us, u_init)
        vs = YFaceField(serial_grid);  set!(vs, v_init)
        cs = CenterField(serial_grid); set!(cs, c_init)
        fill_halo_regions!((us, vs, cs))

        for (_, nranks, pname) in partition_configs
            tag = "field_recon_$(fold_name)_$(pname)"

            @testset "$fold_name $pname reconstructed global" begin
                global_data = jldopen("$(tag)_global.jld2")
                @test global_data["recon_u"] ≈ interior(us)
                @test global_data["recon_v"] ≈ interior(vs)
                @test global_data["recon_c"] ≈ interior(cs)
                close(global_data)
                rm("$(tag)_global.jld2")
            end

            @testset "$fold_name $pname rank $r" for r in 0:(nranks - 1)
                rank_data = jldopen("$(tag)_$(r).jld2")
                rx = rank_data["rx"]; ry = rank_data["ry"]
                Rx = rank_data["Rx"]; Ry = rank_data["Ry"]

                x_offset = global_index_offset(Nx, Rx, rx + 1)
                y_offset = global_index_offset(Ny, Ry, ry + 1)
                irange = axes(rank_data["local_u"], 1) .+ x_offset
                jrange = axes(rank_data["local_u"], 2) .+ y_offset

                for (fname, serial_field) in [("u", us), ("v", vs), ("c", cs)]
                    local_data = rank_data["local_$(fname)"]
                    serial_data = serial_field.data
                    @test local_data[:, :, 1] ≈ serial_data[irange, jrange, 1]
                end
                close(rank_data)
                rm("$(tag)_$(r).jld2")
            end
        end
    end
end

# ============================================================================
# Testset 3: UPivot Boundary Conditions (Partition(2,2), 4 ranks)
# ============================================================================

tripolar_boundary_conditions = """
    using MPI
    MPI.Init()
    using JLD2
    include("distributed_tests_utils.jl")

    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = TripolarGrid(arch; size = (20, 20, 1), z = (-1000, 0))

    serial_grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0))

    vs = YFaceField(serial_grid)
    cs = CenterField(serial_grid)
    I1 = [i + j * 100 for i in 1:20, j in 1:20]
    set!(vs, I1); set!(cs, I1)
    fill_halo_regions!((vs, cs))

    v = YFaceField(grid)
    c = CenterField(grid)
    set!(v, vs); set!(c, cs)
    fill_halo_regions!((v, c))

    filename = "distributed_tripolar_boundary_conditions_" * string(arch.local_rank) * ".jld2"
    jldopen(filename, "w") do file
        file["v"] = v.data
        file["c"] = c.data
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
"""

@testset "Test distributed TripolarGrid boundary conditions..." begin
    grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0))
    I1 = [i + j * 100 for i in 1:20, j in 1:20]
    v = YFaceField(grid); c = CenterField(grid)
    set!(v, I1); set!(c, I1)
    fill_halo_regions!((v, c))

    write("distributed_boundary_tests.jl", tripolar_boundary_conditions)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 --check-bounds=yes distributed_boundary_tests.jl`)
    rm("distributed_boundary_tests.jl")

    vp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["v"]
    cp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["c"]
    vp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["v"]
    cp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["c"]

    @test v.data[-3:14, end-3:end-1, 1] ≈ vp1.parent[:, end-3:end-1, 5]
    @test c.data[-3:14, end-3:end-1, 1] ≈ cp1.parent[:, end-3:end-1, 5]
    @test v.data[7:end, 7:end-1, 1] ≈ vp3.parent[:, 1:end-1, 5]
    @test c.data[7:end, 7:end-1, 1] ≈ cp3.parent[:, 1:end-1, 5]

    for r in 0:3
        rm("distributed_tripolar_boundary_conditions_$(r).jld2", force=true)
    end
end

# ============================================================================
# Testset 4: FPivot Boundary Conditions (Partition(2,2), 4 ranks)
# ============================================================================

fpivot_boundary_conditions = """
    using MPI
    MPI.Init()
    using JLD2
    include("distributed_tests_utils.jl")

    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = TripolarGrid(arch; size = (20, 21, 1), z = (-1000, 0), fold_topology = RightFaceFolded)

    serial_grid = TripolarGrid(size = (20, 21, 1), z = (-1000, 0), fold_topology = RightFaceFolded)

    vs = YFaceField(serial_grid)
    cs = CenterField(serial_grid)
    I_v = [i + j * 100 for i in 1:20, j in 1:21]
    I_c = [i + j * 100 for i in 1:20, j in 1:20]
    set!(vs, I_v); set!(cs, I_c)
    fill_halo_regions!((vs, cs))

    v = YFaceField(grid)
    c = CenterField(grid)
    set!(v, vs); set!(c, cs)
    fill_halo_regions!((v, c))

    filename = "distributed_fpivot_boundary_conditions_" * string(arch.local_rank) * ".jld2"
    jldopen(filename, "w") do file
        file["v"] = v.data
        file["c"] = c.data
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
"""

@testset "Test distributed FPivot TripolarGrid boundary conditions..." begin
    grid = TripolarGrid(size = (20, 21, 1), z = (-1000, 0), fold_topology = RightFaceFolded)
    I_v = [i + j * 100 for i in 1:20, j in 1:21]
    I_c = [i + j * 100 for i in 1:20, j in 1:20]
    v = YFaceField(grid); c = CenterField(grid)
    set!(v, I_v); set!(c, I_c)
    fill_halo_regions!((v, c))

    write("distributed_fpivot_boundary_tests.jl", fpivot_boundary_conditions)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 --check-bounds=yes distributed_fpivot_boundary_tests.jl`)
    rm("distributed_fpivot_boundary_tests.jl")

    vp1 = jldopen("distributed_fpivot_boundary_conditions_1.jld2")["v"]
    cp1 = jldopen("distributed_fpivot_boundary_conditions_1.jld2")["c"]
    vp3 = jldopen("distributed_fpivot_boundary_conditions_3.jld2")["v"]
    cp3 = jldopen("distributed_fpivot_boundary_conditions_3.jld2")["c"]

    @test v.data[-3:14, end-3:end-1, 1] ≈ vp1.parent[:, end-3:end-1, 5]
    @test c.data[-3:14, end-3:end-1, 1] ≈ cp1.parent[:, end-3:end-1, 5]
    @test v.data[7:end, 7:end-1, 1] ≈ vp3.parent[:, 1:end-1, 5]
    @test c.data[7:end, 7:end-1, 1] ≈ cp3.parent[:, 1:end-1, 5]

    for r in 0:3
        rm("distributed_fpivot_boundary_conditions_$(r).jld2", force=true)
    end
end

# ============================================================================
# Testset 5: Index Tracing Halo Tests (4 ranks)
# Tests that distributed halo fills map indices correctly for all stagger
# locations (CC, FC, CF, FF). Each field is filled with its global index
# value, halos are filled, then compared with serial.
# ============================================================================

index_tracing_4rank_script = """
    using MPI
    MPI.Init()
    using JLD2
    include("distributed_tests_utils.jl")
    using Oceananigans.Grids: halo_size
    using Oceananigans.DistributedComputations: global_index_offset

    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    function fill_index_field!(field, dim; offset=0)
        int = interior(field)
        for I in CartesianIndices(int)
            int[I] = Float64(I.I[dim] + offset)
        end
        return nothing
    end

    test_cases = [
        (RightCenterFolded, 40, 40),
        (RightCenterFolded, 40, 43),
        (RightFaceFolded,   40, 41),
        (RightFaceFolded,   40, 42),
    ]

    partitions = [Partition(1, 4), Partition(2, 2)]
    locations = [
        ("CC", (Center(), Center(), Center())),
        ("FC", (Face(),   Center(), Center())),
        ("CF", (Center(), Face(),   Center())),
        ("FF", (Face(),   Face(),   Center())),
    ]
    Hx, Hy, Hz = 5, 5, 5

    results = Dict{String, Bool}()

    for (fold_topology, global_Nx, global_Ny) in test_cases
        fold_name = fold_topology == RightCenterFolded ? "UPivot" : "FPivot"

        for partition in partitions
            arch = Distributed(CPU(); partition)
            grid = TripolarGrid(arch; size=(global_Nx, global_Ny, 1), z=(-1000, 0),
                                halo=(Hx, Hy, Hz), fold_topology)

            x_offset, y_offset, _ = global_index_offset(arch, (global_Nx, global_Ny, 1))

            # Serial grid and fields (rank 0 only)
            serial_fields = Dict{String, Any}()
            if rank == 0
                serial_grid = TripolarGrid(; size=(global_Nx, global_Ny, 1), z=(-1000, 0),
                                           halo=(Hx, Hy, Hz), fold_topology)
                for (loc_name, loc) in locations
                    fi = Field(loc, serial_grid)
                    fj = Field(loc, serial_grid)
                    fill_index_field!(fi, 1)
                    fill_index_field!(fj, 2)
                    fill_halo_regions!((fi, fj))
                    serial_fields["\$(loc_name)_i"] = fi
                    serial_fields["\$(loc_name)_j"] = fj
                end
            end
            MPI.Barrier(MPI.COMM_WORLD)

            config_pass = true
            for (loc_name, loc) in locations
                di = Field(loc, grid)
                dj = Field(loc, grid)
                fill_index_field!(di, 1; offset=x_offset)
                fill_index_field!(dj, 2; offset=y_offset)
                fill_halo_regions!((di, dj))

                gi = reconstruct_global_field(di)
                gj = reconstruct_global_field(dj)

                # Interior check (rank 0)
                if rank == 0
                    si = serial_fields["\$(loc_name)_i"]
                    sj = serial_fields["\$(loc_name)_j"]
                    if interior(gi, :, :, 1) != interior(si, :, :, 1) ||
                       interior(gj, :, :, 1) != interior(sj, :, :, 1)
                        config_pass = false
                    end
                end

                # Local halo check: broadcast serial parent data to all ranks
                if rank == 0
                    si = serial_fields["\$(loc_name)_i"]
                    sj = serial_fields["\$(loc_name)_j"]
                    s_full_i = parent(si.data)[:, :, 1+Hz]
                    s_full_j = parent(sj.data)[:, :, 1+Hz]
                else
                    s_full_i = nothing
                    s_full_j = nothing
                end
                s_full_i = MPI.bcast(s_full_i, 0, MPI.COMM_WORLD)
                s_full_j = MPI.bcast(s_full_j, 0, MPI.COMM_WORLD)

                local_i = di.data[:, :, 1]
                local_j = dj.data[:, :, 1]
                local_mis = 0
                for lj in axes(local_i, 2), li in axes(local_i, 1)
                    spi = li + x_offset + Hx
                    spj = lj + y_offset + Hy
                    if spi in axes(s_full_i, 1) && spj in axes(s_full_i, 2)
                        if Int(local_i[li, lj]) != Int(s_full_i[spi, spj]) ||
                           Int(local_j[li, lj]) != Int(s_full_j[spi, spj])
                            local_mis += 1
                        end
                    end
                end

                all_mis = MPI.Gather(local_mis, 0, MPI.COMM_WORLD)
                if rank == 0 && sum(all_mis) > 0
                    config_pass = false
                end
            end

            pname = partition == Partition(1, 4) ? "1x4" : "2x2"
            key = "\$(fold_name)_Nx\$(global_Nx)_Ny\$(global_Ny)_\$(pname)"
            if rank == 0
                results[key] = config_pass
            end
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end

    # Save results (rank 0 only)
    if rank == 0
        jldopen("index_tracing_4rank_results.jld2", "w") do file
            for (key, pass) in results
                file[key] = pass
            end
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
"""

@testset "Test distributed TripolarGrid index tracing (4 ranks)..." begin
    write("index_tracing_4rank.jl", index_tracing_4rank_script)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 --check-bounds=yes index_tracing_4rank.jl`)
    rm("index_tracing_4rank.jl")

    results = jldopen("index_tracing_4rank_results.jld2")
    for key in keys(results)
        @testset "$key" begin
            @test results[key] == true
        end
    end
    close(results)
    rm("index_tracing_4rank_results.jld2")
end

# ============================================================================
# Testset 6: Index Tracing Halo Tests (8 ranks, Partition(4,2))
# Tests the fold corner case where fold x-reversal maps corners to a
# different rank's data.
# ============================================================================

index_tracing_8rank_script = """
    using MPI
    MPI.Init()
    using JLD2
    include("distributed_tests_utils.jl")
    using Oceananigans.Grids: halo_size
    using Oceananigans.DistributedComputations: global_index_offset

    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    function fill_index_field!(field, dim; offset=0)
        int = interior(field)
        for I in CartesianIndices(int)
            int[I] = Float64(I.I[dim] + offset)
        end
        return nothing
    end

    test_cases = [
        (RightCenterFolded, 80, 80),
        (RightFaceFolded,   80, 81),
    ]

    partition = Partition(4, 2)
    locations = [
        ("CC", (Center(), Center(), Center())),
        ("FC", (Face(),   Center(), Center())),
        ("CF", (Center(), Face(),   Center())),
        ("FF", (Face(),   Face(),   Center())),
    ]
    Hx, Hy, Hz = 5, 5, 5

    results = Dict{String, Bool}()

    for (fold_topology, global_Nx, global_Ny) in test_cases
        fold_name = fold_topology == RightCenterFolded ? "UPivot" : "FPivot"

        arch = Distributed(CPU(); partition)
        grid = TripolarGrid(arch; size=(global_Nx, global_Ny, 1), z=(-1000, 0),
                            halo=(Hx, Hy, Hz), fold_topology)

        x_offset, y_offset, _ = global_index_offset(arch, (global_Nx, global_Ny, 1))

        serial_fields = Dict{String, Any}()
        if rank == 0
            serial_grid = TripolarGrid(; size=(global_Nx, global_Ny, 1), z=(-1000, 0),
                                       halo=(Hx, Hy, Hz), fold_topology)
            for (loc_name, loc) in locations
                fi = Field(loc, serial_grid)
                fj = Field(loc, serial_grid)
                fill_index_field!(fi, 1)
                fill_index_field!(fj, 2)
                fill_halo_regions!((fi, fj))
                serial_fields["\$(loc_name)_i"] = fi
                serial_fields["\$(loc_name)_j"] = fj
            end
        end
        MPI.Barrier(MPI.COMM_WORLD)

        config_pass = true
        for (loc_name, loc) in locations
            di = Field(loc, grid)
            dj = Field(loc, grid)
            fill_index_field!(di, 1; offset=x_offset)
            fill_index_field!(dj, 2; offset=y_offset)
            fill_halo_regions!((di, dj))

            gi = reconstruct_global_field(di)
            gj = reconstruct_global_field(dj)

            if rank == 0
                si = serial_fields["\$(loc_name)_i"]
                sj = serial_fields["\$(loc_name)_j"]
                if interior(gi, :, :, 1) != interior(si, :, :, 1) ||
                   interior(gj, :, :, 1) != interior(sj, :, :, 1)
                    config_pass = false
                end
            end

            if rank == 0
                si = serial_fields["\$(loc_name)_i"]
                sj = serial_fields["\$(loc_name)_j"]
                s_full_i = parent(si.data)[:, :, 1+Hz]
                s_full_j = parent(sj.data)[:, :, 1+Hz]
            else
                s_full_i = nothing
                s_full_j = nothing
            end
            s_full_i = MPI.bcast(s_full_i, 0, MPI.COMM_WORLD)
            s_full_j = MPI.bcast(s_full_j, 0, MPI.COMM_WORLD)

            local_i = di.data[:, :, 1]
            local_j = dj.data[:, :, 1]
            local_mis = 0
            for lj in axes(local_i, 2), li in axes(local_i, 1)
                spi = li + x_offset + Hx
                spj = lj + y_offset + Hy
                if spi in axes(s_full_i, 1) && spj in axes(s_full_i, 2)
                    if Int(local_i[li, lj]) != Int(s_full_i[spi, spj]) ||
                       Int(local_j[li, lj]) != Int(s_full_j[spi, spj])
                        local_mis += 1
                    end
                end
            end

            all_mis = MPI.Gather(local_mis, 0, MPI.COMM_WORLD)
            if rank == 0 && sum(all_mis) > 0
                config_pass = false
            end
        end

        key = "\$(fold_name)_Nx\$(global_Nx)_Ny\$(global_Ny)_4x2"
        if rank == 0
            results[key] = config_pass
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end

    if rank == 0
        jldopen("index_tracing_8rank_results.jld2", "w") do file
            for (key, pass) in results
                file[key] = pass
            end
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
"""

@testset "Test distributed TripolarGrid index tracing (8 ranks, Partition(4,2))..." begin
    write("index_tracing_8rank.jl", index_tracing_8rank_script)
    run(`$(mpiexec()) -n 8 $(Base.julia_cmd()) -O0 --check-bounds=yes index_tracing_8rank.jl`)
    rm("index_tracing_8rank.jl")

    results = jldopen("index_tracing_8rank_results.jld2")
    for key in keys(results)
        @testset "$key" begin
            @test results[key] == true
        end
    end
    close(results)
    rm("index_tracing_8rank_results.jld2")
end

# ============================================================================
# Testset 7: UPivot Simulations (slab, pencil, large-pencil)
# ============================================================================

run_upivot_slab = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_tripolar_grid(arch, "distributed_upivot_slab.jld2")
"""

run_upivot_pencil = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_upivot_pencil.jld2")
"""

run_upivot_large_pencil = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(4, 2))
    run_distributed_tripolar_grid(arch, "distributed_upivot_large_pencil.jld2")
"""

upivot_sim_configs = [
    (run_upivot_slab,         4, "distributed_upivot_slab.jld2",         "slab 1×4"),
    (run_upivot_pencil,       4, "distributed_upivot_pencil.jld2",       "pencil 2×2"),
    (run_upivot_large_pencil, 8, "distributed_upivot_large_pencil.jld2", "large-pencil 4×2"),
]

@testset "Test distributed TripolarGrid simulations..." begin
    # Serial baseline
    grid  = TripolarGrid(size = (80, 80, 1), z = (-1000, 0), halo = (5, 5, 5))
    grid  = analytical_immersed_tripolar_grid(grid)
    model = setup_simulation(grid)
    run_simulation!(model)

    us = interior(model.velocities.u, :, :, 1)
    vs = interior(model.velocities.v, :, :, 1)
    cs = interior(model.tracers.c, :, :, 1)
    ηs = interior(model.free_surface.displacement, :, :, 1)

    @testset "$cfg_name" for (script_str, nranks, jld2file, cfg_name) in upivot_sim_configs
        scriptfile = "distributed_upivot_sim.jl"
        write(scriptfile, script_str)
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
        rm(scriptfile)

        jld = jldopen(jld2file)
        up = jld["u"]; vp = jld["v"]; cp = jld["c"]; ηp = jld["η"]
        close(jld)
        rm(jld2file)

        @test all(us .≈ up)
        @test all(vs .≈ vp)
        @test all(cs .≈ cp)
        @test all(ηs .≈ ηp)
    end
end

# ============================================================================
# Testset 8: FPivot Simulations (slab, pencil, large-pencil)
# ============================================================================

fpivot_sim_script(partition_str, jld2file) = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = $partition_str)
    run_distributed_tripolar_grid(arch, "$jld2file";
                                  fold_topology = RightFaceFolded, Ny = 81)
"""

fpivot_sim_configs = [
    ("Partition(1, 4)", 4, "distributed_fpivot_slab.jld2",         "slab 1×4"),
    ("Partition(2, 2)", 4, "distributed_fpivot_pencil.jld2",       "pencil 2×2"),
    ("Partition(4, 2)", 8, "distributed_fpivot_large_pencil.jld2", "large-pencil 4×2"),
]

@testset "Test distributed FPivot TripolarGrid simulations..." begin
    # Serial baseline
    grid  = TripolarGrid(size = (80, 81, 1), z = (-1000, 0), halo = (5, 5, 5), fold_topology = RightFaceFolded)
    grid  = analytical_immersed_tripolar_grid(grid)
    model = setup_simulation(grid)
    run_simulation!(model)

    us = interior(model.velocities.u, :, :, 1)
    vs = interior(model.velocities.v, :, :, 1)
    cs = interior(model.tracers.c, :, :, 1)
    ηs = interior(model.free_surface.displacement, :, :, 1)

    @testset "$cfg_name" for (partition_str, nranks, jld2file, cfg_name) in fpivot_sim_configs
        script_str = fpivot_sim_script(partition_str, jld2file)
        scriptfile = "distributed_fpivot_sim.jl"
        write(scriptfile, script_str)
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
        rm(scriptfile)

        jld = jldopen(jld2file)
        up = jld["u"]; vp = jld["v"]; cp = jld["c"]; ηp = jld["η"]
        close(jld)
        rm(jld2file)

        @test all(us .≈ up)
        @test all(vs .≈ vp)
        @test all(cs .≈ cp)
        @test all(ηs .≈ ηp)
    end
end
