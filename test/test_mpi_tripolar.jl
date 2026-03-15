include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

using MPI
using Oceananigans.DistributedComputations: global_index_offset

grid_var_names = [:Δxᶠᶠᵃ, :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ,
                  :Δyᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ,
                  :Azᶠᶠᵃ, :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ]

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
        u_init = [i + 10j for i in 1:Nx, j in 1:center_Ny]
        v_init = [i + 10j for i in 1:Nx, j in 1:Ny]
        c_init = [i + 10j for i in 1:Nx, j in 1:center_Ny]

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
        u_init = [i + 10j for i in 1:Nx, j in 1:center_Ny]
        v_init = [i + 10j for i in 1:Nx, j in 1:Ny]
        c_init = [i + 10j for i in 1:Nx, j in 1:center_Ny]

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

index_tracing_4rank_script = """
    using MPI
    MPI.Init()
    using JLD2
    include("distributed_tests_utils.jl")
    using Oceananigans.Grids: halo_size
    using Oceananigans.DistributedComputations: global_index_offset

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)

    function print_rank0(args...)
        rank == 0 && println(args...)
        return nothing
    end

    function classify_region(li, lj, Nx, Ny)
        in_west  = li < 1
        in_east  = li > Nx
        in_north = lj > Ny
        in_south = lj < 1
        if in_north && in_west;  return "NW corner"
        elseif in_north && in_east;  return "NE corner"
        elseif in_south && in_west;  return "SW corner"
        elseif in_south && in_east;  return "SE corner"
        elseif in_north;  return "north halo"
        elseif in_south;  return "south halo"
        elseif in_west;   return "west halo"
        elseif in_east;   return "east halo"
        else;             return "interior"
        end
    end

    function format_mismatch(li, lj, spi, spj, loc_i, loc_j, ser_i, ser_j)
        if loc_i == 0 && loc_j == 0
            return "local(\$li,\$lj) [= global(\$spi,\$spj)]: NEVER WRITTEN (zero) - expected global(\$ser_i,\$ser_j)"
        else
            return "local(\$li,\$lj) [= global(\$spi,\$spj)]: WRONG SOURCE - got global(\$loc_i,\$loc_j), expected global(\$ser_i,\$ser_j)"
        end
    end

    function analyze_mismatches(mismatch_list, Nx, Ny, Hy)
        isempty(mismatch_list) && return String[]
        lines = String[]
        region_counts = Dict{String, Int}()
        for (li, lj, _, _, _, _, _, _) in mismatch_list
            r = classify_region(li, lj, Nx, Ny)
            region_counts[r] = get(region_counts, r, 0) + 1
        end
        push!(lines, "Affected regions: " * join(["\$r (\$n)" for (r, n) in sort(collect(region_counts), by=x->-x[2])], ", "))
        y_counts = Dict{Int,Int}()
        all_li = Set{Int}()
        for (li, lj, _, _, _, _, _, _) in mismatch_list
            y_counts[lj] = get(y_counts, lj, 0) + 1
            push!(all_li, li)
        end
        li_range = sort(collect(all_li))
        row_strs = String[]
        for lj in sort(collect(keys(y_counts)))
            n = y_counts[lj]
            label = lj > Ny ? "y=\$lj (fold halo \$(lj-Ny))" : lj == Ny ? "y=\$lj (fold line)" : "y=\$lj"
            prefix = n >= length(li_range) && length(li_range) > 1 ? "ENTIRE " : ""
            push!(row_strs, "\$prefix\$label (\$n cells)")
        end
        !isempty(row_strs) && push!(lines, "Affected rows: " * join(row_strs, ", "))
        x_counts = Dict{Int,Int}()
        for (li, _, _, _, _, _, _, _) in mismatch_list
            x_counts[li] = get(x_counts, li, 0) + 1
        end
        col_strs = ["x=\$li (\$(li < 1 ? "west halo" : li > Nx ? "east halo" : li == 1 ? "col 1" : "interior")) (\$n cells)" for (li, n) in sort(collect(x_counts))]
        length(col_strs) <= 20 && push!(lines, "Affected cols: " * join(col_strs, ", "))
        n_zero  = count(m -> m[5] == 0 && m[6] == 0, mismatch_list)
        n_wrong = length(mismatch_list) - n_zero
        if n_zero > 0 && n_wrong > 0;    push!(lines, "Error types: \$n_zero NEVER WRITTEN, \$n_wrong WRONG SOURCE")
        elseif n_zero > 0;               push!(lines, "Error type: ALL \$n_zero NEVER WRITTEN")
        else;                            push!(lines, "Error type: ALL \$n_wrong WRONG SOURCE")
        end
        push!(lines, "First mismatches:")
        for (k, m) in enumerate(mismatch_list)
            k > 20 && break
            push!(lines, "  " * format_mismatch(m...))
        end
        return lines
    end

    function fill_index_field!(field, dim; offset=0)
        int = interior(field)
        for I in CartesianIndices(int)
            int[I] = I.I[dim] + offset
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

    rank == 0 && println("\\nRunning index-tracing halo fill tests...")

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
                    mis_i = count(interior(gi, :, :, 1) .!= interior(si, :, :, 1))
                    mis_j = count(interior(gj, :, :, 1) .!= interior(sj, :, :, 1))
                    if mis_i == 0 && mis_j == 0
                        print_rank0("  \$(fold_name) \$(loc_name) interior: PASS")
                    else
                        config_pass = false
                        print_rank0("  \$(fold_name) \$(loc_name) interior: FAIL (\$(mis_i) i-mismatches, \$(mis_j) j-mismatches)")
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
                local_mismatches = 0
                mismatch_list = Tuple{Int,Int,Int,Int,Int,Int,Int,Int}[]
                for lj in axes(local_i, 2), li in axes(local_i, 1)
                    spi = li + x_offset + Hx
                    spj = lj + y_offset + Hy
                    if spi in axes(s_full_i, 1) && spj in axes(s_full_i, 2)
                        loc_iv = Int(local_i[li, lj])
                        loc_jv = Int(local_j[li, lj])
                        ser_iv = Int(s_full_i[spi, spj])
                        ser_jv = Int(s_full_j[spi, spj])
                        if loc_iv != ser_iv || loc_jv != ser_jv
                            local_mismatches += 1
                            length(mismatch_list) < 200 && push!(mismatch_list, (li, lj, spi, spj, loc_iv, loc_jv, ser_iv, ser_jv))
                        end
                    end
                end

                all_local_mis = MPI.Gather(local_mismatches, 0, MPI.COMM_WORLD)
                if rank == 0
                    total_mis = sum(all_local_mis)
                    if total_mis == 0
                        print_rank0("  \$(fold_name) \$(loc_name) local halos: PASS (all ranks match serial)")
                    else
                        config_pass = false
                        per_rank = join(["r\$r=\$(all_local_mis[r+1])" for r in 0:nranks-1 if all_local_mis[r+1] > 0], ", ")
                        print_rank0("  \$(fold_name) \$(loc_name) local halos: FAIL (\$(total_mis) total: \$(per_rank))")
                    end
                end

                # Per-rank structural analysis (sequential to avoid interleaving)
                Nx_loc, Ny_loc, _ = size(grid)
                for r in 0:(nranks - 1)
                    if rank == r && !isempty(mismatch_list)
                        println("    Rank \$r (\$(local_mismatches) mismatches):")
                        for line in analyze_mismatches(mismatch_list, Nx_loc, Ny_loc, Hy)
                            println("      ", line)
                        end
                    end
                    MPI.Barrier(MPI.COMM_WORLD)
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

    if rank == 0
        println("\\n", "="^60)
        println("INDEX TRACING SUMMARY")
        println("="^60)
        for (key, pass) in sort(collect(results))
            println("  \$(key): \$(pass ? "PASS" : "FAIL")")
        end
        n_pass = count(values(results))
        println("\\n  \$(n_pass) / \$(length(results)) configurations passed")
        println("="^60)
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

index_tracing_8rank_script = """
    using MPI
    MPI.Init()
    using JLD2
    include("distributed_tests_utils.jl")
    using Oceananigans.Grids: halo_size
    using Oceananigans.DistributedComputations: global_index_offset

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)

    function print_rank0(args...)
        rank == 0 && println(args...)
        return nothing
    end

    function classify_region(li, lj, Nx, Ny)
        in_west  = li < 1
        in_east  = li > Nx
        in_north = lj > Ny
        in_south = lj < 1
        if in_north && in_west;  return "NW corner"
        elseif in_north && in_east;  return "NE corner"
        elseif in_south && in_west;  return "SW corner"
        elseif in_south && in_east;  return "SE corner"
        elseif in_north;  return "north halo"
        elseif in_south;  return "south halo"
        elseif in_west;   return "west halo"
        elseif in_east;   return "east halo"
        else;             return "interior"
        end
    end

    function format_mismatch(li, lj, spi, spj, loc_i, loc_j, ser_i, ser_j)
        if loc_i == 0 && loc_j == 0
            return "local(\$li,\$lj) [= global(\$spi,\$spj)]: NEVER WRITTEN (zero) - expected global(\$ser_i,\$ser_j)"
        else
            return "local(\$li,\$lj) [= global(\$spi,\$spj)]: WRONG SOURCE - got global(\$loc_i,\$loc_j), expected global(\$ser_i,\$ser_j)"
        end
    end

    function analyze_mismatches(mismatch_list, Nx, Ny, Hy)
        isempty(mismatch_list) && return String[]
        lines = String[]
        region_counts = Dict{String, Int}()
        for (li, lj, _, _, _, _, _, _) in mismatch_list
            r = classify_region(li, lj, Nx, Ny)
            region_counts[r] = get(region_counts, r, 0) + 1
        end
        push!(lines, "Affected regions: " * join(["\$r (\$n)" for (r, n) in sort(collect(region_counts), by=x->-x[2])], ", "))
        y_counts = Dict{Int,Int}()
        all_li = Set{Int}()
        for (li, lj, _, _, _, _, _, _) in mismatch_list
            y_counts[lj] = get(y_counts, lj, 0) + 1
            push!(all_li, li)
        end
        li_range = sort(collect(all_li))
        row_strs = String[]
        for lj in sort(collect(keys(y_counts)))
            n = y_counts[lj]
            label = lj > Ny ? "y=\$lj (fold halo \$(lj-Ny))" : lj == Ny ? "y=\$lj (fold line)" : "y=\$lj"
            prefix = n >= length(li_range) && length(li_range) > 1 ? "ENTIRE " : ""
            push!(row_strs, "\$prefix\$label (\$n cells)")
        end
        !isempty(row_strs) && push!(lines, "Affected rows: " * join(row_strs, ", "))
        x_counts = Dict{Int,Int}()
        for (li, _, _, _, _, _, _, _) in mismatch_list
            x_counts[li] = get(x_counts, li, 0) + 1
        end
        col_strs = ["x=\$li (\$(li < 1 ? "west halo" : li > Nx ? "east halo" : li == 1 ? "col 1" : "interior")) (\$n cells)" for (li, n) in sort(collect(x_counts))]
        length(col_strs) <= 20 && push!(lines, "Affected cols: " * join(col_strs, ", "))
        n_zero  = count(m -> m[5] == 0 && m[6] == 0, mismatch_list)
        n_wrong = length(mismatch_list) - n_zero
        if n_zero > 0 && n_wrong > 0;    push!(lines, "Error types: \$n_zero NEVER WRITTEN, \$n_wrong WRONG SOURCE")
        elseif n_zero > 0;               push!(lines, "Error type: ALL \$n_zero NEVER WRITTEN")
        else;                            push!(lines, "Error type: ALL \$n_wrong WRONG SOURCE")
        end
        push!(lines, "First mismatches:")
        for (k, m) in enumerate(mismatch_list)
            k > 20 && break
            push!(lines, "  " * format_mismatch(m...))
        end
        return lines
    end

    function fill_index_field!(field, dim; offset=0)
        int = interior(field)
        for I in CartesianIndices(int)
            int[I] = I.I[dim] + offset
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

    rank == 0 && println("\\nRunning index-tracing halo fill tests...")

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

            # Interior check (rank 0)
            if rank == 0
                si = serial_fields["\$(loc_name)_i"]
                sj = serial_fields["\$(loc_name)_j"]
                mis_i = count(interior(gi, :, :, 1) .!= interior(si, :, :, 1))
                mis_j = count(interior(gj, :, :, 1) .!= interior(sj, :, :, 1))
                if mis_i == 0 && mis_j == 0
                    print_rank0("  \$(fold_name) \$(loc_name) interior: PASS")
                else
                    config_pass = false
                    print_rank0("  \$(fold_name) \$(loc_name) interior: FAIL (\$(mis_i) i-mismatches, \$(mis_j) j-mismatches)")
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
            local_mismatches = 0
            mismatch_list = Tuple{Int,Int,Int,Int,Int,Int,Int,Int}[]
            for lj in axes(local_i, 2), li in axes(local_i, 1)
                spi = li + x_offset + Hx
                spj = lj + y_offset + Hy
                if spi in axes(s_full_i, 1) && spj in axes(s_full_i, 2)
                    loc_iv = Int(local_i[li, lj])
                    loc_jv = Int(local_j[li, lj])
                    ser_iv = Int(s_full_i[spi, spj])
                    ser_jv = Int(s_full_j[spi, spj])
                    if loc_iv != ser_iv || loc_jv != ser_jv
                        local_mismatches += 1
                        length(mismatch_list) < 200 && push!(mismatch_list, (li, lj, spi, spj, loc_iv, loc_jv, ser_iv, ser_jv))
                    end
                end
            end

            all_local_mis = MPI.Gather(local_mismatches, 0, MPI.COMM_WORLD)
            if rank == 0
                total_mis = sum(all_local_mis)
                if total_mis == 0
                    print_rank0("  \$(fold_name) \$(loc_name) local halos: PASS (all ranks match serial)")
                else
                    config_pass = false
                    per_rank = join(["r\$r=\$(all_local_mis[r+1])" for r in 0:nranks-1 if all_local_mis[r+1] > 0], ", ")
                    print_rank0("  \$(fold_name) \$(loc_name) local halos: FAIL (\$(total_mis) total: \$(per_rank))")
                end
            end

            # Per-rank structural analysis (sequential to avoid interleaving)
            Nx_loc, Ny_loc, _ = size(grid)
            for r in 0:(nranks - 1)
                if rank == r && !isempty(mismatch_list)
                    println("    Rank \$r (\$(local_mismatches) mismatches):")
                    for line in analyze_mismatches(mismatch_list, Nx_loc, Ny_loc, Hy)
                        println("      ", line)
                    end
                end
                MPI.Barrier(MPI.COMM_WORLD)
            end
        end

        key = "\$(fold_name)_Nx\$(global_Nx)_Ny\$(global_Ny)_4x2"
        if rank == 0
            results[key] = config_pass
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end

    if rank == 0
        println("\\n", "="^60)
        println("INDEX TRACING SUMMARY")
        println("="^60)
        for (key, pass) in sort(collect(results))
            println("  \$(key): \$(pass ? "PASS" : "FAIL")")
        end
        n_pass = count(values(results))
        println("\\n  \$(n_pass) / \$(length(results)) configurations passed")
        println("="^60)
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

run_slab_upivot_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_tripolar_grid(arch, "distributed_yslab_upivot_tripolar.jld2")
"""

run_pencil_upivot_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_pencil_upivot_tripolar.jld2")
"""

run_large_pencil_upivot_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(4, 2))
    run_distributed_tripolar_grid(arch, "distributed_large_pencil_upivot_tripolar.jld2")
"""

upivot_sim_configs = [
    (run_slab_upivot_distributed_grid,         4, "distributed_yslab_upivot_tripolar.jld2",         "slab 1×4"),
    (run_pencil_upivot_distributed_grid,       4, "distributed_pencil_upivot_tripolar.jld2",       "pencil 2×2"),
    (run_large_pencil_upivot_distributed_grid, 8, "distributed_large_pencil_upivot_tripolar.jld2", "large-pencil 4×2"),
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
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 $scriptfile`)
        rm(scriptfile, force=true)

        jld = jldopen(jld2file)
        up = jld["u"]; vp = jld["v"]; cp = jld["c"]; ηp = jld["η"]
        close(jld)
        rm(jld2file, force=true)

        @test all(us .≈ up)
        @test all(vs .≈ vp)
        @test all(cs .≈ cp)
        @test all(ηs .≈ ηp)
    end
end

run_slab_fpivot_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_tripolar_grid(arch, "distributed_yslab_fpivot_tripolar.jld2";
                                  fold_topology = RightFaceFolded, Ny = 81)
"""

run_pencil_fpivot_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_pencil_fpivot_tripolar.jld2";
                                  fold_topology = RightFaceFolded, Ny = 81)
"""

run_large_pencil_fpivot_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(4, 2))
    run_distributed_tripolar_grid(arch, "distributed_large_pencil_fpivot_tripolar.jld2";
                                  fold_topology = RightFaceFolded, Ny = 81)
"""

fpivot_sim_configs = [
    (run_slab_fpivot_distributed_grid,         4, "distributed_yslab_fpivot_tripolar.jld2",         "slab 1×4"),
    (run_pencil_fpivot_distributed_grid,       4, "distributed_pencil_fpivot_tripolar.jld2",       "pencil 2×2"),
    (run_large_pencil_fpivot_distributed_grid, 8, "distributed_large_pencil_fpivot_tripolar.jld2", "large-pencil 4×2"),
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

    @testset "$cfg_name" for (script_str, nranks, jld2file, cfg_name) in fpivot_sim_configs
        scriptfile = "distributed_fpivot_sim.jl"
        write(scriptfile, script_str)
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
        rm(scriptfile, force=true)

        jld = jldopen(jld2file)
        up = jld["u"]; vp = jld["v"]; cp = jld["c"]; ηp = jld["η"]
        close(jld)
        rm(jld2file, force=true)

        @test all(us .≈ up)
        @test all(vs .≈ vp)
        @test all(cs .≈ cp)
        @test all(ηs .≈ ηp)
    end
end
