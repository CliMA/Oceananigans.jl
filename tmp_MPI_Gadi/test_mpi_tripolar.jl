include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

using MPI

const TESTSET = parse(Int, get(ENV, "TRIPOLAR_TESTSET", "0"))
const CONFIG  = parse(Int, get(ENV, "TRIPOLAR_CONFIG", "0"))

grid_var_names = ["Δxᶠᶠᵃ", "Δxᶜᶜᵃ", "Δxᶠᶜᵃ", "Δxᶜᶠᵃ",
                  "Δyᶠᶠᵃ", "Δyᶜᶜᵃ", "Δyᶠᶜᵃ", "Δyᶜᶠᵃ",
                  "Azᶠᶠᵃ", "Azᶜᶜᵃ", "Azᶠᶜᵃ", "Azᶜᶠᵃ"]

grid_recon_configs = [
    # (fold_name, fold_topology_str, Nx, Ny, halo)
    ("UPivot", "",                              12, 20, (2, 2, 2)),
    ("FPivot", "fold_topology = RightFaceFolded", 12, 21, (2, 2, 2)),
]

partition_configs = [
    # (partition_str, nranks, pname)
    ("Partition(1, 4)", 4, "1x4"),
    ("Partition(2, 2)", 4, "2x2"),
]

config_id = 0
@testset "Test distributed $fold_name TripolarGrid grid reconstruction" for (fold_name, fold_kw, Nx, Ny, halo) in grid_recon_configs
    @testset "$pname partition" for (partition_str, nranks, pname) in partition_configs
        global config_id += 1
        TESTSET in (0, 1) || continue
        CONFIG in (0, config_id) || continue
        tag = "grid_recon_$(fold_name)_$(pname)"

        # Build the fold_topology keyword string for interpolation
        fold_part = isempty(fold_kw) ? "" : ", $fold_kw"

        script = """
            using MPI
            MPI.Init()

            include("distributed_tests_utils.jl")

            arch = Distributed(CPU(); partition = $partition_str)
            local_grid = TripolarGrid(arch; size = ($Nx, $Ny, 1), z = (-1000, 0), halo = $halo$fold_part)

            nx, ny, _ = size(local_grid)
            rx, ry, _ = arch.local_index .- 1
            Rx, Ry, _ = arch.ranks

            filename = "$(tag)_" * string(arch.local_rank) * ".jld2"
            jldopen(filename, "w") do file
                file["local_Δxᶠᶠᵃ"] = local_grid.Δxᶠᶠᵃ
                file["local_Δxᶜᶜᵃ"] = local_grid.Δxᶜᶜᵃ
                file["local_Δxᶠᶜᵃ"] = local_grid.Δxᶠᶜᵃ
                file["local_Δxᶜᶠᵃ"] = local_grid.Δxᶜᶠᵃ
                file["local_Δyᶠᶠᵃ"] = local_grid.Δyᶠᶠᵃ
                file["local_Δyᶜᶜᵃ"] = local_grid.Δyᶜᶜᵃ
                file["local_Δyᶠᶜᵃ"] = local_grid.Δyᶠᶜᵃ
                file["local_Δyᶜᶠᵃ"] = local_grid.Δyᶜᶠᵃ
                file["local_Azᶠᶠᵃ"] = local_grid.Azᶠᶠᵃ
                file["local_Azᶜᶜᵃ"] = local_grid.Azᶜᶜᵃ
                file["local_Azᶠᶜᵃ"] = local_grid.Azᶠᶜᵃ
                file["local_Azᶜᶠᵃ"] = local_grid.Azᶜᶠᵃ
                file["nx"] = nx
                file["ny"] = ny
                file["rx"] = rx
                file["ry"] = ry
                file["Rx"] = Rx
                file["Ry"] = Ry
            end

            recon = reconstruct_global_grid(local_grid)
            if arch.local_rank == 0
                jldopen("$(tag)_global.jld2", "w") do file
                    file["recon_Δxᶠᶠᵃ"] = recon.Δxᶠᶠᵃ
                    file["recon_Δxᶜᶜᵃ"] = recon.Δxᶜᶜᵃ
                    file["recon_Δxᶠᶜᵃ"] = recon.Δxᶠᶜᵃ
                    file["recon_Δxᶜᶠᵃ"] = recon.Δxᶜᶠᵃ
                    file["recon_Δyᶠᶠᵃ"] = recon.Δyᶠᶠᵃ
                    file["recon_Δyᶜᶜᵃ"] = recon.Δyᶜᶜᵃ
                    file["recon_Δyᶠᶜᵃ"] = recon.Δyᶠᶜᵃ
                    file["recon_Δyᶜᶠᵃ"] = recon.Δyᶜᶠᵃ
                    file["recon_Azᶠᶠᵃ"] = recon.Azᶠᶠᵃ
                    file["recon_Azᶜᶜᵃ"] = recon.Azᶜᶜᵃ
                    file["recon_Azᶠᶜᵃ"] = recon.Azᶠᶜᵃ
                    file["recon_Azᶜᶠᵃ"] = recon.Azᶜᶠᵃ
                end
            end

            MPI.Barrier(MPI.COMM_WORLD)
            MPI.Finalize()
        """

        scriptfile = "$(tag)_script.jl"
        write(scriptfile, script)
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
        rm(scriptfile)

        # Create serial grid for comparison
        serial_grid = if isempty(fold_kw)
            TripolarGrid(; size = (Nx, Ny, 1), z = (-1000, 0), halo = halo)
        else
            TripolarGrid(; size = (Nx, Ny, 1), z = (-1000, 0), halo = halo, fold_topology = RightFaceFolded)
        end

        @testset "$fold_name $pname global var=$vname" for vname in grid_var_names
            global_data = jldopen("$(tag)_global.jld2")
            @test global_data["recon_$(vname)"] ≈ getproperty(serial_grid, Symbol(vname))
            close(global_data)
        end
        rm("$(tag)_global.jld2")

        @testset "$fold_name $pname rank $r local grid" for r in 0:(nranks - 1)
            rank_data = jldopen("$(tag)_$(r).jld2")
            nx = rank_data["nx"]
            ny = rank_data["ny"]
            rx = rank_data["rx"]
            ry = rank_data["ry"]
            Rx = rank_data["Rx"]
            Ry = rank_data["Ry"]

            x_offset = global_index_offset(Nx, Rx, rx + 1)
            y_offset = global_index_offset(Ny, Ry, ry + 1)
            irange = (1 + x_offset):(x_offset + nx)
            jrange = (1 + y_offset):(y_offset + ny)

            @testset "var=$vname" for vname in grid_var_names
                local_arr = rank_data["local_$(vname)"]
                serial_arr = getproperty(serial_grid, Symbol(vname))
                @test local_arr[1:nx, 1:ny] ≈ serial_arr[irange, jrange]
            end
            close(rank_data)
            rm("$(tag)_$(r).jld2")
        end
    end
end

field_recon_configs = [
    # (fold_name, fold_topology_str, Nx, Ny, halo)
    ("UPivot", "",                                40, 40, (5, 5, 5)),
    ("FPivot", "fold_topology = RightFaceFolded", 40, 41, (5, 5, 5)),
]

config_id = 0
@testset "Test distributed $fold_name TripolarGrid field reconstruction" for (fold_name, fold_kw, Nx, Ny, halo) in field_recon_configs
    @testset "$partition partition" for (partition_str, nranks, pname) in partition_configs
        global config_id += 1
        TESTSET in (0, 2) || continue
        CONFIG in (0, config_id) || continue
        tag = "field_recon_$(fold_name)_$(pname)"

        fold_part = isempty(fold_kw) ? "" : ", $fold_kw"

        script = """
            using MPI
            MPI.Init()

            include("distributed_tests_utils.jl")

            arch = Distributed(CPU(); partition = $partition_str)
            local_grid = TripolarGrid(arch; size = ($Nx, $Ny, 1), z = (-1000, 0), halo = $halo$fold_part)

            center_Ny = $(isempty(fold_kw) ? Ny : Ny - 1)
            u_init = [i + 10 * j for i in 1:$Nx, j in 1:center_Ny]
            v_init = [i + 10 * j for i in 1:$Nx, j in 1:$Ny]
            c_init = [i + 10 * j for i in 1:$Nx, j in 1:center_Ny]

            up = XFaceField(local_grid)
            vp = YFaceField(local_grid)
            cp = CenterField(local_grid)

            set!(up, u_init)
            set!(vp, v_init)
            set!(cp, c_init)

            fill_halo_regions!((up, vp, cp))

            nx, ny, _ = size(local_grid)
            rx, ry, _ = arch.local_index .- 1
            Rx, Ry, _ = arch.ranks

            filename = "$(tag)_" * string(arch.local_rank) * ".jld2"
            jldopen(filename, "w") do file
                file["local_u"] = up.data
                file["local_v"] = vp.data
                file["local_c"] = cp.data
                file["nx"] = nx
                file["ny"] = ny
                file["rx"] = rx
                file["ry"] = ry
                file["Rx"] = Rx
                file["Ry"] = Ry
            end

            gu = reconstruct_global_field(up)
            gv = reconstruct_global_field(vp)
            gc = reconstruct_global_field(cp)
            if arch.local_rank == 0
                jldopen("$(tag)_global.jld2", "w") do file
                    file["recon_u"] = Array(interior(gu))
                    file["recon_v"] = Array(interior(gv))
                    file["recon_c"] = Array(interior(gc))
                end
            end

            MPI.Barrier(MPI.COMM_WORLD)
            MPI.Finalize()
        """

        scriptfile = "$(tag)_script.jl"
        write(scriptfile, script)
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
        rm(scriptfile)

        # Create serial grid and fields
        serial_grid = if isempty(fold_kw)
            TripolarGrid(; size = (Nx, Ny, 1), z = (-1000, 0), halo = halo)
        else
            TripolarGrid(; size = (Nx, Ny, 1), z = (-1000, 0), halo = halo, fold_topology = RightFaceFolded)
        end

        # Center-y interior size: Ny for UPivot, Ny-1 for FPivot
        center_Ny = isempty(fold_kw) ? Ny : Ny - 1
        u_init = [i + 10 * j for i in 1:Nx, j in 1:center_Ny]
        v_init = [i + 10 * j for i in 1:Nx, j in 1:Ny]
        c_init = [i + 10 * j for i in 1:Nx, j in 1:center_Ny]

        us = XFaceField(serial_grid)
        vs = YFaceField(serial_grid)
        cs = CenterField(serial_grid)

        set!(us, u_init)
        set!(vs, v_init)
        set!(cs, c_init)

        fill_halo_regions!((us, vs, cs))

        @testset "$fold_name $pname reconstructed global fields" begin
            global_data = jldopen("$(tag)_global.jld2")
            @test global_data["recon_u"] ≈ interior(us)
            @test global_data["recon_v"] ≈ interior(vs)
            @test global_data["recon_c"] ≈ interior(cs)
            close(global_data)
            rm("$(tag)_global.jld2")
        end

        @testset "$fold_name $pname rank $r local fields" for r in 0:(nranks - 1)
            rank_data = jldopen("$(tag)_$(r).jld2")
            rx = rank_data["rx"]
            ry = rank_data["ry"]
            Rx = rank_data["Rx"]
            Ry = rank_data["Ry"]

            x_offset = global_index_offset(Nx, Rx, rx + 1)
            y_offset = global_index_offset(Ny, Ry, ry + 1)

            @testset "field=$fname" for (fname, serial_field) in [("u", us), ("v", vs), ("c", cs)]
                local_data  = rank_data["local_$(fname)"]
                serial_data = serial_field.data

                # Array-level comparison: local[i,j,1] ≈ serial[i+x_offset, j+y_offset, 1]
                irange = axes(local_data, 1) .+ x_offset
                jrange = axes(local_data, 2) .+ y_offset
                @test all(irange .∈ Ref(axes(serial_data, 1)))
                @test all(jrange .∈ Ref(axes(serial_data, 2)))
                @test local_data[:, :, 1] ≈ serial_data[irange, jrange, 1]
            end
            close(rank_data)
            rm("$(tag)_$(r).jld2")
        end
    end
end

tripolar_boundary_conditions = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")

    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = TripolarGrid(arch; size = (20, 20, 1), z = (-1000, 0))

    # Build initial condition
    serial_grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0))

    vs =  YFaceField(serial_grid)
    cs = CenterField(serial_grid)

    I1 = [i + j * 100 for i in 1:20, j in 1:20]

    set!(vs, I1)
    set!(cs, I1)

    fill_halo_regions!((vs, cs))

    v =  YFaceField(grid)
    c = CenterField(grid)

    set!(v, vs)
    set!(c, cs)

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
    TESTSET in (0, 3) || return nothing
    # Run the serial computation
    grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0))

    I1 = [i + j * 100 for i in 1:20, j in 1:20]

    v = YFaceField(grid)
    c = CenterField(grid)

    set!(v, I1)
    set!(c, I1)

    fill_halo_regions!((v, c))

    write("distributed_boundary_tests.jl", tripolar_boundary_conditions)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 --check-bounds=yes distributed_boundary_tests.jl`)
    rm("distributed_boundary_tests.jl")

    # Retrieve Parallel quantities from rank 1 (the north-west rank)
    vp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["v"];
    cp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["c"];

    # Retrieve Parallel quantities from rank 3 (the north-east rank)
    vp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["v"];
    cp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["c"];

    @test v.data[-3:14, end-3:end-1, 1] ≈ vp1.parent[:, end-3:end-1, 5]
    @test c.data[-3:14, end-3:end-1, 1] ≈ cp1.parent[:, end-3:end-1, 5]
    @test v.data[7:end, 7:end-1, 1] ≈ vp3.parent[:, 1:end-1, 5]
    @test c.data[7:end, 7:end-1, 1] ≈ cp3.parent[:, 1:end-1, 5]
end

fpivot_boundary_conditions = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")

    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = TripolarGrid(arch; size = (20, 21, 1), z = (-1000, 0), fold_topology = RightFaceFolded)

    # Build initial condition
    serial_grid = TripolarGrid(size = (20, 21, 1), z = (-1000, 0), fold_topology = RightFaceFolded)

    vs =  YFaceField(serial_grid)
    cs = CenterField(serial_grid)

    I_v = [i + j * 100 for i in 1:20, j in 1:21]
    I_c = [i + j * 100 for i in 1:20, j in 1:20]

    set!(vs, I_v)
    set!(cs, I_c)

    fill_halo_regions!((vs, cs))

    v =  YFaceField(grid)
    c = CenterField(grid)

    set!(v, vs)
    set!(c, cs)

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
    TESTSET in (0, 4) || return nothing
    # Run the serial computation
    grid = TripolarGrid(size = (20, 21, 1), z = (-1000, 0), fold_topology = RightFaceFolded)

    I_v = [i + j * 100 for i in 1:20, j in 1:21]
    I_c = [i + j * 100 for i in 1:20, j in 1:20]

    v = YFaceField(grid)
    c = CenterField(grid)

    set!(v, I_v)
    set!(c, I_c)

    fill_halo_regions!((v, c))

    write("distributed_fpivot_boundary_tests.jl", fpivot_boundary_conditions)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 --check-bounds=yes distributed_fpivot_boundary_tests.jl`)
    rm("distributed_fpivot_boundary_tests.jl")

    # Retrieve Parallel quantities from rank 1 (the north-west rank)
    vp1 = jldopen("distributed_fpivot_boundary_conditions_1.jld2")["v"];
    cp1 = jldopen("distributed_fpivot_boundary_conditions_1.jld2")["c"];

    # Retrieve Parallel quantities from rank 3 (the north-east rank)
    vp3 = jldopen("distributed_fpivot_boundary_conditions_3.jld2")["v"];
    cp3 = jldopen("distributed_fpivot_boundary_conditions_3.jld2")["c"];

    # Ny=21 with Partition(2,2) means each y-rank gets ~10-11 rows.
    # Rank 1 (NW) covers rows 1:10 in x and 12:21 in y (the northern half).
    # With halo=3, compare the north halo region of rank 1 with serial data.
    @test v.data[-3:14, end-3:end-1, 1] ≈ vp1.parent[:, end-3:end-1, 5]
    @test c.data[-3:14, end-3:end-1, 1] ≈ cp1.parent[:, end-3:end-1, 5]
    @test v.data[7:end, 7:end-1, 1] ≈ vp3.parent[:, 1:end-1, 5]
    @test c.data[7:end, 7:end-1, 1] ≈ cp3.parent[:, 1:end-1, 5]
end

run_slab_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_tripolar_grid(arch, "distributed_yslab_tripolar.jld2")
"""

run_pencil_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_pencil_tripolar.jld2")
"""

run_large_pencil_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(4, 2))
    run_distributed_tripolar_grid(arch, "distributed_large_pencil_tripolar.jld2")
"""

sim_configs = [
    # (script_string, nranks, jld2_filename, config_name)
    (run_slab_distributed_grid,         4, "distributed_yslab_tripolar.jld2",         "slab 1×4"),
    (run_pencil_distributed_grid,       4, "distributed_pencil_tripolar.jld2",        "pencil 2×2"),
    (run_large_pencil_distributed_grid, 8, "distributed_large_pencil_tripolar.jld2",  "large-pencil 4×2"),
]

@testset "Test distributed TripolarGrid simulations..." begin
    TESTSET in (0, 5) || return nothing

    # Run the serial computation
    grid  = TripolarGrid(size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5))
    grid  = analytical_immersed_tripolar_grid(grid)
    model = run_distributed_simulation(grid)

    us = Array(interior(model.velocities.u))
    vs = Array(interior(model.velocities.v))
    cs = Array(interior(model.tracers.c))
    ηs = Array(interior(model.free_surface.displacement))

    @testset "$cfg_name" for (cfg_id, (script_str, nranks, jld2file, cfg_name)) in enumerate(sim_configs)
        CONFIG in (0, cfg_id) || continue

        scriptfile = "distributed_sim_$(cfg_id).jl"
        write(scriptfile, script_str)
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
        rm(scriptfile)

        jld = jldopen(jld2file)
        up = jld["u"]; vp = jld["v"]; cp = jld["c"]; ηp = jld["η"]
        close(jld)
        rm(jld2file)

        @test us ≈ up
        @test vs ≈ vp
        @test cs ≈ cp
        @test ηs ≈ ηp
    end
end

fpivot_sim_script(partition_str, nranks, jld2file) = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = $partition_str)
    run_distributed_tripolar_grid(arch, "$jld2file";
                                  fold_topology = RightFaceFolded, Ny = 121)
"""

fpivot_sim_configs = [
    ("Partition(1, 4)", 4, "distributed_fpivot_slab_tripolar.jld2",        "slab 1×4"),
    ("Partition(2, 2)", 4, "distributed_fpivot_pencil_tripolar.jld2",      "pencil 2×2"),
    ("Partition(4, 2)", 8, "distributed_fpivot_large_pencil_tripolar.jld2", "large-pencil 4×2"),
]

@testset "Test distributed FPivot TripolarGrid simulations..." begin
    TESTSET in (0, 6) || return nothing

    # Serial: setup, capture ICs, run
    grid  = TripolarGrid(size = (40, 121, 1), z = (-1000, 0), halo = (5, 5, 5), fold_topology = RightFaceFolded)
    grid  = analytical_immersed_tripolar_grid(grid)
    model = setup_simulation(grid)

    us0 = Array(interior(model.velocities.u))
    vs0 = Array(interior(model.velocities.v))
    cs0 = Array(interior(model.tracers.c))
    ηs0 = Array(interior(model.free_surface.displacement))

    run_simulation!(model)

    us = Array(interior(model.velocities.u))
    vs = Array(interior(model.velocities.v))
    cs = Array(interior(model.tracers.c))
    ηs = Array(interior(model.free_surface.displacement))

    @testset "$cfg_name" for (cfg_id, (partition_str, nranks, jld2file, cfg_name)) in enumerate(fpivot_sim_configs)
        CONFIG in (0, cfg_id) || continue

        script_str = fpivot_sim_script(partition_str, nranks, jld2file)
        scriptfile = "distributed_fpivot_sim_$(cfg_id).jl"
        write(scriptfile, script_str)
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --check-bounds=yes $scriptfile`)
        rm(scriptfile)

        jld = jldopen(jld2file)
        up  = jld["u"];  vp  = jld["v"];  cp  = jld["c"];  ηp  = jld["η"]
        up0 = jld["u0"]; vp0 = jld["v0"]; cp0 = jld["c0"]; ηp0 = jld["η0"]
        close(jld)
        rm(jld2file)

        @testset "ICs" begin
            @test us0 ≈ up0
            @test vs0 ≈ vp0
            @test cs0 ≈ cp0
            @test ηs0 ≈ ηp0
        end

        @testset "final" begin
            @test us ≈ up
            @test vs ≈ vp
            @test cs ≈ cp
            @test ηs ≈ ηp
        end
    end
end
