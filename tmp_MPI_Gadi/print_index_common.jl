using MPI
MPI.Initialized() || MPI.Init()

using Oceananigans
using Oceananigans.Grids: halo_size, topology
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: set!, interior
using Oceananigans.DistributedComputations

const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const nranks = MPI.Comm_size(MPI.COMM_WORLD)

function print_rank(r, args...)
    rank == r && println(args...)
    flush(stdout)
    MPI.Barrier(MPI.COMM_WORLD)
    return nothing
end

function print_int_matrix(name, data; max_cols=20)
    ax1, ax2 = axes(data)
    println("  $name  axes=$(ax1) × $(ax2)  parent_size=$(size(parent(data)))")
    # Print column headers
    cols = collect(ax2)
    if length(cols) > max_cols
        cols = cols[1:max_cols]
        println("  (showing first $max_cols columns)")
    end
    header = lpad("", 6) * join([lpad(j, 4) for j in cols])
    println(header)
    for i in ax1
        row = lpad(i, 5) * " " * join([lpad(Int(data[i, j]), 4) for j in cols])
        println(row)
    end
    println()
end

function print_int_matrix(name, data::Array; max_cols=20)
    ax1, ax2 = axes(data)
    println("  $name  axes=$(ax1) × $(ax2)  size=$(size(data))")
    cols = collect(ax2)
    if length(cols) > max_cols
        cols = cols[1:max_cols]
        println("  (showing first $max_cols columns)")
    end
    header = lpad("", 6) * join([lpad(j, 4) for j in cols])
    println(header)
    for i in ax1
        row = lpad(i, 5) * " " * join([lpad(Int(data[i, j]), 4) for j in cols])
        println(row)
    end
    println()
end

function fill_index_field!(field, dim; offset=0)
    int = interior(field)
    for I in CartesianIndices(int)
        int[I] = Float64(I.I[dim] + offset)
    end
    return nothing
end

function run_diagnostic(; loc, loc_name, fold_topology, partition, global_Nx, global_Ny, Hx, Hy, Hz)
    fold_name = fold_topology == RightCenterFolded ? "UPivot" : "FPivot"
    halo = (Hx, Hy, Hz)

    print_rank(0, "\n", "="^70)
    print_rank(0, "$loc_name $fold_name partition=$partition size=($global_Nx,$global_Ny,1) halo=($Hx,$Hy,$Hz)")
    print_rank(0, "="^70)

    # --- Serial field on rank 0 ---
    if rank == 0
        serial_grid = TripolarGrid(; size=(global_Nx, global_Ny, 1), z=(-1000, 0), halo, fold_topology)
        fi = Field(loc, serial_grid)
        fj = Field(loc, serial_grid)
        fill_index_field!(fi, 1)
        fill_index_field!(fj, 2)
        fill_halo_regions!((fi, fj))

        println("\n--- Serial $loc_name i-field (offset array, k=1 slice) ---")
        print_int_matrix("serial_i", fi.data[:, :, 1])
        println("--- Serial $loc_name j-field (offset array, k=1 slice) ---")
        print_int_matrix("serial_j", fj.data[:, :, 1])
        println("--- Serial $loc_name i-field (parent array, k=1 slice) ---")
        print_int_matrix("parent_i", parent(fi.data)[:, :, 1+Hz])
    end
    MPI.Barrier(MPI.COMM_WORLD)

    # --- Distributed fields ---
    arch = Distributed(CPU(); partition)
    dist_grid = TripolarGrid(arch; size=(global_Nx, global_Ny, 1), z=(-1000, 0), halo, fold_topology)

    rx, ry, _ = arch.local_index .- 1
    Rx, Ry, _ = arch.ranks
    x_offset = rx * (global_Nx ÷ Rx)
    y_offset = ry * (global_Ny ÷ Ry)

    di = Field(loc, dist_grid)
    dj = Field(loc, dist_grid)
    fill_index_field!(di, 1; offset=x_offset)
    fill_index_field!(dj, 2; offset=y_offset)
    fill_halo_regions!((di, dj))

    # Print each rank's local data sequentially
    for r in 0:(nranks - 1)
        if rank == r
            println("\n--- Rank $r ($loc_name, local_index=$(arch.local_index)) ---")
            println("  x_offset=$x_offset  y_offset=$y_offset")
            println("  grid size=$(size(dist_grid))  field size=$(size(di))")
            println("  data axes: $(axes(di.data))")
            println("  i-field (k=1 slice):")
            print_int_matrix("local_i", di.data[:, :, 1])
            println("  j-field (k=1 slice):")
            print_int_matrix("local_j", dj.data[:, :, 1])
        end
        flush(stdout)
        MPI.Barrier(MPI.COMM_WORLD)
    end
end
