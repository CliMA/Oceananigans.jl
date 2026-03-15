# ============================================================================
# Index-Tracing Halo Fill Test
# ============================================================================
#
# PURPOSE: Verify that distributed TripolarGrid halo filling produces exactly
# the same values as serial halo filling, for all field locations (CC, FC, CF, FF).
#
# METHODOLOGY - INDEX ENCODING:
# Each field cell is filled with its own global index value before halo filling:
#   field_i[x,y] = global x-index of that cell
#   field_j[x,y] = global y-index of that cell
#
# After fill_halo_regions!, halo cells (including fold halos at the north boundary)
# should contain the index values of the SOURCE cell they were copied from.
# For fold halos, this means x-reversed indices from the fold partner.
#
# HOW TO INTERPRET MISMATCHES:
# The VALUE in a mismatched cell tells you WHERE the data came from:
#
#   "WRONG SOURCE - got data from global(36,40), expected from global(5,40)"
#     → The cell received data from global column 36 instead of column 5.
#       This means the fold reversal or MPI communication used the wrong source.
#       The j-value tells you which row the data came from.
#
#   "NEVER WRITTEN (zero) - expected data from global(5,40)"
#     → The cell was never filled (still zero-initialized).
#       This means the MPI buffer was empty, recv didn't copy data, or the
#       fold reversal skipped this cell entirely.
#
# STRUCTURAL ANALYSIS:
# The test also reports which regions are affected (north halo, west halo,
# corners, interior) and whether entire rows or columns are failing, with
# labels for the fold line and parent column 1 (common failure points).
#
# TWO LEVELS OF COMPARISON:
#   1. Interior comparison: reconstructed global field vs serial (rank 0 only)
#   2. Local halo comparison: each rank's full local data vs serial parent array
#      This catches per-rank halo fill bugs that interior comparison misses.
# ============================================================================

using MPI
MPI.Init()

using Oceananigans
using Oceananigans.Grids: halo_size, topology
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: set!, interior
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: reconstruct_global_field, reconstruct_global_grid,
                                            global_index_offset

const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const nranks = MPI.Comm_size(MPI.COMM_WORLD)

function print_rank0(args...)
    rank == 0 && println(args...)
    return nothing
end

function print_methodology()
    print_rank0("""

    INDEX-TRACING HALO FILL TEST
    ============================
    Each cell is filled with its own global index (field_i = x-index, field_j = y-index).
    After halo fill, the VALUE in each cell tells you WHERE the data came from.

    How to read mismatches:
      WRONG SOURCE  = cell got data from the wrong global position (fold bug or wrong MPI source)
      NEVER WRITTEN = cell is still zero (MPI buffer empty, recv skipped, or fold didn't write it)

    Structural labels:
      fold line     = the y-row at the fold boundary (Ny for UPivot CC/FC, FPivot CF/FF)
      fold halo N   = the Nth row above the fold line
      parent col 1  = first column of parent array (west halo col 1, common FC/FF failure point)
      west/east halo = x-halo columns, NW/NE corner = x-halo AND north halo overlap
    """)
end

# Fill a field with its interior index values along dimension `dim`
# offset shifts values so distributed fields match global serial indices
function fill_index_field!(field, dim; offset=0)
    int = interior(field)
    for I in CartesianIndices(int)
        int[I] = Float64(I.I[dim] + offset)
    end
    return nothing
end

# Classify a local index position into a human-readable region name
function classify_region(li, lj, Nx, Ny)
    in_west  = li < 1
    in_east  = li > Nx
    in_north = lj > Ny
    in_south = lj < 1

    if in_north && in_west
        return "NW corner"
    elseif in_north && in_east
        return "NE corner"
    elseif in_south && in_west
        return "SW corner"
    elseif in_south && in_east
        return "SE corner"
    elseif in_north
        return "north halo"
    elseif in_south
        return "south halo"
    elseif in_west
        return "west halo"
    elseif in_east
        return "east halo"
    else
        return "interior"
    end
end

# Label a local y-index relative to fold structure
function label_y(lj, Ny, Hy, fold_topology)
    if fold_topology == RightCenterFolded
        # UPivot: fold line at y=Ny for CC/FC
        if lj == Ny
            return "y=$lj (FOLD LINE for CC/FC)"
        elseif lj > Ny
            return "y=$lj (fold halo $(lj - Ny))"
        end
    else  # RightFaceFolded
        # FPivot: fold line at y=Ny for CF/FF, extra row at y=Ny for CC
        if lj == Ny
            return "y=$lj (FOLD LINE for CF/FF)"
        elseif lj > Ny
            return "y=$lj (fold halo $(lj - Ny))"
        end
    end
    return "y=$lj"
end

# Label a local x-index
function label_x(li, Nx)
    if li == 1 - size_parent_offset(Nx)
        return "x=$li (parent col 1)"
    elseif li < 1
        return "x=$li (west halo)"
    elseif li > Nx
        return "x=$li (east halo)"
    else
        return "x=$li"
    end
end

# Parent col 1 corresponds to local offset index 1-Hx, but we don't know Hx here.
# Use a simpler approach: just label the extremes.
size_parent_offset(Nx) = 0  # placeholder, not used

# Format a single mismatch as plain English
function format_mismatch(li, lj, spi, spj, loc_i, loc_j, ser_i, ser_j)
    if loc_i == 0 && loc_j == 0
        return "local($li,$lj) [= global($spi,$spj)]: " *
               "NEVER WRITTEN (zero) - expected data from global($ser_i,$ser_j)"
    else
        return "local($li,$lj) [= global($spi,$spj)]: " *
               "WRONG SOURCE - got data from global($loc_i,$loc_j), expected from global($ser_i,$ser_j)"
    end
end

# Analyze and report structural patterns in mismatches
function analyze_mismatches(mismatch_list, Nx, Ny, Hy, fold_topology)
    # mismatch_list: Vector of (li, lj, spi, spj, loc_i, loc_j, ser_i, ser_j)
    isempty(mismatch_list) && return String[]

    lines = String[]

    # Count by region
    region_counts = Dict{String, Int}()
    for (li, lj, _, _, _, _, _, _) in mismatch_list
        region = classify_region(li, lj, Nx, Ny)
        region_counts[region] = get(region_counts, region, 0) + 1
    end
    region_str = join(["$r ($n)" for (r, n) in sort(collect(region_counts), by=x->-x[2])], ", ")
    push!(lines, "Affected regions: $region_str")

    # Count mismatches per y-row and check if entire rows are affected
    y_counts = Dict{Int, Int}()
    y_expected = Dict{Int, Int}()  # how many x-positions exist for this y
    all_li = Set{Int}()
    for (li, lj, _, _, _, _, _, _) in mismatch_list
        y_counts[lj] = get(y_counts, lj, 0) + 1
        push!(all_li, li)
    end

    # Determine total x-positions checked (from the range of li values in mismatches)
    # For "entire row", we check if all x-positions that appear in mismatches for ANY row
    # also appear for THIS row. This is approximate but useful.
    li_range = sort(collect(all_li))

    affected_rows = String[]
    for lj in sort(collect(keys(y_counts)))
        count = y_counts[lj]
        label = label_y(lj, Ny, Hy, fold_topology)
        if count >= length(li_range) && length(li_range) > 1
            push!(affected_rows, "ENTIRE $label ($count cells)")
        else
            push!(affected_rows, "$label ($count cells)")
        end
    end
    if !isempty(affected_rows)
        push!(lines, "Affected rows: " * join(affected_rows, ", "))
    end

    # Count mismatches per x-column
    x_counts = Dict{Int, Int}()
    all_lj = Set{Int}()
    for (li, lj, _, _, _, _, _, _) in mismatch_list
        x_counts[li] = get(x_counts, li, 0) + 1
        push!(all_lj, li)
    end
    lj_range = sort(collect(all_lj))

    affected_cols = String[]
    for li in sort(collect(keys(x_counts)))
        count = x_counts[li]
        if li < 1
            desc = "x=$li (west halo)"
        elseif li > Nx
            desc = "x=$li (east halo)"
        elseif li == 1
            desc = "x=$li (parent col 1 interior)"
        elseif li == Nx + 1
            desc = "x=$li (parent col Nx+1)"
        else
            desc = "x=$li"
        end
        push!(affected_cols, "$desc ($count cells)")
    end
    if length(affected_cols) <= 20
        push!(lines, "Affected cols: " * join(affected_cols, ", "))
    else
        push!(lines, "Affected cols: $(length(affected_cols)) columns (too many to list)")
    end

    # Categorize by error type
    n_zero = count(m -> m[5] == 0 && m[6] == 0, mismatch_list)
    n_wrong = length(mismatch_list) - n_zero
    if n_zero > 0 && n_wrong > 0
        push!(lines, "Error types: $n_zero NEVER WRITTEN, $n_wrong WRONG SOURCE")
    elseif n_zero > 0
        push!(lines, "Error type: ALL $n_zero cells NEVER WRITTEN (zero)")
    else
        push!(lines, "Error type: ALL $n_wrong cells WRONG SOURCE")
    end

    # First N mismatches in plain English
    push!(lines, "First mismatches:")
    for (i, m) in enumerate(mismatch_list)
        i > 20 && break
        li, lj, spi, spj, loc_i, loc_j, ser_i, ser_j = m
        push!(lines, "  " * format_mismatch(li, lj, spi, spj, loc_i, loc_j, ser_i, ser_j))
    end

    return lines
end

# Compare two arrays element-wise; report mismatches with decoded meaning
function compare_arrays(name, a, b; atol=0)
    mismatches = 0
    first_mismatches = String[]
    for j in axes(a, 2), i in axes(a, 1)
        if abs(a[i, j] - b[i, j]) > atol
            mismatches += 1
            ai, bi = Int(a[i,j]), Int(b[i,j])
            if length(first_mismatches) < 10
                if bi == 0
                    push!(first_mismatches, "  [$i,$j]: NEVER WRITTEN (zero), expected $ai")
                else
                    push!(first_mismatches, "  [$i,$j]: got $bi, expected $ai (WRONG SOURCE)")
                end
            end
        end
    end
    return mismatches, first_mismatches
end

# Test one configuration: partition x fold_topology
function test_halo_fill(; partition, fold_topology, global_Nx, global_Ny)
    fold_name = fold_topology == RightCenterFolded ? "UPivot" : "FPivot"
    print_rank0("\n", "="^70)
    print_rank0("Testing $fold_name with partition=$partition, size=($global_Nx, $global_Ny, 1)")
    print_rank0("="^70)

    Hx, Hy, Hz = 5, 5, 5
    halo = (Hx, Hy, Hz)

    # --- Serial grid and fields (rank 0 only) ---
    serial_fields = Dict{String, Any}()
    if rank == 0
        serial_grid = TripolarGrid(; size=(global_Nx, global_Ny, 1), z=(-1000, 0), halo, fold_topology)

        locations = [
            ("CC", (Center(), Center(), Center())),
            ("FC", (Face(),   Center(), Center())),
            ("CF", (Center(), Face(),   Center())),
            ("FF", (Face(),   Face(),   Center())),
        ]

        for (loc_name, loc) in locations
            fi = Field(loc, serial_grid)
            fj = Field(loc, serial_grid)
            fill_index_field!(fi, 1)
            fill_index_field!(fj, 2)
            fill_halo_regions!((fi, fj))
            serial_fields["$(loc_name)_i"] = fi
            serial_fields["$(loc_name)_j"] = fj
        end
        print_rank0("Serial fields created and halos filled.")
    end

    MPI.Barrier(MPI.COMM_WORLD)

    # --- Distributed grid and fields ---
    arch = Distributed(CPU(); partition)
    dist_grid = TripolarGrid(arch; size=(global_Nx, global_Ny, 1), z=(-1000, 0), halo, fold_topology)

    locations = [
        ("CC", (Center(), Center(), Center())),
        ("FC", (Face(),   Center(), Center())),
        ("CF", (Center(), Face(),   Center())),
        ("FF", (Face(),   Face(),   Center())),
    ]

    all_pass = true
    Nx, Ny, _ = size(dist_grid)

    # Compute global offsets for this rank using generic function
    x_offset, y_offset, _ = global_index_offset(arch, (global_Nx, global_Ny, 1))

    for (loc_name, loc) in locations
        di = Field(loc, dist_grid)
        dj = Field(loc, dist_grid)
        fill_index_field!(di, 1; offset=x_offset)
        fill_index_field!(dj, 2; offset=y_offset)
        fill_halo_regions!((di, dj))

        # Reconstruct global fields from distributed
        gi = reconstruct_global_field(di)
        gj = reconstruct_global_field(dj)

        if rank == 0
            si = serial_fields["$(loc_name)_i"]
            sj = serial_fields["$(loc_name)_j"]

            # Compare only interior (reconstructed global doesn't have halos filled)
            s_int_i = Int.(interior(si, :, :, 1))
            s_int_j = Int.(interior(sj, :, :, 1))
            g_int_i = Int.(interior(gi, :, :, 1))
            g_int_j = Int.(interior(gj, :, :, 1))

            mis_i, details_i = compare_arrays("$(loc_name)_i interior", s_int_i, g_int_i)
            mis_j, details_j = compare_arrays("$(loc_name)_j interior", s_int_j, g_int_j)

            if mis_i == 0 && mis_j == 0
                print_rank0("  $loc_name interior: PASS (i and j match)")
            else
                all_pass = false
                print_rank0("  $loc_name interior: FAIL ($mis_i i-mismatches, $mis_j j-mismatches)")
                for d in details_i
                    print_rank0("    i: ", d)
                end
                for d in details_j
                    print_rank0("    j: ", d)
                end
            end
        end

        # --- Local field halo check ---
        # Each rank checks its local field halos against the serial field
        # Broadcast serial parent data to all ranks
        if rank == 0
            si = serial_fields["$(loc_name)_i"]
            sj = serial_fields["$(loc_name)_j"]
            s_full_i = parent(si.data)[:, :, 1+Hz]  # 1-based parent array, k=1 slice
            s_full_j = parent(sj.data)[:, :, 1+Hz]
        else
            s_full_i = nothing
            s_full_j = nothing
        end

        s_full_i = MPI.bcast(s_full_i, 0, MPI.COMM_WORLD)
        s_full_j = MPI.bcast(s_full_j, 0, MPI.COMM_WORLD)

        # Compare local field data (including halos) with serial
        local_i = di.data[:, :, 1]
        local_j = dj.data[:, :, 1]

        local_mismatches = 0
        # Store full mismatch tuples for structural analysis
        mismatch_list = Tuple{Int,Int,Int,Int,Int,Int,Int,Int}[]

        for lj in axes(local_i, 2), li in axes(local_i, 1)
            # Map local offset index to serial parent (1-based) index
            spi = li + x_offset + Hx
            spj = lj + y_offset + Hy

            # Check bounds in serial parent array
            if spi in axes(s_full_i, 1) && spj in axes(s_full_i, 2)
                loc_iv = Int(local_i[li, lj])
                loc_jv = Int(local_j[li, lj])
                ser_iv = Int(s_full_i[spi, spj])
                ser_jv = Int(s_full_j[spi, spj])
                if loc_iv != ser_iv || loc_jv != ser_jv
                    local_mismatches += 1
                    if length(mismatch_list) < 200  # keep enough for structural analysis
                        push!(mismatch_list, (li, lj, spi, spj, loc_iv, loc_jv, ser_iv, ser_jv))
                    end
                end
            end
        end

        # Gather mismatch counts
        all_local_mis = MPI.Gather(local_mismatches, 0, MPI.COMM_WORLD)

        if rank == 0
            total_local_mis = sum(all_local_mis)
            if total_local_mis == 0
                print_rank0("  $loc_name local halos: PASS (all ranks match serial)")
            else
                all_pass = false
                per_rank = join(["r$r=$(all_local_mis[r+1])" for r in 0:nranks-1 if all_local_mis[r+1] > 0], ", ")
                print_rank0("  $loc_name local halos: FAIL ($total_local_mis total: $per_rank)")
            end
        end

        # Print per-rank structural analysis sequentially
        for r in 0:(nranks - 1)
            if rank == r && !isempty(mismatch_list)
                println("    Rank $r ($local_mismatches mismatches):")
                analysis = analyze_mismatches(mismatch_list, Nx, Ny, Hy, fold_topology)
                for line in analysis
                    println("      ", line)
                end
            end
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)
    return all_pass
end

# ============================================================================
# Run test matrix
# ============================================================================

print_methodology()

partitions_4 = [Partition(1, 4), Partition(2, 2)]
partitions_8 = [Partition(4, 2)]
partitions_16 = [Partition(4, 4)]

results = Dict{String, Bool}()

# Test multiple Ny values per topology to stress remainder handling:
#   Ny=40 UPivot: 10,10,10,10 (even) / 20,20 (even)
#   Ny=43 UPivot: 10,10,10,13 (remainder 3) / 21,22 (remainder 1)
#   Ny=41 FPivot: 10,10,10,11 (remainder 1) / 20,21 (remainder 1)
#   Ny=42 FPivot: 10,10,10,12 (remainder 2) / 21,21 (even)

test_cases_4 = [
    (RightCenterFolded, 40, 40),  # UPivot, even
    (RightCenterFolded, 40, 43),  # UPivot, remainder
    (RightFaceFolded,   40, 41),  # FPivot, remainder
    (RightFaceFolded,   40, 42),  # FPivot, different remainder
]

for (fold_topology, global_Nx, global_Ny) in test_cases_4
    fold_name = fold_topology == RightCenterFolded ? "UPivot" : "FPivot"

    if nranks == 4
        for partition in partitions_4
            key = "$fold_name Nx=$global_Nx Ny=$global_Ny $(partition)"
            results[key] = test_halo_fill(; partition, fold_topology, global_Nx, global_Ny)
        end
    end

    if nranks == 16
        for partition in partitions_16
            key = "$fold_name Nx=$global_Nx Ny=$global_Ny $(partition)"
            results[key] = test_halo_fill(; partition, fold_topology, global_Nx, global_Ny)
        end
    end
end

# Partition(4,2) tests: match the exact grids from failing simulation tests.
test_cases_8 = [
    (RightCenterFolded, 80, 80),  # UPivot, matches testset 5 cfg3
    (RightFaceFolded,   80, 81),  # FPivot, matches testset 6 cfg3
]

if nranks == 8
    for (fold_topology, global_Nx, global_Ny) in test_cases_8
        fold_name = fold_topology == RightCenterFolded ? "UPivot" : "FPivot"
        for partition in partitions_8
            key = "$fold_name Nx=$global_Nx Ny=$global_Ny $(partition)"
            results[key] = test_halo_fill(; partition, fold_topology, global_Nx, global_Ny)
        end
    end
end

# ============================================================================
# Summary
# ============================================================================

if rank == 0
    println("\n", "="^70)
    println("SUMMARY")
    println("="^70)
    for (key, pass) in sort(collect(results))
        status = pass ? "PASS" : "FAIL"
        println("  $key: $status")
    end
    n_pass = count(values(results))
    n_total = length(results)
    println("\n  $n_pass / $n_total configurations passed")
    println("="^70)
end

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
