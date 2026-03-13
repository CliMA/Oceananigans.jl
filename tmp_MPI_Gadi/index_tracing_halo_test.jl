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

# Fill a field with its interior index values along dimension `dim`
# offset shifts values so distributed fields match global serial indices
function fill_index_field!(field, dim; offset=0)
    int = interior(field)
    for I in CartesianIndices(int)
        int[I] = Float64(I.I[dim] + offset)
    end
    return nothing
end

# Compare two arrays element-wise; report mismatches with (i,j) location and values
function compare_arrays(name, a, b; atol=0)
    mismatches = 0
    first_mismatches = String[]
    for j in axes(a, 2), i in axes(a, 1)
        if abs(a[i, j] - b[i, j]) > atol
            mismatches += 1
            if length(first_mismatches) < 10
                push!(first_mismatches, "  [$i,$j]: serial=$(Int(a[i,j])) vs distributed=$(Int(b[i,j]))")
            end
        end
    end
    return mismatches, first_mismatches
end

# Test one configuration: partition × fold_topology
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

            # Compare full data arrays (interior + halos) of reconstructed global vs serial
            s_data_i = Int.(si[:, :, 1])
            s_data_j = Int.(sj[:, :, 1])
            g_data_i = Int.(gi[:, :, 1])
            g_data_j = Int.(gj[:, :, 1])

            # Compare only interior (reconstructed global doesn't have halos filled)
            Nx, Ny, _ = size(gi)
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
        # We need the serial field data broadcast to all ranks
        # Use parent() to get 1-based Array from OffsetArray
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
        # local_i/local_j are OffsetArrays with indices from 1-Hx to Nx+Hx
        local_i = di.data[:, :, 1]
        local_j = dj.data[:, :, 1]

        local_mismatches = 0
        local_details = String[]

        for lj in axes(local_i, 2), li in axes(local_i, 1)
            # Map local offset index to serial parent (1-based) index
            # local offset index li corresponds to global offset index li + x_offset
            # parent array index = global offset index + Hx (since parent[1] = global[-Hx+1])
            spi = li + x_offset + Hx
            spj = lj + y_offset + Hy

            # Check bounds in serial parent array
            if spi in axes(s_full_i, 1) && spj in axes(s_full_i, 2)
                if Int(local_i[li, lj]) != Int(s_full_i[spi, spj]) ||
                   Int(local_j[li, lj]) != Int(s_full_j[spi, spj])
                    local_mismatches += 1
                    if length(local_details) < 5
                        push!(local_details,
                            "  local[$li,$lj]→serial_parent[$spi,$spj]: " *
                            "local_i=$(Int(local_i[li,lj])) serial_i=$(Int(s_full_i[spi,spj])) " *
                            "local_j=$(Int(local_j[li,lj])) serial_j=$(Int(s_full_j[spi,spj]))")
                    end
                end
            end
        end

        # Gather mismatch counts (Int is isbitstype, works with MPI)
        all_local_mis = MPI.Gather(local_mismatches, 0, MPI.COMM_WORLD)

        if rank == 0
            total_local_mis = sum(all_local_mis)
            if total_local_mis == 0
                print_rank0("  $loc_name local halos: PASS (all ranks match serial)")
            else
                all_pass = false
                print_rank0("  $loc_name local halos: FAIL ($total_local_mis total mismatches)")
            end
        end

        # Print per-rank details sequentially (avoids MPI.Gather of String)
        for r in 0:(nranks - 1)
            if rank == r && !isempty(local_details)
                println("    Rank $r:")
                for d in local_details
                    println("      ", d)
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

    if nranks >= 4
        for partition in partitions_4
            key = "$fold_name Nx=$global_Nx Ny=$global_Ny $(partition)"
            results[key] = test_halo_fill(; partition, fold_topology, global_Nx, global_Ny)
        end
    end

    if nranks >= 16
        for partition in partitions_16
            key = "$fold_name Nx=$global_Nx Ny=$global_Ny $(partition)"
            results[key] = test_halo_fill(; partition, fold_topology, global_Nx, global_Ny)
        end
    end
end

# Partition(4,2) tests: match the exact grids from failing simulation tests.
# These test the _fold_corner_write! bug with Rx=4 where fold x-reversal
# maps corners to a DIFFERENT rank's data (not local).
test_cases_8 = [
    (RightCenterFolded, 80, 80),  # UPivot, matches testset 5 cfg3
    (RightFaceFolded,   80, 81),  # FPivot, matches testset 6 cfg3
]

if nranks >= 8
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
