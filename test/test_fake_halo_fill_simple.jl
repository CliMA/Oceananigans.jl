# Simplified test for fake halo fill kernels with Reactant
# Uses regular 1-based arrays (no OffsetArrays) to avoid potential issues

using CUDA
using KernelAbstractions
using Random
using Reactant

# Set Reactant backend
Reactant.set_default_backend("cpu")

#####
##### Simple halo fill kernels with 1-based indexing
##### Array is (Nx+2, Ny+2, Nz+2) with halos at indices 1 and N+2
##### Interior is 2:N+1
#####

# X-direction: Bounded (left halo copies from first interior, right from last interior)
@kernel function fill_x_halo_bounded!(c, Nx)
    j, k = @index(Global, NTuple)
    # Left halo (index 1) copies from first interior (index 2)
    @inbounds c[1, j, k] = c[2, j, k]
    # Right halo (index Nx+2) copies from last interior (index Nx+1)
    @inbounds c[Nx+2, j, k] = c[Nx+1, j, k]
end

# X-direction: Periodic (left halo copies from right interior, right from left interior)
@kernel function fill_x_halo_periodic!(c, Nx)
    j, k = @index(Global, NTuple)
    # Left halo (index 1) copies from right interior (index Nx+1)
    @inbounds c[1, j, k] = c[Nx+1, j, k]
    # Right halo (index Nx+2) copies from left interior (index 2)
    @inbounds c[Nx+2, j, k] = c[2, j, k]
end

# Y-direction: Bounded
@kernel function fill_y_halo_bounded!(c, Ny)
    i, k = @index(Global, NTuple)
    @inbounds c[i, 1, k] = c[i, 2, k]
    @inbounds c[i, Ny+2, k] = c[i, Ny+1, k]
end

# Y-direction: Periodic
@kernel function fill_y_halo_periodic!(c, Ny)
    i, k = @index(Global, NTuple)
    @inbounds c[i, 1, k] = c[i, Ny+1, k]
    @inbounds c[i, Ny+2, k] = c[i, 2, k]
end

# Z-direction: Bounded
@kernel function fill_z_halo_bounded!(c, Nz)
    i, j = @index(Global, NTuple)
    @inbounds c[i, j, 1] = c[i, j, 2]
    @inbounds c[i, j, Nz+2] = c[i, j, Nz+1]
end

# Z-direction: Periodic
@kernel function fill_z_halo_periodic!(c, Nz)
    i, j = @index(Global, NTuple)
    @inbounds c[i, j, 1] = c[i, j, Nz+1]
    @inbounds c[i, j, Nz+2] = c[i, j, 2]
end

#####
##### Combined fill function
#####

function simple_fill_halos!(c, Nx, Ny, Nz, topo_x, topo_y, topo_z)
    backend = KernelAbstractions.get_backend(c)
    
    # X-direction
    if topo_x == :bounded
        fill_x_halo_bounded!(backend)(c, Nx, ndrange=(Ny+2, Nz+2))
    else
        fill_x_halo_periodic!(backend)(c, Nx, ndrange=(Ny+2, Nz+2))
    end
    
    # Y-direction
    if topo_y == :bounded
        fill_y_halo_bounded!(backend)(c, Ny, ndrange=(Nx+2, Nz+2))
    else
        fill_y_halo_periodic!(backend)(c, Ny, ndrange=(Nx+2, Nz+2))
    end
    
    # Z-direction
    if topo_z == :bounded
        fill_z_halo_bounded!(backend)(c, Nz, ndrange=(Nx+2, Ny+2))
    else
        fill_z_halo_periodic!(backend)(c, Nz, ndrange=(Nx+2, Ny+2))
    end
    
    KernelAbstractions.synchronize(backend)
    return nothing
end

#####
##### Test runner
#####

function run_simple_halo_tests(; raise=true)
    Nx, Ny, Nz = 4, 5, 3
    
    topologies = [:periodic, :bounded]
    
    results = Tuple{String, Bool, String}[]
    
    println("=" ^ 80)
    println("Testing SIMPLE halo fill kernels with raise=$raise")
    println("Pure KernelAbstractions (1-based arrays, no OffsetArrays)")
    println("Julia ", VERSION)
    println("=" ^ 80)
    println()
    
    test_count = 0
    for topo_x in topologies, topo_y in topologies, topo_z in topologies
        test_count += 1
        test_name = "topo=($topo_x, $topo_y, $topo_z)"
        
        print("[$test_count/8] $test_name ... ")
        flush(stdout)
        
        try
            # Create vanilla (CPU) array with random interior data
            Random.seed!(12345)
            vanilla_c = zeros(Float64, Nx+2, Ny+2, Nz+2)
            for k in 2:(Nz+1), j in 2:(Ny+1), i in 2:(Nx+1)
                vanilla_c[i, j, k] = randn()
            end
            
            # Create Reactant array with same data
            Random.seed!(12345)
            reactant_data = zeros(Float64, Nx+2, Ny+2, Nz+2)
            for k in 2:(Nz+1), j in 2:(Ny+1), i in 2:(Nx+1)
                reactant_data[i, j, k] = randn()
            end
            reactant_c = Reactant.to_rarray(reactant_data)
            
            # Fill halos on vanilla array
            simple_fill_halos!(vanilla_c, Nx, Ny, Nz, topo_x, topo_y, topo_z)
            
            # Fill halos on reactant array using @jit
            @jit raise=raise simple_fill_halos!(reactant_c, Nx, Ny, Nz, topo_x, topo_y, topo_z)
            
            # Compare results (convert RArray to Array for comparison)
            reactant_result = Array(reactant_c)
            
            max_diff = maximum(abs.(vanilla_c .- reactant_result))
            
            if max_diff < 1e-10
                push!(results, (test_name, true, ""))
                println("✓ PASSED")
            else
                push!(results, (test_name, false, "max_diff = $max_diff"))
                println("✗ MISMATCH (max_diff = $max_diff)")
            end
        catch e
            err_type = string(typeof(e).name.name)
            err_msg = sprint(showerror, e)
            err_lines = split(err_msg, "\n")
            err_summary = length(err_lines) > 0 ? first(err_lines) : err_msg
            if length(err_summary) > 100
                err_summary = err_summary[1:100] * "..."
            end
            push!(results, (test_name, false, "$err_type: $err_summary"))
            println("✗ $err_type")
            println("  → $err_summary")
        end
    end
    
    # Summary
    println("\n" * "=" ^ 80)
    println("SUMMARY")
    println("=" ^ 80)
    
    passed = filter(r -> r[2], results)
    failed = filter(r -> !r[2], results)
    
    println("\nTotal: $(length(results))")
    println("Passed: $(length(passed))")
    println("Failed: $(length(failed))")
    
    if length(passed) > 0
        println("\n### Passed tests:")
        for (name, _, _) in passed
            println("  ✓ $name")
        end
    end
    
    if length(failed) > 0
        println("\n### Failed tests:")
        for (name, _, msg) in failed
            println("  ✗ $name")
            println("    $msg")
        end
    end
    
    return results
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    run_simple_halo_tests(raise=true)
end

