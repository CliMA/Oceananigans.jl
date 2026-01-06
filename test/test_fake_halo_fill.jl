# Test script for fake halo fill kernels with Reactant
# This tests pure KernelAbstractions + OffsetArrays without Oceananigans

using CUDA
using KernelAbstractions
using OffsetArrays
using Random
using Reactant

# Include the fake halo fill kernels
include("fake_halo_fill_kernels.jl")

# Set Reactant backend
Reactant.set_default_backend("cpu")

#####
##### Test functions
#####

"""
Run a single fake halo fill test for a given topology combination.
Returns (passed::Bool, error_message::String)
"""
function test_fake_halo_fill(Nx, Ny, Nz, topo_x, topo_y, topo_z; raise=true)
    # Create vanilla (CPU) array
    vanilla_c = create_offset_array(Nx, Ny, Nz)
    
    # Create Reactant array (must use ConcreteRArray)
    reactant_data = zeros(Float64, Nx+2, Ny+2, Nz+2)
    reactant_raw = Reactant.to_rarray(reactant_data)
    reactant_c = OffsetArray(reactant_raw, 0:(Nx+1), 0:(Ny+1), 0:(Nz+1))
    
    # Set random interior data
    Random.seed!(12345)
    interior_data = randn(Nx, Ny, Nz)
    set_interior!(vanilla_c, interior_data, Nx, Ny, Nz)
    
    # Set the same data in reactant array
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        reactant_raw[i, j, k] = interior_data[i, j, k]
    end
    
    # Fill halos on vanilla array
    fake_fill_halo_regions!(vanilla_c, Nx, Ny, Nz, topo_x, topo_y, topo_z, CPU())
    
    # Fill halos on reactant array using @jit
    function do_fake_fill!(c, Nx, Ny, Nz, topo_x, topo_y, topo_z)
        fake_fill_halo_regions!(c, Nx, Ny, Nz, topo_x, topo_y, topo_z, CPU())
        return nothing
    end
    
    try
        @jit raise=raise do_fake_fill!(reactant_c, Nx, Ny, Nz, topo_x, topo_y, topo_z)
    catch e
        err_msg = sprint(showerror, e)
        err_lines = split(err_msg, "\n")
        return (false, first(err_lines))
    end
    
    # Compare results
    if compare_arrays("halo", vanilla_c, reactant_c)
        return (true, "")
    else
        return (false, "Comparison mismatch")
    end
end

"""
Wrapper function that can be JIT'd for fake halo filling.
"""
function jit_fake_fill_halo_regions!(c, Nx, Ny, Nz, topo_x, topo_y, topo_z)
    fake_fill_halo_regions!(c, Nx, Ny, Nz, topo_x, topo_y, topo_z, CPU())
    return nothing
end

#####
##### Main test loop
#####

function run_fake_halo_tests(; raise=true)
    Nx, Ny, Nz = 4, 5, 3
    
    topologies = [:periodic, :bounded]
    
    results = Tuple{String, Bool, String}[]
    
    println("=" ^ 80)
    println("Testing FAKE halo fill kernels with raise=$raise")
    println("Pure KernelAbstractions + OffsetArrays (no Oceananigans)")
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
            # Set random interior data (same seed for both)
            Random.seed!(12345)
            
            # Create vanilla (CPU) array with data already set
            vanilla_data = zeros(Float64, Nx+2, Ny+2, Nz+2)
            for k in 1:Nz, j in 1:Ny, i in 1:Nx
                vanilla_data[i, j, k] = randn()
            end
            vanilla_c = OffsetArray(vanilla_data, 0:(Nx+1), 0:(Ny+1), 0:(Nz+1))
            
            # Create Reactant array with same data (set BEFORE converting to RArray)
            Random.seed!(12345)  # Reset seed to get same random numbers
            reactant_data = zeros(Float64, Nx+2, Ny+2, Nz+2)
            for k in 1:Nz, j in 1:Ny, i in 1:Nx
                reactant_data[i, j, k] = randn()
            end
            reactant_raw = Reactant.to_rarray(reactant_data)
            reactant_c = OffsetArray(reactant_raw, 0:(Nx+1), 0:(Ny+1), 0:(Nz+1))
            
            # Fill halos on vanilla array
            fake_fill_halo_regions!(vanilla_c, Nx, Ny, Nz, topo_x, topo_y, topo_z, CPU())
            
            # Fill halos on reactant array using @jit
            @jit raise=raise jit_fake_fill_halo_regions!(reactant_c, Nx, Ny, Nz, topo_x, topo_y, topo_z)
            
            # Compare results
            if compare_arrays("halo", vanilla_c, reactant_c)
                push!(results, (test_name, true, ""))
                println("✓ PASSED")
            else
                push!(results, (test_name, false, "Comparison mismatch"))
                println("✗ MISMATCH")
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

#####
##### Run tests if executed as a script
#####

if abspath(PROGRAM_FILE) == @__FILE__
    run_fake_halo_tests(raise=true)
end

