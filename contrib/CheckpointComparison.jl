module CheckpointComparison

using JLD2
using Oceananigans.TimeSteppers: Clock
using Oceananigans.BoundaryConditions: BoundaryCondition

export compare_all

function compare_all(a::T, b::T; name="", verbose=false) where T
    success = true

    if a isa JLD2.Group
        a_keys = keys(a)
        b_keys = keys(b)

        @assert issetequal(a_keys, b_keys) "a_keys=$(a_keys), b_keys=$(b_keys)"

        # recurse
        for key in a_keys
            fullkey = !isempty(name) ? string(name, ".", key) : key
            if isnothing(a[key])
                @assert isnothing(b[key])
                verbose && println("Note: No $fullkey")
            else
                success &= compare_all(a[key], b[key]; name=fullkey, verbose=verbose)
            end
        end

    elseif a isa Clock
        if a != b
            if a ≈ b
                println("$name are approximately equal")
            else
                println("$name DIFFER!?")
                success = false
            end
            println(a)
            println(b)
        elseif verbose
            println("$name are identical")
        end

    elseif a isa AbstractFloat
        if a != b
            if a ≈ b
                println("$name are approximately equal")
            else
                println("$name DIFFER!?")
                success = false
            end
            println("$a $b")
        elseif verbose
            println("$name are identical")
        end

    elseif endswith(name, "boundary_conditions")
        verbose && println("Skip comparison of ",name)
    else
        if a != b
            println("Compare arrays $name")
            abs_diff = @. abs(a - b)
            rel_diff = [max(abs(aval), abs(bval)) == 0 ? 0.0 : abs(aval - bval) / max(abs(aval), abs(bval))
                        for (aval, bval) in zip(a, b)]

            if all(a .≈ b)
                println("$name are approximately equal ", maximum(abs_diff), " ", maximum(rel_diff))
            else
                println("$name DIFFER!? max abs/rel diff : ", maximum(abs_diff), " ", maximum(rel_diff))
                success = false
            end

            max_idx = argmax(abs_diff)
            i, j, k = Tuple(max_idx)
            println("  location of max abs diff: ", (i, j, k))

            max_idx = argmax(rel_diff)
            i, j, k = Tuple(max_idx)
            println("  location of max rel diff: ", (i, j, k))

        elseif verbose
            println("$name are identical")
        end
    end

    return success
end

"""(If only Comonicon worked)

Recursively compare data structures inside a checkpointed model instance

# Args

- `filepath1`: First .jld2 checkpoint file
- `filepath2`: Second .jld2 checkpoint file

# Flags

- `-v, --verbose`: Print extra information
"""
function main()
    # Defaults
    verbose = false
    positional_args = String[]

    # Parse the built-in ARGS array
    for arg in ARGS
        if arg in ("-v", "--verbose")
            verbose = true
        elseif startswith(arg, "-")
            println(stderr, "Error: Unknown flag '$arg'")
            exit(1)
        else
            push!(positional_args, arg)
        end
    end

    # Validate we got exactly the files we need
    if length(positional_args) != 2
        println(stderr, "Usage: julia compare_checkpoints.jl <filepath1> <filepath2> [-v|--verbose]")
        exit(1)
    end
    filepath1 = positional_args[1]
    filepath2 = positional_args[2]

    # Check filepaths
    filepath1 = abspath(expanduser(filepath1))
    !isfile(filepath1) && throw(ArgumentError("File not found: $filepath1"))
    filepath2 = abspath(expanduser(filepath2))
    !isfile(filepath2) && throw(ArgumentError("File not found: $filepath2"))

    # Load and compare
    chk1 = jldopen(filepath1, "r")
    chk2 = jldopen(filepath2, "r")

    compare_all(chk1["simulation"]["model"],
                chk2["simulation"]["model"];
                verbose)

    close(chk1)
    close(chk2)
end

end # module CheckpointComparison

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    CheckpointComparison.main()
end
