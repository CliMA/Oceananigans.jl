"""Validate that an argument tuple is the right length and has elements of type `argtype`."""
function validate_tupled_argument(arg, argtype, argname)
    length(arg) == 3        || throw(ArgumentError("length($argname) must be 3."))
    all(isa.(arg, argtype)) || throw(ArgumentError("$argname=$arg must contain $argtype s."))
    all(arg .> 0)           || throw(ArgumentError("Elements of $argname=$arg must be > 0!"))
    return nothing
end

function validate_grid_size_and_length(sz, len, halo, x, y, z)
    validate_tupled_argument(sz, Integer, "size")
    validate_tupled_argument(halo, Integer, "halo")

    # Find domain endpoints or domain length, depending on user input:
    if !isnothing(len) # the user has specified a length!
        (!isnothing(x) || !isnothing(y) || !isnothing(z)) &&
            throw(ArgumentError("Cannot specify both length and x, y, z keyword arguments."))

        validate_tupled_argument(len, Number, "length")

        Lx, Ly, Lz = len

        # An "oceanic" default domain
        x = (0, Lx)
        y = (0, Ly)
        z = (-Lz, 0)

    else # isnothing(length) === true implies that user has not specified a length
        (isnothing(x) || isnothing(y) || isnothing(z)) &&
            throw(ArgumentError("Must supply length or x, y, z keyword arguments."))

        function coord2xyz(c)
            c == 1 && return "x"
            c == 2 && return "y"
            c == 3 && return "z"
        end

        for (i, c) in enumerate((x, y, z))
            name = coord2xyz(i)
            length(c) == 2       || throw(ArgumentError("$name length($c) must be 2."))
            all(isa.(c, Number)) || throw(ArgumentError("$name=$c should contain numbers."))
            c[2] >= c[1]         || throw(ArgumentError("$name=$c should be an increasing interval."))
        end

        Lx = x[2] - x[1]
        Ly = y[2] - y[1]
        Lz = z[2] - z[1]
    end
    return Lx, Ly, Lz, x, y, z
end

function validate_variable_grid_spacing(zF, ΔzF, zC, ΔzC, z₁, z₂)
    if isnothing(zF) && isnothing(ΔzF) && isnothing(zC) && isnothing(ΔzC)
        throw(ArgumentError("Must supply a variable vertical grid spacing using " *
                            "one of the zF, ΔzF, zC, or ΔzC keyword arguments."))
    end

    if sum(isnothing.([zF, ΔzF, zC, ΔzC])) > 1
        throw(ArgumentError("Must only specify one of zF, ΔzF, zC, or ΔzC."))
    end

    if !isnothing(zF)
        zF[1]   != z₁ && throw(ArgumentError("First face zF[1]=$(zF[1]) must equal left endpoint z₁=$z₁"))
        zF[end] != z₂ && throw(ArgumentError("Last face zF[1]=$(zF[1]) must equal right endpoint z₁=$z₁"))
    end
end
