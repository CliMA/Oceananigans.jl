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

@inline get_grid_spacing(z::Function, k) = z(k)
@inline get_grid_spacing(z::AbstractArray{T,1}) where T = @inbounds z[k]

function generate_variable_grid_spacings_from_zF(zF_source, Nz)
    zF  = zeros(Nz+1)
    zC  = zeros(Nz)
    ΔzF = zeros(Nz)
    ΔzC = zeros(Nz-1)

    for k in 1:Nz+1
        zF[k] = get_grid_spacing(zF_source, k)
    end

    for k in 1:Nz
        zC[k] = (zF[k] + zF[k+1]) / 2
        ΔzF[k] = zF[k+1] - zF[k]
    end

    for k in 1:Nz-1
        ΔzC[k] = zC[k+1] - zC[k]
    end

    return zF, zC, ΔzF, ΔzC
end

function validate_and_generate_variable_grid_spacing(zF_source, Nz, z₁, z₂)
    zF, zC, ΔzF, ΔzC = generate_variable_grid_spacings_from_zF(zF_source, Nz)

    !isapprox(zF[1],   z₁) && throw(ArgumentError("Bottom face zF[1]=$(zF[1]) must equal bottom endpoint z₁=$z₁"))
    !isapprox(zF[end], z₂) && throw(ArgumentError("Top face zF[end]=$(zF[end]) must equal top endpoint z₂=$z₂"))

    return zF, zC, ΔzF, ΔzC
end
