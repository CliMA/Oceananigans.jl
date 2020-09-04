"""
    struct AveragedField{X, Y, Z, A, G, I, N} <: AbstractReducedField{X, Y, Z, A, G, N}

Type representing an average over a field-like object.
"""
struct AveragedField{X, Y, Z, A, G, I, N} <: AbstractReducedField{X, Y, Z, A, G, N}
         data :: A
         grid :: G
         dims :: NTuple{N, Int}
    integrand :: I

    function AveragedField{X, Y, Z}(data, grid, dims, integrand) where {X, Y, Z}

        # Check
        dims isa Union{Int, Tuple} || error("Average dims must be an integer or tuple!")
        dims isa Int && (dims = tuple(dims))

        length(dims) == 0 && error("dims is empty! Must average over at least one dimension.")
        length(dims) > 3  && error("Models are 3-dimensional. Cannot average over 4+ dimensions.")
        all(1 <= d <= 3 for d in dims) || error("Dimensions must be one of 1, 2, 3.")

        Tx, Ty, Tz = total_size((X, Y, Z), grid)

        if size(data) != (Tx, Ty, Tz)
            e = "Cannot construct field at ($X, $Y, $Z) with size(data)=$(size(data)). " *
                "`data` must have size ($Tx, $Ty, $Tz)."
            throw(ArgumentError(e))
        end

        return new{X, Y, Z, typeof(data),
                   typeof(grid), typeof(dims), typeof(integrand)}(data, grid, dims, integrand)
end

reduced_location(loc, dims) = Tuple(i ∈ dims ? Nothing : loc[i] for i = 1:3)

function AveragedField(integrand; dims, data=nothing)
    
    arch = architecture(integrand)
    loc = reduced_location(location(integrand), dims)
    Tx, Ty, Tx = total_size(loc, grid)

    if isnothing(data)
        data = new_data(arch, grid(integrand), loc)
    else
        total_size(data) === (Tx, Ty, Tz) || error("`data` must have size ($Tx, $Ty, $Tz).")
    end

    return AveragedField(data, grid, dims, integrand)
end


