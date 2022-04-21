using Random
using Oceananigans.Utils: instantiate
using Oceananigans.Grids: Face, Center

# Utilities for inferring the interpolation function needed to
# interpolate a field from one location to the next.
interpolation_code(from, to) = interpolation_code(to)

interpolation_code(::Type{Face}) = :ᶠ
interpolation_code(::Type{Center}) = :ᶜ
interpolation_code(::Type{Nothing}) = :ᶜ
interpolation_code(::Face) = :ᶠ
interpolation_code(::Center) = :ᶜ
interpolation_code(::Nothing) = :ᶜ

# Intercept non-interpolations
interpolation_code(from::L, to::L) where L = :ᵃ
interpolation_code(::Nothing, to) = :ᵃ
interpolation_code(from, ::Nothing) = :ᵃ
interpolation_code(::Nothing, ::Nothing) = :ᵃ

for ξ in ("x", "y", "z")
    ▶sym = Symbol(:ℑ, ξ, :sym)
    @eval begin
        $▶sym(s::Symbol) = $▶sym(Val(s))
        $▶sym(::Union{Val{:ᶠ}, Val{:ᶜ}}) = $ξ
        $▶sym(::Val{:ᵃ}) = ""
    end
end

# It's not Oceananigans for nothing
const number_of_identities = 6 # hopefully enough for Oceananigans (most need just one)

for i = 1:number_of_identities
    identity = Symbol(:identity, i)

    @eval begin
        @inline $identity(i, j, k, grid, c) = @inbounds c[i, j, k]
        @inline $identity(i, j, k, grid, a::Number) = a
        @inline $identity(i, j, k, grid, F::TF, args...) where TF<:Function = F(i, j, k, grid, args...)
    end
end
 
torus(x, lower, upper) = lower + rem(x - lower, upper - lower, RoundDown)
identify_an_identity(number) = Symbol(:identity, torus(number, 1, number_of_identities))
identity_counter = 0

"""
    interpolation_operator(from, to)

Returns the function to interpolate a field `from = (XA, YZ, ZA)`, `to = (XB, YB, ZB)`,
where the `XA`s and `XB`s are `Face()` or `Center()` instances.
"""
function interpolation_operator(from, to)
    from, to = instantiate.(from), instantiate.(to)
    x, y, z = (interpolation_code(X, Y) for (X, Y) in zip(from, to))

    if all(ξ === :ᵃ for ξ in (x, y, z))

        # This is crazy, but here's my number...
        global identity_counter += 1
        identity = identify_an_identity(identity_counter)

        return @eval $identity
    else
        return eval(Symbol(:ℑ, ℑxsym(x), ℑysym(y), ℑzsym(z), x, y, z))
    end
end

"""
    interpolation_operator(::Nothing, to)

Return the `identity` interpolator function. This is needed to obtain the interpolation
operator for fields that have no intrinsic location, like numbers or functions.
"""
function interpolation_operator(::Nothing, to)
    global identity_counter += 1
    identity = identify_an_identity(identity_counter)
    return @eval $identity
end

assumed_field_location(name) = name === :u  ? (Face, Center, Center) :
                               name === :v  ? (Center, Face, Center) :
                               name === :w  ? (Center, Center, Face) :
                               name === :uh ? (Face, Center, Center) :
                               name === :vh ? (Center, Face, Center) :
                                              (Center, Center, Center)

"""
    index_and_interp_dependencies(X, Y, Z, dependencies, model_field_names)

Returns a tuple of indices and interpolation functions to the location `X, Y, Z`
for each name in `dependencies`.

The indices correspond to the position of each dependency within `model_field_names`.

The interpolation functions interpolate the dependent field to `X, Y, Z`.
"""
function index_and_interp_dependencies(X, Y, Z, dependencies, model_field_names)
    interps = Tuple(interpolation_operator(assumed_field_location(name), (X, Y, Z))
                    for name in dependencies)

    indices = ntuple(length(dependencies)) do i
        name = dependencies[i]
        findfirst(isequal(name), model_field_names)
    end

    return indices, interps
end
