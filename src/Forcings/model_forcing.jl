using Oceananigans.Fields: AbstractField

@inline zeroforcing(args...) = 0

"""
    materialize_forcing(forcing, field, field_name, model_field_names)

"Regularizes" or "adds information" to user-defined forcing objects that are passed to
model constructors. `materialize_forcing` is called inside `model_forcing`.

We need `materialize_forcing` because it is only until `model_forcing` is called that
the fields (and field locations) of various forcing functions are available. The `field`
can be used to infer the location at which the forcing is applied, or to add a field
dependency to a special forcing object, as for `Relxation`.
"""
materialize_forcing(forcing, field, field_name, model_field_names) = forcing # fallback

"""
    materialize_forcing(forcing::Function, field, field_name, model_field_names)

Wrap `forcing` in a `ContinuousForcing` at the location of `field`.
"""
function materialize_forcing(forcing::Function, field, field_name, model_field_names)
    LX, LY, LZ = location(field)
    return ContinuousForcing{LX, LY, LZ}(forcing)
end

materialize_forcing(::Nothing, field::AbstractField, field_name, model_field_names) = zeroforcing

# TODO: some checking that `array` is validly-sized could be done here
materialize_forcing(array::AbstractArray, field::AbstractField, field_name, model_field_names) = Forcing(array)
materialize_forcing(fts::FlavorOfFTS, field::AbstractField, field_name, model_field_names) = Forcing(fts)


"""
    model_forcing(user_forcings, model_fields, prognostic_fields=model_fields)

Return a `NamedTuple` of forcing functions for each field in `model_fields`, wrapping
forcing functions in `ContinuousForcing`s and ensuring that `ContinuousForcing`s are
located correctly for each field.
"""
function model_forcing(user_forcings, model_fields, prognostic_fields=model_fields)

    user_forcing_names = keys(user_forcings)

    for name in user_forcing_names
        if name ∉ keys(prognostic_fields)
            msg = string("Invalid forcing: forcing contains an entry for $name, but $name is not a prognostic field!", '\n',
                         "The prognostic fields are ", keys(prognostic_fields))
            throw(ArgumentError(msg))   
        end
    end

    model_field_names = keys(model_fields)

    materialized = Tuple(
        name in keys(user_forcings) ?
            materialize_forcing(user_forcings[name], field, name, model_field_names) :
            Returns(zero(eltype(field)))
            for (name, field) in pairs(prognostic_fields)
    )

    prognostic_names = keys(prognostic_fields)
    forcings = NamedTuple{prognostic_names}(materialized)

    return forcings
end
