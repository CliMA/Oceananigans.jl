@inline field_argument(i, j, k, grid, model_fields, ℑ, ::Val{n}) where n = @inbounds ℑ(i, j, k, grid, model_fields[n])
@inline field_argument(i, j, k, grid, model_fields, ℑ, n::Integer) = @inbounds ℑ(i, j, k, grid, model_fields[n])

@inline field_arguments(i, j, k, grid, model_fields, ℑ, idx::NTuple{1, Any}) =
    @inbounds (field_argument(i, j, k, grid, model_fields, ℑ[1], idx[1]),)

@inline field_arguments(i, j, k, grid, model_fields, ℑ, idx::NTuple{2, Any}) =
    @inbounds (field_argument(i, j, k, grid, model_fields, ℑ[1], idx[1]),
               field_argument(i, j, k, grid, model_fields, ℑ[2], idx[2]))

@inline field_arguments(i, j, k, grid, model_fields, ℑ, idx::NTuple{3, Any}) =
    @inbounds (field_argument(i, j, k, grid, model_fields, ℑ[1], idx[1]),
               field_argument(i, j, k, grid, model_fields, ℑ[2], idx[2]),
               field_argument(i, j, k, grid, model_fields, ℑ[3], idx[3]))

@inline function field_arguments(i, j, k, grid, model_fields, ℑ, idx::NTuple{N, Any}) where N
    f = ntuple(Val(N)) do n
        Base.@_inline_meta
        @inbounds field_argument(i, j, k, grid, model_fields, ℑ[n], idx[n])
    end
    return f
end

""" Returns field arguments in user-defined functions for forcing and boundary conditions."""
@inline function user_function_arguments(i, j, k, grid, model_fields, ::Nothing, user_func)

    ℑ = user_func.field_dependencies_interp
    idx = user_func.field_dependencies_indices
    return field_arguments(i, j, k, grid, model_fields, ℑ, idx)
end

""" Returns field arguments plus parameters in user-defined functions for forcing and boundary conditions."""
@inline function user_function_arguments(i, j, k, grid, model_fields, parameters, user_func)

    ℑ = user_func.field_dependencies_interp
    idx = user_func.field_dependencies_indices
    parameters = user_func.parameters

    field_args = field_arguments(i, j, k, grid, model_fields, ℑ, idx)

    return tuple(field_args..., parameters)
end
