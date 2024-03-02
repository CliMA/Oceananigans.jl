module CubedSpheres

export ConformalCubedSphereGrid, face, faces, λnodes, φnodes

include("cubed_sphere_utils.jl")
include("conformal_cubed_sphere_grid.jl")
include("cubed_sphere_exchange_bcs.jl")
include("cubed_sphere_faces.jl")
include("cubed_sphere_halo_filling.jl")
include("cubed_sphere_kernel_launching.jl")
include("immersed_conformal_cubed_sphere_grid.jl")

#####
##### Validating cubed sphere stuff
#####

using Oceananigans.Utils
import Oceananigans.Grids: validate_index
import Oceananigans.Fields: validate_field_data, validate_boundary_conditions
import Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_vertical_velocity_boundary_conditions

function validate_field_data(loc, data, grid::ConformalCubedSphereGrid, indices)

    for (face_data, face_grid) in zip(data.faces, grid.faces)
        validate_field_data(loc, face_data, face_grid, indices)
    end

    return nothing
end


# We don't support validating cubed sphere boundary conditions at this time
validate_boundary_conditions(loc, grid::ConformalCubedSphereGrid, bcs::CubedSphereFaces) = nothing
validate_boundary_conditions(loc, grid::OrthogonalSphericalShellGrid, bcs) = nothing

validate_vertical_velocity_boundary_conditions(w::AbstractCubedSphereField) =
    [validate_vertical_velocity_boundary_conditions(w_face) for w_face in faces(w)]

#####
##### Regularizing field boundary conditions
#####

import Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

function regularize_field_boundary_conditions(bcs::CubedSphereFaces,
                                              grid::AbstractGrid,
                                              field_name::Symbol,
                                              prognostic_field_names)

    faces = Tuple(regularize_field_boundary_conditions(face_bcs, face_grid, field_name, prognostic_field_names)
                  for (face_bcs, face_grid) in zip(bcs.faces, grid.faces))

    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::ConformalCubedSphereGrid,
                                              field_name::Symbol,
                                              prognostic_field_names)

    faces = Tuple(
        inject_cubed_sphere_exchange_boundary_conditions(
            regularize_field_boundary_conditions(bcs, face_grid, field_name, prognostic_field_names),
            face_index,
            grid.face_connectivity
        )
        for (face_index, face_grid) in enumerate(grid.faces)
    )

    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

# This means we don't support fluxes through immersed boundaries on the cubed sphere
import Oceananigans.Fields: immersed_boundary_condition
immersed_boundary_condition(::AbstractCubedSphereField) = nothing

import Oceananigans.OutputWriters: construct_output, fetch_output

construct_output(user_output::AbstractCubedSphereField, args...) = user_output

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

function fetch_output(field::AbstractCubedSphereField, model)
    compute_at!(field, time(model))
    data = Tuple(parent(field.data[n]) for n in 1:length(field.data))
    return data
end

#####
##### Applying flux boundary conditions
#####

import Oceananigans.Models.HydrostaticFreeSurfaceModels: apply_flux_bcs!

function apply_flux_bcs!(Gcⁿ::AbstractCubedSphereField, c::AbstractCubedSphereField, arch, args...)

    for (face_index, Gcⁿ_face) in enumerate(faces(Gcⁿ))
        apply_flux_bcs!(Gcⁿ_face, get_face(c, face_index), arch,
                        Tuple(get_face(a, face_index) for a in args)...)
    end

    return nothing
end

import Oceananigans.Models.HydrostaticFreeSurfaceModels: top_tracer_boundary_conditions

function face_tracers(tracers, face)
    tracer_names = propertynames(tracers)
    return NamedTuple(name => get_face(tracers[name], face) for name in tracer_names)
end

function top_tracer_boundary_conditions(grid::ConformalCubedSphereGrid, tracers)
    tracer_names = propertynames(tracers)
    top_tracer_bcs = Tuple(top_tracer_boundary_conditions(get_face(grid, i), face_tracers(tracers, i))
                           for i = 1:length(grid.faces))
    return CubedSphereFaces(top_tracer_bcs)
end

#####
##### Forcing functions on the cubed sphere
#####

using Oceananigans.Forcings: user_function_arguments
import Oceananigans.Forcings: ContinuousForcing

@inline function (forcing::ContinuousForcing{LX, LY, LZ})(i, j, k, grid::OrthogonalSphericalShellGrid, clock, model_fields) where {LX, LY, LZ}

    args = user_function_arguments(i, j, k, grid, model_fields, forcing.parameters, forcing)

    λ = λnode(i, j, k, grid, LX(), LY(), LZ())
    φ = φnode(i, j, k, grid, LX(), LY(), LZ())
    z = znode(i, j, k, grid, LX(), LY(), LZ())

    return @inbounds forcing.func(λ, φ, z, clock.time, args...)
end

#####
##### NaN checker for cubed sphere fields
#####

import Oceananigans.Simulations: hasnan

function hasnan(field::AbstractCubedSphereField)
    for (face_index, face_field) in enumerate(faces(field))
        nanface = hasnan(face_field)
        if nanface
            @info "NaN found on face $face_index..."
            return true
        end
    end
    return false
end

#####
##### CFL for cubed sphere fields
#####

import Oceananigans.Advection: cell_advection_timescale

function cell_advection_timescale(grid::ConformalCubedSphereGrid, velocities)

    min_timescale_on_faces = []

    for (face_index, grid_face) in enumerate(grid.faces)
        velocities_face = get_face(velocities, face_index)
        min_timescale_on_face = cell_advection_timescale(grid_face, velocities_face)
        push!(min_timescale_on_faces, min_timescale_on_face)
    end

    return minimum(min_timescale_on_faces)
end

#####
##### compute...
#####

using Oceananigans.AbstractOperations: _compute!
using Oceananigans.Fields: compute_at!, offset_index

import Oceananigans.Fields: compute!

const CubedSphereComputedField{LX, LY, LZ} = Field{LX, LY, LZ,
                                                   <:AbstractOperation,
                                                   <:ConformalCubedSphereGrid} where {LX, LY, LZ}

function compute!(comp::CubedSphereComputedField, time=nothing)
    # First compute `dependencies`:
    compute_at!(comp.operand, time)

    arch = architecture(comp)
    foreach(faces(comp)) do c
        parameters = KernelParameters(size(c), map(offset_index, c.indices))
        launch!(arch, c.grid, parameters, _compute!, c.data, c.operand)
    end
    
    fill_halo_regions!(comp)

    return comp
end

#####
##### Output writing for cubed sphere fields
#####

import Oceananigans.OutputWriters: fetch_output

function fetch_output(field::AbstractCubedSphereField, model, field_slicer)
    compute!(field)
    return Tuple(fetch_output(face_field, model, field_slicer) for face_field in faces(field))
end

#####
##### StateChecker for each face is useful for debugging
#####

import Oceananigans.Diagnostics: state_check

function state_check(field::AbstractCubedSphereField, name, pad)
    face_fields = faces(field)
    Nf = length(face_fields)
    for (face_index, face_field) in enumerate(face_fields)
        face_str = " face $face_index"
        state_check(face_field, string(name) * face_str, pad + length(face_str))

        # Leave empty line between fields for easier visual inspection.
        face_index == Nf && @info ""
    end
end

end # module
