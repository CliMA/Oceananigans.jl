module CubedSpheres

export
    ConformalCubedSphereGrid,
    ConformalCubedSphereField,
    λnodes, φnodes

include("cubed_sphere_utils.jl")
include("conformal_cubed_sphere_grid.jl")
include("cubed_sphere_exchange_bcs.jl")
include("cubed_sphere_fields.jl")
include("cubed_sphere_set!.jl")
include("cubed_sphere_halo_filling.jl")
include("cubed_sphere_kernel_launching.jl")

#####
##### Proper launch! when `ExplicitFreeSurface` is an argument
#####

using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, PrescribedVelocityFields

maybe_replace_with_face(free_surface::ExplicitFreeSurface, cubed_sphere_grid, face_number) =
    ExplicitFreeSurface(free_surface.η.faces[face_number], free_surface.gravitational_acceleration)

maybe_replace_with_face(velocities::PrescribedVelocityFields, cubed_sphere_grid, face_number) =
    PrescribedVelocityFields(velocities.u.faces[face_number], velocities.v.faces[face_number], velocities.w.faces[face_number], velocities.parameters)

#####
##### NaN checker for cubed sphere fields
#####

import Oceananigans.Diagnostics: error_if_nan_in_field

function error_if_nan_in_field(field::AbstractCubedSphereField, name, clock)
    for (face_number, field_face) in enumerate(field.faces)
        error_if_nan_in_field(field_face, string(name) * " (face $face_number)", clock)
    end
end

#####
##### CFL for cubed sphere fields
#####

import Oceananigans.Diagnostics: accurate_cell_advection_timescale

function accurate_cell_advection_timescale(grid::ConformalCubedSphereGrid, velocities)

    min_timescale_on_faces = []

    for (face_number, grid_face) in enumerate(grid.faces)
        velocities_face = maybe_replace_with_face(velocities, grid, face_number)
        min_timescale_on_face = accurate_cell_advection_timescale(grid_face, velocities_face)
        push!(min_timescale_on_faces, min_timescale_on_face)
    end

    return minimum(min_timescale_on_faces)
end

#####
##### Output writing for cubed sphere fields
#####

import Oceananigans.OutputWriters: fetch_output

fetch_output(field::AbstractCubedSphereField, model, field_slicer) =
    Tuple(fetch_output(field_face, model, field_slicer) for field_face in field.faces)

#####
##### StateChecker for each face is useful for debugging
#####

import Oceananigans.Diagnostics: state_check

function state_check(field::AbstractCubedSphereField, name, pad)
    Nf = length(field.faces)
    for (face_number, field_face) in enumerate(field.faces)
        face_str = " face $face_number"
        state_check(field_face, string(name) * face_str, pad + length(face_str))

        # Leave empty line between fields for easier visual inspection.
        face_number == Nf && @info ""
    end
end

end # module
