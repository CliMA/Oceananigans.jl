using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: BCType

import Base: show
import Oceananigans.BoundaryConditions: bctype_str, print_condition

struct CubedSphereExchange <: BCType end

const CubedSphereExchangeBC = BoundaryCondition{<:CubedSphereExchange}

bctype_str(::CubedSphereExchangeBC) ="CubedSphereExchange"

CubedSphereExchangeBoundaryCondition(val; kwargs...) = BoundaryCondition(CubedSphereExchange, val; kwargs...)

struct CubedSphereExchangeInformation{F, S}
    from_face :: F
      to_face :: F
    from_side :: S
      to_side :: S
end

CubedSphereExchangeInformation(; from_face, to_face, from_side, to_side) =
    CubedSphereExchangeInformation(from_face, to_face, from_side, to_side)

Base.show(io::IO, ex::CubedSphereExchangeInformation) =
    print(io, "CubedSphereExchangeInformation: (from: face $(ex.from_face) $(ex.from_side) side, to: face $(ex.to_face) $(ex.to_side) side)")

print_condition(info::CubedSphereExchangeInformation) =
    "(from: face $(info.from_face) $(info.from_side) side, to: face $(info.to_face) $(info.to_side) side)"

function inject_cubed_sphere_exchange_boundary_conditions(field_bcs, face_number, face_connectivity)

    west_exchange_info = CubedSphereExchangeInformation(
        from_face = face_number,
        from_side = :west,
          to_face = face_connectivity[face_number].west.face,
          to_side = face_connectivity[face_number].west.side
    )

    east_exchange_info = CubedSphereExchangeInformation(
        from_face = face_number,
        from_side = :east,
          to_face = face_connectivity[face_number].east.face,
          to_side = face_connectivity[face_number].east.side
    )

    south_exchange_info = CubedSphereExchangeInformation(
        from_face = face_number,
        from_side = :south,
          to_face = face_connectivity[face_number].south.face,
          to_side = face_connectivity[face_number].south.side
    )

    north_exchange_info = CubedSphereExchangeInformation(
        from_face = face_number,
        from_side = :north,
          to_face = face_connectivity[face_number].north.face,
          to_side = face_connectivity[face_number].north.side
    )

    west_exchange_bc = CubedSphereExchangeBoundaryCondition(west_exchange_info)
    east_exchange_bc = CubedSphereExchangeBoundaryCondition(east_exchange_info)
    south_exchange_bc = CubedSphereExchangeBoundaryCondition(south_exchange_info)
    north_exchange_bc = CubedSphereExchangeBoundaryCondition(north_exchange_info)

    x_bcs = CoordinateBoundaryConditions(west_exchange_bc, east_exchange_bc)
    y_bcs = CoordinateBoundaryConditions(south_exchange_bc, north_exchange_bc)

    return FieldBoundaryConditions(x_bcs, y_bcs, field_bcs.z)
end
