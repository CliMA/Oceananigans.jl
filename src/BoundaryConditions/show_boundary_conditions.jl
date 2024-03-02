import Base: show
using Oceananigans.Utils: prettysummary

const DFBC = DefaultBoundaryCondition
const IBC = BoundaryCondition{Open, Nothing} # ImpenetrableBoundaryCondition

bc_str(::FBC)     = "Flux"
bc_str(::PBC)     = "Periodic"
bc_str(::OBC)     = "Open"
bc_str(::VBC)     = "Value"
bc_str(::GBC)     = "Gradient"
bc_str(::ZFBC)    = "ZeroFlux"
bc_str(::IBC)     = "Impenetrable"
bc_str(::DFBC)    = "Default"
bc_str(::MCBC)    = "MultiRegionCommunication"
bc_str(::DCBC)    = "DistributedCommunication"
bc_str(::Nothing) = "Nothing"

#####
##### BoundaryCondition
#####

Base.summary(bc::DFBC) = string("DefaultBoundaryCondition (", summary(bc.boundary_condition), ")")
Base.summary(bc::OBC)  = string("OpenBoundaryCondition: ", prettysummary(bc.condition))
Base.summary(bc::FBC)  = string("FluxBoundaryCondition: ", prettysummary(bc.condition))
Base.summary(bc::VBC)  = string("ValueBoundaryCondition: ", prettysummary(bc.condition))
Base.summary(bc::GBC)  = string("GradientBoundaryCondition: ", prettysummary(bc.condition))
Base.summary(::PBC)    = string("PeriodicBoundaryCondition")
Base.summary(bc::DCBC) = string("DistributedBoundaryCondition: ", prettysummary(bc.condition))

show(io::IO, bc::BoundaryCondition) = print(io, summary(bc))

#####
##### FieldBoundaryConditions
#####

Base.summary(fbcs::FieldBoundaryConditions) = "FieldBoundaryConditions"

show_field_boundary_conditions(bcs::FieldBoundaryConditions, padding="") =
    string("Oceananigans.FieldBoundaryConditions, with boundary conditions", "\n",
           padding, "├── west: ",     summary(bcs.west), "\n",
           padding, "├── east: ",     summary(bcs.east), "\n",
           padding, "├── south: ",    summary(bcs.south), "\n",
           padding, "├── north: ",    summary(bcs.north), "\n",
           padding, "├── bottom: ",   summary(bcs.bottom), "\n",
           padding, "├── top: ",      summary(bcs.top), "\n",
           padding, "└── immersed: ", summary(bcs.immersed))

Base.show(io::IO, fieldbcs::FieldBoundaryConditions) = print(io, show_field_boundary_conditions(fieldbcs))
