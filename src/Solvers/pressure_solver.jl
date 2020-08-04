using Oceananigans.BoundaryConditions: BC, Periodic

struct PressureSolver{T, A, W, S, R, C}
          type :: T
  architecture :: A
   wavenumbers :: W
       storage :: S
    transforms :: R
     constants :: C
end

abstract type PressureSolverType end

struct TriplyPeriodic       <: PressureSolverType end
struct HorizontallyPeriodic <: PressureSolverType end
struct Channel              <: PressureSolverType end
struct Box                  <: PressureSolverType end

poisson_bc_symbol(::BC) = :N
poisson_bc_symbol(::BC{<:Periodic}) = :P

function PressureSolver(arch, grid, pressure_bcs, planner_flag=FFTW.PATIENT)
    x = poisson_bc_symbol(pressure_bcs.x.left)
    y = poisson_bc_symbol(pressure_bcs.y.left)
    z = poisson_bc_symbol(pressure_bcs.z.left)
    bc_symbol = Symbol(x, y, z)

    if bc_symbol == :PPP
        @warn "TriplyPeriodicPressureSolver is still untested."
        return TriplyPeriodicPressureSolver(arch, grid, pressure_bcs, planner_flag)
    elseif bc_symbol == :PPN
        return HorizontallyPeriodicPressureSolver(arch, grid, pressure_bcs, planner_flag)
    elseif bc_symbol == :PNN
        return ChannelPressureSolver(arch, grid, pressure_bcs, planner_flag)
    elseif bc_symbol == :NNN
        return BoxPressureSolver(arch, grid, pressure_bcs, planner_flag)
    else
        throw(ArgumentError("Unsupported pressure boundary conditions: $bc_symbol"))
    end
end
