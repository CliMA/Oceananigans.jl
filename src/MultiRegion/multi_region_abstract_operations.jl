using Oceananigans.AbstractOperations: UnaryOperation, BinaryOperation, MultiaryOperation, Derivative, KernelFunctionOperation, ConditionalOperation

# Field and FunctionField (both fields with "grids attached")
const MultiRegionUnaryOperation{LX, LY, LZ, O, A, I}          = UnaryOperation{LX, LY, LZ, O, A, I, <:MultiRegionGrids} where {LX, LY, LZ, O, A, I}
const MultiRegionBinaryOperation{LX, LY, LZ, O, A, B, IA, IB} = BinaryOperation{LX, LY, LZ, O, A, B, IA, IB, <:MultiRegionGrids} where {LX, LY, LZ, O, A, B, IA, IB}
const MultiRegionMultiaryOperation{LX, LY, LZ, N, O, A, I}    = MultiaryOperation{LX, LY, LZ, N, O, A, I, <:MultiRegionGrids} where {LX, LY, LZ, N, O, A, I}
const MultiRegionDerivative{LX, LY, LZ, D, A, IN, AD}         = Derivative{LX, LY, LZ, D, A, IN, AD, <:MultiRegionGrids} where {LX, LY, LZ, D, A, IN, AD}
const MultiRegionKernelFunctionOperation{LX, LY, LZ}          = KernelFunctionOperation{LX, LY, LZ, <:MultiRegionGrids} where {LX, LY, LZ, P}
const MultiRegionConditionalOperation{LX, LY, LZ, O, F}       = ConditionalOperation{LX, LY, LZ, O, F, <:MultiRegionGrids} where {LX, LY, LZ, O, F}

const MultiRegionAbstractOperation = Union{MultiRegionBinaryOperation, 
                                           MultiRegionUnaryOperation,
                                           MultiRegionMultiaryOperation,
                                           MultiRegionDerivative,
                                           MultiRegionKernelFunctionOperation,
                                           MultiRegionConditionalOperation}
# Utils
Base.size(f::MultiRegionAbstractOperation) = size(getregion(f.grid, 1))

@inline isregional(f::MultiRegionAbstractOperation) = true
@inline devices(f::MultiRegionAbstractOperation)    = devices(f.grid)
sync_all_devices!(f::MultiRegionAbstractOperation)  = sync_all_devices!(devices(f.grid))

@inline switch_device!(f::MultiRegionAbstractOperation, d) = switch_device!(f.grid, d)
@inline getdevice(f::MultiRegionAbstractOperation, d)      = getdevice(f.grid, d)

for T in [:BinaryOperation, :UnaryOperation, :MultiaryOperation, :Derivative, :ConditionalOperation]
    @eval begin
        @inline getregion(f::$T{LX, LY, LZ}, r) where {LX, LY, LZ} =
                          $T{LX, LY, LZ}(Tuple(_getregion(getproperty(f, n), r) for n in fieldnames($T))...)

        @inline _getregion(f::$T{LX, LY, LZ}, r) where {LX, LY, LZ} =
                           $T{LX, LY, LZ}(Tuple(getregion(getproperty(f, n), r) for n in fieldnames($T))...)
    end
end

@inline getregion(κ::KernelFunctionOperation{LX, LY, LZ}, r) where {LX, LY, LZ} = 
                KernelFunctionOperation{LX, LY, LZ}(_getregion(κ.kernel_function, r),
                                                    _getregion(κ.grid, r), 
                                                    _getregion(κ.arguments, r)...)

@inline _getregion(κ::KernelFunctionOperation{LX, LY, LZ}, r) where {LX, LY, LZ} = 
                KernelFunctionOperation{LX, LY, LZ}(getregion(κ.kernel_function, r),
                                                    getregion(κ.grid, r), 
                                                    getregion(κ.arguments, r)...)