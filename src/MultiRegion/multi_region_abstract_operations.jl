using Oceananigans.AbstractOperations: UnaryOperation, BinaryOperation, MultiaryOperation, Derivative, KernelFunctionOperation

const MultiRegionUnaryOperation{LX, LY, LZ, O, A, I, G, T} = UnaryOperation{LX, LY, LZ, O, A, I, <:MultiRegionGrid, T} where {LX, LY, LZ, O, A, I, T}
const MultiRegionBinaryOperation{LX, LY, LZ, O, A, B, IA, IB, G, T} = BinaryOperation{LX, LY, LZ, O, A, B, IA, IB, <:MultiRegionGrid, T} where {LX, LY, LZ, O, A, B, IA, IB, T}
const MultiRegionMultiaryOperation{LX, LY, LZ, N, O, A, I, G, T} = MultiaryOperation{LX, LY, LZ, N, O, A, I, <:MultiRegionGrid, T} where {LX, LY, LZ, N, O, A, I, T}
const MultiRegionDerivative{LX, LY, LZ, D, A, IN, AD, G, T} = Derivative{LX, LY, LZ, D, A, IN, AD, <:MultiRegionGrid, T} where {LX, LY, LZ, D, A, IN, AD, T}
const MultiRegionKernelFunctionOperation{LX, LY, LZ, P, G, T, K, D} = KernelFunctionOperation{LX, LY, LZ, P, <:MultiRegionGrid, T, K, D} where {LX, LY, LZ, P, T, K, D}

const MultiRegionAbstractOperation = Union{MultiRegionBinaryOperation, 
                                            MultiRegionUnaryOperation,
                                         MultiRegionMultiaryOperation,
                                                MultiRegionDerivative,
                                   MultiRegionKernelFunctionOperation}

# Utils
Base.size(f::MultiRegionAbstractOperation) = size(getregion(f.grid, 1))

@inline isregional(f::MultiRegionAbstractOperation) = true
@inline devices(f::MultiRegionAbstractOperation)    = devices(f.grid)
sync_all_devices!(f::MultiRegionAbstractOperation)  = sync_all_devices!(devices(f.grid))

@inline switch_device!(f::MultiRegionAbstractOperation, d) = switch_device!(f.grid, d)
@inline getdevice(f::MultiRegionAbstractOperation, d)      = getdevice(f.grid, d)

## Functions applied regionally
compute_at!(f::MultiRegionAbstractOperation, time) = apply_regionally!(compute_at!, f, time)
compute!(f::MultiRegionAbstractOperation)          = apply_regionally!(compute!, f, time)

for T in [:BinaryOperation, :UnaryOperation, :MultiaryOperation, :Derivative, :KernelFunctionOperation]
    @eval begin
        @inline getregion(f::$T{LX, LY, LZ}, r) where {LX, LY, LZ} =
                        $T{LX, LY, LZ}(Tuple(_getregion(getproperty(f, n), r) for n in fieldnames($T))...)

        @inline _getregion(f::$T{LX, LY, LZ}, r) where {LX, LY, LZ} =
                        $T{LX, LY, LZ}(Tuple(getregion(getproperty(f, n), r) for n in fieldnames($T))...)
    end
end
