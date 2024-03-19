# All coefficients are taken from
# Balsara & Shu, "Monotonicity Preserving Weighted Essentially Non-oscillatory Schemes with Inceasingly High Order of Accuracy"

## ------------ OPTIMAL RECONSTRUCTION COEFFICIENTS ----------- ##

const coeff_left_20 = 2
const coeff_left_21 = 1

const coeff_left_30 = 3
const coeff_left_31 = 6
const coeff_left_32 = 1

const coeff_left_40 = 4
const coeff_left_41 = 18
const coeff_left_42 = 12
const coeff_left_43 = 1

const coeff_left_50 = 5
const coeff_left_51 = 40
const coeff_left_52 = 60
const coeff_left_53 = 40
const coeff_left_54 = 1

const coeff_left_60 = 6
const coeff_left_61 = 75
const coeff_left_62 = 200
const coeff_left_63 = 150
const coeff_left_64 = 30
const coeff_left_65 = 1

for buffer in [2, 3, 4, 5, 6]
    for stencil in 0:buffer - 1
        coeff_right = Symbol(:coeff_right_, buffer, stencil)
        coeff_left  = Symbol(:coeff_left_,  buffer, buffer - stencil - 1)
        @eval const $coeff_right = $coeff_left
    end
end

## ------------ COEFFICIENTS FOR SMOOTHNESS CALCULATION ----------- ##

# The constants are defined as: Cβ_$(buffer)$(stencil)$(element)
# Where buffer = (weno order + 1) / 2 and stencil = 0 : buffer - 1

#####
##### Third order WENO coefficients
#####

# Stencil 0
const Cβ_201 =  1
const Cβ_202 = -2
const Cβ_203 =  1

# Stencil 1
const Cβ_211 =  1
const Cβ_212 = -2
const Cβ_213 =  1

#####
##### Fifth order WENO coefficients
#####

# Stencil 0
const Cβ_301 =  10
const Cβ_302 = -31
const Cβ_303 =  11
const Cβ_304 =  25
const Cβ_305 = -19
const Cβ_306 =   4

# Stencil 1
const Cβ_311 =   4 
const Cβ_312 = -13 
const Cβ_313 =   5
const Cβ_314 =  13
const Cβ_315 = -13
const Cβ_316 =   4

# Stencil 2
const Cβ_321 =   4 
const Cβ_322 = -19 
const Cβ_323 =  11
const Cβ_324 =  25
const Cβ_325 = -31
const Cβ_326 =  10

#####
##### Seventh order WENO coefficients
#####

# Stencil 0
const Cβ_401  =   2107
const Cβ_402  =  -9402
const Cβ_403  =   7042
const Cβ_404  =  -1854
const Cβ_405  =  11003
const Cβ_406  = -17246
const Cβ_407  =   4642
const Cβ_408  =   7043
const Cβ_409  =  -3882
const Cβ_4010 =   0547

# Stencil 1
const Cβ_411  =   547
const Cβ_412  = -2522
const Cβ_413  =  1922
const Cβ_414  =  -494
const Cβ_415  =  3443
const Cβ_416  = -5966
const Cβ_417  =  1602
const Cβ_418  =  2843
const Cβ_419  = -1642
const Cβ_4110 =   267

# Stencil 2
const Cβ_421  =   267
const Cβ_422  = -1642
const Cβ_423  =  1602
const Cβ_424  =  -494
const Cβ_425  =  2843
const Cβ_426  = -5966
const Cβ_427  =  1922
const Cβ_428  =  3443
const Cβ_429  = -2522
const Cβ_4210 =   547

# Stencil 3
const Cβ_431  =    547
const Cβ_432  =  -3882
const Cβ_433  =   4642
const Cβ_434  =  -1854
const Cβ_435  =   7043
const Cβ_436  = -17246
const Cβ_437  =   7042
const Cβ_438  =  11003
const Cβ_439  =  -9402
const Cβ_4310 =   2107

#####
##### Ninth order WENO coefficients
#####

# Stencil 0
const Cβ_501  =   107918
const Cβ_502  =  -649501
const Cβ_503  =   758823
const Cβ_504  =  -411487
const Cβ_505  =   086329
const Cβ_506  =  1020563
const Cβ_507  = -2462076
const Cβ_508  =  1358458
const Cβ_509  =  -288007
const Cβ_5010 =  1521393
const Cβ_5011 = -1704396
const Cβ_5012 =   364863
const Cβ_5013 =   482963
const Cβ_5014 =  -208501
const Cβ_5015 =   022658

# Stencil 1
const Cβ_511  =  022658
const Cβ_512  = -140251
const Cβ_513  =  165153
const Cβ_514  = -088297
const Cβ_515  =  018079
const Cβ_516  =  242723
const Cβ_517  = -611976
const Cβ_518  =  337018
const Cβ_519  = -070237
const Cβ_5110 =  406293
const Cβ_5111 = -464976
const Cβ_5112 =  099213
const Cβ_5113 =  138563
const Cβ_5114 = -060871
const Cβ_5115 =  006908

# Stencil 2
const Cβ_521  =  006908
const Cβ_522  = -051001
const Cβ_523  =  067923
const Cβ_524  = -038947
const Cβ_525  =  008209
const Cβ_526  =  104963
const Cβ_527  = -299076
const Cβ_528  =  179098
const Cβ_529  = -038947
const Cβ_5210 =  231153
const Cβ_5211 = -299076
const Cβ_5212 =  067923
const Cβ_5213 =  104963
const Cβ_5214 = -051001
const Cβ_5215 =  006908

# Stencil 3
const Cβ_531  =  006908
const Cβ_532  = -060871
const Cβ_533  =  099213
const Cβ_534  = -070237
const Cβ_535  =  018079
const Cβ_536  =  138563
const Cβ_537  = -464976
const Cβ_538  =  337018
const Cβ_539  = -088297
const Cβ_5310 =  406293
const Cβ_5311 = -611976
const Cβ_5312 =  165153
const Cβ_5313 =  242723
const Cβ_5314 = -140251
const Cβ_5315 =  022658

# Stencil 4
const Cβ_541  =   022658
const Cβ_542  =  -208501
const Cβ_543  =   364863
const Cβ_544  =  -288007
const Cβ_545  =   086329
const Cβ_546  =   482963
const Cβ_547  = -1704396
const Cβ_548  =  1358458
const Cβ_549  =  -411487
const Cβ_5410 =  1521393
const Cβ_5411 = -2462076
const Cβ_5412 =   758823
const Cβ_5413 =  1020563
const Cβ_5414 =  -649501
const Cβ_5415 =   107918

#####
##### Eleventh order WENO coefficients
#####

const Cβ_601  =   0.6150211
const Cβ_602  =  -4.7460464
const Cβ_603  =   7.6206736
const Cβ_604  =  -6.3394124
const Cβ_605  =   2.706017
const Cβ_606  =  -0.471274
const Cβ_607  =   9.4851237
const Cβ_608  = -31.1771244
const Cβ_609  =  26.2901672
const Cβ_6010 = -11.3206788
const Cβ_6011 =   1.983435
const Cβ_6012 =  26.0445372
const Cβ_6013 = -44.4003904
const Cβ_6014 =  19.2596472
const Cβ_6015 =  -3.3918804
const Cβ_6016 =  19.0757572
const Cβ_6017 = -16.6461044
const Cβ_6018 =   2.9442256
const Cβ_6019 =   3.6480687
const Cβ_6020 =  -1.2950184
const Cβ_6021 =   0.1152561

const Cβ_611  =  0.1152561
const Cβ_612  = -0.9117992
const Cβ_613  =  1.474248
const Cβ_614  = -1.2183636
const Cβ_615  =  0.5134574
const Cβ_616  = -0.0880548
const Cβ_617  =  1.9365967
const Cβ_618  = -6.5224244
const Cβ_619  =  5.5053752
const Cβ_6110 = -2.3510468
const Cβ_6111 =  0.4067018
const Cβ_6112 =  5.6662212
const Cβ_6113 = -9.7838784
const Cβ_6114 =  4.2405032
const Cβ_6115 = -0.7408908
const Cβ_6116 =  4.3093692
const Cβ_6117 = -3.7913324
const Cβ_6118 =  0.6694608
const Cβ_6119 =  0.8449957
const Cβ_6120 = -0.3015728
const Cβ_6121 =  0.0271779

const Cβ_621  =  0.0271779
const Cβ_622  = -0.23808
const Cβ_623  =  0.4086352
const Cβ_624  = -0.3462252
const Cβ_625  =  0.1458762
const Cβ_626  = -0.024562
const Cβ_627  =  0.5653317
const Cβ_628  = -2.0427884
const Cβ_629  =  1.7905032
const Cβ_6210 = -0.7727988
const Cβ_6211 =  0.1325006
const Cβ_6212 =  1.9510972
const Cβ_6213 = -3.5817664
const Cβ_6214 =  1.5929912
const Cβ_6215 = -0.279266
const Cβ_6216 =  1.7195652
const Cβ_6217 = -1.5880404
const Cβ_6218 =  0.2863984
const Cβ_6219 =  0.3824847
const Cβ_6220 = -0.1429976
const Cβ_6221 =  0.0139633

## FROM HERE NEED TO CHANGE THEM!!!

const Cβ_631  =  0.0271779
const Cβ_632  = -0.23808
const Cβ_633  =  0.4086352
const Cβ_634  = -0.3462252
const Cβ_635  =  0.1458762
const Cβ_636  = -0.024562
const Cβ_637  =  0.5653317
const Cβ_638  = -2.0427884
const Cβ_639  =  1.7905032
const Cβ_6310 = -0.7727988
const Cβ_6311 =  0.1325006
const Cβ_6312 =  1.9510972
const Cβ_6313 = -3.5817664
const Cβ_6314 =  1.5929912
const Cβ_6315 = -0.279266
const Cβ_6316 =  1.7195652
const Cβ_6317 = -1.5880404
const Cβ_6318 =  0.2863984
const Cβ_6319 =  0.3824847
const Cβ_6320 = -0.1429976
const Cβ_6321 =  0.0139633

const Cβ_641  =   0.0271779
const Cβ_642  =  -0.23808
const Cβ_643  =   0.4086352
const Cβ_644  =  -0.3462252
const Cβ_645  =   0.1458762
const Cβ_646  =  -0.024562
const Cβ_647  =   0.5653317
const Cβ_648  =  -2.0427884
const Cβ_649  =   1.7905032
const Cβ_6410 =  -0.7727988
const Cβ_6411 =   0.1325006
const Cβ_6412 =   1.9510972
const Cβ_6413 =  -3.5817664
const Cβ_6414 =   1.5929912
const Cβ_6415 =  -0.279266
const Cβ_6416 =   1.7195652
const Cβ_6417 =  -1.5880404
const Cβ_6418 =   0.2863984
const Cβ_6419 =   0.3824847
const Cβ_6420 =  -0.1429976
const Cβ_6421 =   0.0139633

const Cβ_651  =   0.0271779
const Cβ_652  =  -0.23808
const Cβ_653  =   0.4086352
const Cβ_654  =  -0.3462252
const Cβ_655  =   0.1458762
const Cβ_656  =  -0.024562
const Cβ_657  =   0.5653317
const Cβ_658  =  -2.0427884
const Cβ_659  =   1.7905032
const Cβ_6510 =  -0.7727988
const Cβ_6511 =   0.1325006
const Cβ_6512 =   1.9510972
const Cβ_6513 =  -3.5817664
const Cβ_6514 =   1.5929912
const Cβ_6515 =  -0.279266
const Cβ_6516 =   1.7195652
const Cβ_6517 =  -1.5880404
const Cβ_6518 =   0.2863984
const Cβ_6519 =   0.3824847
const Cβ_6520 =  -0.1429976
const Cβ_6521 =   0.0139633

# # _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)
# @inline smoothness_coefficients(scheme::WENO{2, FT}, ::Val{0}) where FT = @inbounds FT.((1, -2, 1))
# @inline smoothness_coefficients(scheme::WENO{2, FT}, ::Val{1}) where FT = @inbounds FT.((1, -2, 1))

# @inline smoothness_coefficients(scheme::WENO{3, FT}, ::Val{0}) where FT = @inbounds FT.((10, -31, 11, 25, -19,  4))
# @inline smoothness_coefficients(scheme::WENO{3, FT}, ::Val{1}) where FT = @inbounds FT.((4,  -13, 5,  13, -13,  4))
# @inline smoothness_coefficients(scheme::WENO{3, FT}, ::Val{2}) where FT = @inbounds FT.((4,  -19, 11, 25, -31, 10))

# @inline smoothness_coefficients(scheme::WENO{4, FT}, ::Val{0}) where FT = @inbounds FT.((2.107,  -9.402, 7.042, -1.854, 11.003,  -17.246,  4.642,  7.043,  -3.882, 0.547))
# @inline smoothness_coefficients(scheme::WENO{4, FT}, ::Val{1}) where FT = @inbounds FT.((0.547,  -2.522, 1.922, -0.494,  3.443,  - 5.966,  1.602,  2.843,  -1.642, 0.267))
# @inline smoothness_coefficients(scheme::WENO{4, FT}, ::Val{2}) where FT = @inbounds FT.((0.267,  -1.642, 1.602, -0.494,  2.843,  - 5.966,  1.922,  3.443,  -2.522, 0.547))
# @inline smoothness_coefficients(scheme::WENO{4, FT}, ::Val{3}) where FT = @inbounds FT.((0.547,  -3.882, 4.642, -1.854,  7.043,  -17.246,  7.042, 11.003,  -9.402, 2.107))

# @inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{0}) where FT = @inbounds FT.((1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658)) 
# @inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{1}) where FT = @inbounds FT.((0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908)) 
# @inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{2}) where FT = @inbounds FT.((0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908)) 
# @inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{3}) where FT = @inbounds FT.((0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658)) 
# @inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{4}) where FT = @inbounds FT.((0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)) 

# @inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{0}) where FT = @inbounds FT.((0.6150211, -4.7460464, 7.6206736, -6.3394124, 2.7060170, -0.4712740,  9.4851237, -31.1771244, 26.2901672, -11.3206788,  1.9834350, 26.0445372, -44.4003904, 19.2596472, -3.3918804, 19.0757572, -16.6461044, 2.9442256, 3.6480687, -1.2950184, 0.1152561)) 
# @inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{1}) where FT = @inbounds FT.((0.1152561, -0.9117992, 1.4742480, -1.2183636, 0.5134574, -0.0880548,  1.9365967,  -6.5224244,  5.5053752,  -2.3510468,  0.4067018,  5.6662212,  -9.7838784,  4.2405032, -0.7408908,  4.3093692,  -3.7913324, 0.6694608, 0.8449957, -0.3015728, 0.0271779)) 
# @inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{2}) where FT = @inbounds FT.((0.0271779, -0.2380800, 0.4086352, -0.3462252, 0.1458762, -0.0245620,  0.5653317,  -2.0427884,  1.7905032,  -0.7727988,  0.1325006,  1.9510972,  -3.5817664,  1.5929912, -0.2792660,  1.7195652,  -1.5880404, 0.2863984, 0.3824847, -0.1429976, 0.0139633)) 
# @inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{3}) where FT = @inbounds FT.((0.0139633, -0.1429976, 0.2863984, -0.2792660, 0.1325006, -0.0245620,  0.3824847,  -1.5880404,  1.5929912,  -0.7727988,  0.1458762,  1.7195652,  -3.5817664,  1.7905032, -0.3462252,  1.9510972,  -2.0427884, 0.4086352, 0.5653317, -0.2380800, 0.0271779)) 
# @inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{4}) where FT = @inbounds FT.((0.0271779, -0.3015728, 0.6694608, -0.7408908, 0.4067018, -0.0880548,  0.8449957,  -3.7913324,  4.2405032,  -2.3510468,  0.5134574,  4.3093692,  -9.7838784,  5.5053752, -1.2183636,  5.6662212,  -6.5224244, 1.4742480, 1.9365967, -0.9117992, 0.1152561)) 
# @inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{5}) where FT = @inbounds FT.((0.1152561, -1.2950184, 2.9442256, -3.3918804, 1.9834350, -0.4712740,  3.6480687, -16.6461044, 19.2596472, -11.3206788,  2.7060170, 19.0757572, -44.4003904, 26.2901672, -6.3394124, 26.0445372, -31.1771244, 7.6206736, 9.4851237, -4.7460464, 0.6150211)) 
