push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/DMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/InfiniteDMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/GeneralHamiltonians/src")

push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")

using Test, Random
using TEBD

Random.seed!(1234)

include("util.jl")

## algorithms
include("algorithm/tebd.jl")
include("algorithm/itebd.jl")
include("algorithm/correlations.jl")