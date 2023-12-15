push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/SphericalTensors/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/DMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/Hamiltonians/src")

push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")

using Test, Random
using SphericalTensors, TEBD

Random.seed!(1234)

include("util.jl")

## algorithms
include("algorithm/tebd.jl")
include("algorithm/correlations.jl")
