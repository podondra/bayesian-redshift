include("BayesianSZNet.jl")
using .BayesianSZNet

p = parse(Float64, ARGS[1])
λ = parse(Float64, ARGS[2])
patience = parse(Int, ARGS[3])

X_tr, z_tr, X_va, z_va = getdr12data()
multitrain(FCNN, X_tr, z_tr, X_va, z_va, constructorname="fcnn"; p, patience, λ)
