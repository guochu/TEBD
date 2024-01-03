# infinite TEBD


function trotter_propagator(ham::InfiniteQuantumOperator, tspan::Tuple{<:Number, <:Number}; unitcellsize::Int=length(ham), kwargs...) 
	ham2 = changeunitcell(absorb_one_bodies(ham), unitcellsize=unitcellsize)
	return _trotter_propagator(oneperiod(ham2), tspan; kwargs...)
end
trotter_propagator(ham::InfiniteQuantumOperator, dt::Number; kwargs...) = trotter_propagator(ham, (0, dt); kwargs...)
