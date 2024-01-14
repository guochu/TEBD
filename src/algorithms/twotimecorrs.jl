# two time correlation for pure state and density matrices
# the complexity arises from the fact that the A, B operators may change the sector of the state

_to_n(x::AdjointMPO) = MPO(unsafe_mpotensor_adjoint.(storage(x.parent)))
_to_a(x::MPO) = adjoint(MPO(unsafe_mpotensor_adjoint.(storage(x))))

_time_reversal(t::Number) = -conj(t)
_time_reversal(a::Tuple{<:Number, <:Number}) = (_time_reversal(a[1]), _time_reversal(a[2]))

DMRG.canonicalize!(x::ExactMPS, args...; kwargs...) = nothing

function _unitary_tt_corr_at_b(h, A::AdjointMPO, B::MPO, state, times, stepper)
	state_right = B * state
	canonicalize!(state_right)
	state_left = copy(state)

	result = scalartype(state)[]
	local cache_left, cache_right	
	for i in 1:length(times)	
		# println("state norm $(norm(state_left)), $(norm(state_right)).")
		# tspan = (i == 1) ? (0., -im*times[1]) : (-im*times[i-1], -im*times[i])
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			stepper_right = similar(stepper, tspan=tspan)
			stepper_left = similar(stepper, tspan=_time_reversal(tspan))
			if (@isdefined cache_left)
				state_left, cache_left = timeevo!(state_left, h, stepper_left, cache_left)
			else
				state_left, cache_left = timeevo!(state_left, h, stepper_left)
			end
			if (@isdefined cache_right)
				state_right, cache_right = timeevo!(state_right, h, stepper_right, cache_right)
			else
				state_right, cache_right = timeevo!(state_right, h, stepper_right)
			end
		end
		push!(result, dot(A' * state_left, state_right) / dim(space_r(state_right)) )
	end
	return result
end
_unitary_tt_corr_at_b(h, A::MPO, B::AdjointMPO, state, times, stepper) = _unitary_tt_corr_at_b(h, _to_a(A), _to_n(B), state, times, stepper)


function _unitary_tt_corr_a_bt(h, A::AdjointMPO, B::MPO, state, times, stepper)
	state_right = copy(state)
	state_left = A' * state
	canonicalize!(state_left)

	result = scalartype(state)[]
	local cache_left, cache_right	
	for i in 1:length(times)	
		# tspan = (i == 1) ? (0., -im*times[1]) : (-im*times[i-1], -im*times[i])
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			stepper_right = similar(stepper, tspan=tspan)
			stepper_left = similar(stepper, tspan=_time_reversal(tspan))
			if (@isdefined cache_left)
				state_left, cache_left = timeevo!(state_left, h, stepper_left, cache_left)
			else
				state_left, cache_left = timeevo!(state_left, h, stepper_left)
			end
			if (@isdefined cache_right)
				state_right, cache_right = timeevo!(state_right, h, stepper_right, cache_right)
			else
				state_right, cache_right = timeevo!(state_right, h, stepper_right)
			end
		end
		push!(result, expectation(state_left, B, state_right) / dim(space_r(state_left)) )
	end
	return result
end
_unitary_tt_corr_a_bt(h, A::MPO, B::AdjointMPO, state, times, stepper) = _unitary_tt_corr_a_bt(h, _to_a(A), _to_n(B), state, times, stepper)

# in case one knows the state is a ground state
function _gs_unitary_tt_corr_at_b(h, A::AdjointMPO, B::MPO, state, gs_E::Real, times, stepper)
	state_right = B * state
	canonicalize!(state_right)
	state_left = state

	result = scalartype(state)[]
	local cache_right	
	for i in 1:length(times)	
		# println("state norm $(norm(state_left)), $(norm(state_right)).")
		# tspan = (i == 1) ? (0., -im*times[1]) : (-im*times[i-1], -im*times[i])
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		# tspan_left = _time_reversal(tspan)
		if abs(tspan[2] - tspan[1]) > 0.
			stepper_right = similar(stepper, tspan=tspan)
			if (@isdefined cache_right)
				state_right, cache_right = timeevo!(state_right, h, stepper_right, cache_right)
			else
				state_right, cache_right = timeevo!(state_right, h, stepper_right)
			end

		end
		push!(result, exp( -(times[i]) * gs_E) * dot(A' * state_left, state_right) / dim(space_r(state_right)) )
	end
	return result
end
_gs_unitary_tt_corr_at_b(h, A::MPO, B::AdjointMPO, state, gs_E::Real, times, stepper) = _gs_unitary_tt_corr_at_b(
	h, _to_a(A), _to_n(B), state, gs_E, times, stepper)


function _gs_unitary_tt_corr_a_bt(h, A::AdjointMPO, B::MPO, state, gs_E::Real, times, stepper)
	state_right = copy(state)
	state_left = A' * state
	canonicalize!(state_left)

	result = scalartype(state)[]
	local cache_left	
	for i in 1:length(times)	
		# tspan = (i == 1) ? (0., -im*times[1]) : (-im*times[i-1], -im*times[i])
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			stepper_left = similar(stepper, tspan=_time_reversal(tspan))
			if (@isdefined cache_left)
				state_left, cache_left = timeevo!(state_left, h, stepper_left, cache_left)
			else
				state_left, cache_left = timeevo!(state_left, h, stepper_left)
			end
		end
		push!(result, exp( times[i] * gs_E ) * expectation(state_left, B, state_right) / dim(space_r(state_left)) )
	end
	return result
end
_gs_unitary_tt_corr_a_bt(h, A::MPO, B::AdjointMPO, state, gs_E::Real, times, stepper) = _gs_unitary_tt_corr_a_bt(h, _to_a(A), _to_n(B), state, gs_E, times, stepper)

"""
	correlation_2op_1t(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::MPS, times::Vector{<:Real}, stepper::AbstractStepper; 
	reverse::Bool=false) 
	for a unitary system with hamiltonian h, compute <a(t)b> if revere=false and <a b(t)> if reverse=true
	for an open system with superoperator h, and a, b to be normal operators, compute <a(t)b> if revere=false and <a b(t)> if reverse=true.
	For open system see definitions of <a(t)b> or <a b(t)> on Page 146 of Gardiner and Zoller (Quantum Noise)
"""
function correlation_2op_1t(h::Union{QuantumOperator, MPO}, a::Union{MPO, AdjointMPO}, b::Union{MPO, AdjointMPO}, state::AbstractMPS, times::Vector{<:Real};
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	if scalartype(state) <: Real
		state = complex(state)
	end
	times = -im .* times
	reverse ? _unitary_tt_corr_a_bt(h, a, b, state, times, stepper) : _unitary_tt_corr_at_b(h, a, b, state, times, stepper)
end

"""
	gs_correlation_2op_1t(h::Union{QuantumOperator, MPO}, a::MPO, b::MPO, state::MPS, times::Vector{<:Real}; kwargs...)
	ground state two-time correlation
"""
function gs_correlation_2op_1t(h::Union{QuantumOperator, MPO}, a::Union{MPO, AdjointMPO}, b::Union{MPO, AdjointMPO}, state::AbstractMPS, times::Vector{<:Real};
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), 
	gs_energy::Real = real(expectation(h, state)), reverse::Bool=false)
	if scalartype(state) <: Real
		state = complex(state)
	end
	times = -im .* times
	reverse ? _gs_unitary_tt_corr_a_bt(h, a, b, state, gs_energy, times, stepper) : _gs_unitary_tt_corr_at_b(h, a, b, state, gs_energy, times, stepper)
end

"""
	correlation_2op_1τ(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::MPS, times::Vector{<:Real}, stepper::AbstractStepper; 
	reverse::Bool=false) 
	for a unitary system with hamiltonian h, compute <a(τ)b> if revere=false and <a b(τ)> if reverse=true
"""
function correlation_2op_1τ(h::Union{QuantumOperator, MPO}, a::Union{MPO, AdjointMPO}, b::Union{MPO, AdjointMPO}, state::AbstractMPS, times::Vector{<:Real};
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	times = -times
	reverse ? _unitary_tt_corr_a_bt(h, a, b, state, times, stepper) : _unitary_tt_corr_at_b(h, a, b, state, times, stepper)
end

"""
	gs_correlation_2op_1τ(h::Union{QuantumOperator, MPO}, a::MPO, b::MPO, state::MPS, times::Vector{<:Real}; kwargs...)
	ground state two imaginary time correlation
"""
function gs_correlation_2op_1τ(h::Union{QuantumOperator, MPO}, a::Union{MPO, AdjointMPO}, b::Union{MPO, AdjointMPO}, state::AbstractMPS, times::Vector{<:Real};
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), 
	gs_energy::Real = real(expectation(h, state)),
	reverse::Bool=false)
	times = -times
	reverse ? _gs_unitary_tt_corr_a_bt(h, a, b, state, gs_energy, times, stepper) : _gs_unitary_tt_corr_at_b(h, a, b, state, gs_energy, times, stepper)
end


# exact two-time correlations
function _exact_unitary_tt_corr_at_b(h, A::AdjointMPO, B::MPO, state, times)
	state_right = B * state
	state_left = copy(state)

	state_left = ExactMPS(state_left)
	state_right = ExactMPS(state_right)
	# (state_left.center == state_right.center) || error("something wrong.")
	cache_left = environments(h, state_left)
	cache_right = environments(h, state_right)
	# sr_hleft, sr_hright = init_h_center(h, state_right)
	# sl_hleft, sl_hright = init_h_center(h, state_left)

	result = scalartype(state)[]
	for i in 1:length(times)	
		tspan_right = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		tspan_left = _time_reversal(tspan_right)
		if abs(tspan_right[2] - tspan_right[1]) > 0.
			# state_left = _exact_timeevolution_util!(h, tspan_left[2]-tspan_left[1], state_left, sl_hleft, sl_hright, ishermitian=true)
			# state_right = _exact_timeevolution_util!(h, tspan_right[2]-tspan_right[1], state_right, sr_hleft, sr_hright, ishermitian=true)
			exact_timeevolution!(cache_left, tspan_left[2]-tspan_left[1], ishermitian=true)
			exact_timeevolution!(cache_right, tspan_right[2]-tspan_right[1], ishermitian=true)
		end
		push!(result, dot(A' * MPS(state_left), MPS(state_right)) / dim(space_r(state_right)) )
	end
	return result
end
_exact_unitary_tt_corr_at_b(h, A::MPO, B::AdjointMPO, state, times) = _exact_unitary_tt_corr_at_b(h, _to_a(A), _to_n(B), state, times)


function _exact_unitary_tt_corr_a_bt(h, A::AdjointMPO, B::MPO, state, times)
	state_right = copy(state)
	state_left = A' * state

	state_left = ExactMPS(state_left)
	state_right = ExactMPS(state_right)
	# (state_left.center == state_right.center) || error("something wrong.")
	# sr_hleft, sr_hright = init_h_center(h, state_right)
	# sl_hleft, sl_hright = init_h_center(h, state_left)
	cache_left = environments(h, state_left)
	cache_right = environments(h, state_right)

	result = scalartype(state)[]
	local cache_left, cache_right	
	for i in 1:length(times)	
		tspan_right = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		tspan_left = _time_reversal(tspan_right)
		if abs(tspan_right[2] - tspan_right[1]) > 0.
			# state_left = _exact_timeevolution_util!(h, tspan_left[2]-tspan_left[1], state_left, sl_hleft, sl_hright, ishermitian=true)
			# state_right = _exact_timeevolution_util!(h, tspan_right[2]-tspan_right[1], state_right, sr_hleft, sr_hright, ishermitian=true)
			exact_timeevolution!(cache_left, tspan_left[2]-tspan_left[1], ishermitian=true)
			exact_timeevolution!(cache_right, tspan_right[2]-tspan_right[1], ishermitian=true)
		end
		push!(result, expectation(state_left, B, state_right) / dim(space_r(state_left)) )
	end
	return result
end
_exact_unitary_tt_corr_a_bt(h, A::MPO, B::AdjointMPO, state, times) = _exact_unitary_tt_corr_a_bt(h, _to_a(A), _to_n(B), state, times)


# exact diagonalization, used for small systems or debug
function exact_correlation_2op_1t(h::MPO, a::Union{MPO, AdjointMPO}, b::Union{MPO, AdjointMPO}, state::AbstractMPS, times::Vector{<:Real}; reverse::Bool=false)
	if scalartype(state) <: Real
		state = complex(state)
	end
	times = -im .* times
	reverse ? _exact_unitary_tt_corr_a_bt(h, a, b, state, times) : _exact_unitary_tt_corr_at_b(h, a, b, state, times)
end
function exact_correlation_2op_1t(h::QuantumOperator, a::Union{MPO, AdjointMPO}, b::Union{MPO, AdjointMPO}, state::AbstractMPS, times::Vector{<:Real}; reverse::Bool=false)
	return exact_correlation_2op_1t(MPO(h), a, b, state, times, reverse=reverse)
end
function exact_correlation_2op_1τ(h::MPO, a::Union{MPO, AdjointMPO}, b::Union{MPO, AdjointMPO}, state::AbstractMPS, times::Vector{<:Real}; reverse::Bool=false)
	times = -times
	reverse ? _exact_unitary_tt_corr_a_bt(h, a, b, state, times) : _exact_unitary_tt_corr_at_b(h, a, b, state, times)
end
function exact_correlation_2op_1τ(h::QuantumOperator, a::Union{MPO, AdjointMPO}, b::Union{MPO, AdjointMPO}, state::AbstractMPS, times::Vector{<:Real}; reverse::Bool=false)
	return exact_correlation_2op_1τ(MPO(h), a, b, state, times, reverse=reverse)
end


