println("------------------------------------")
println("----------|    TEBD    |------------")
println("------------------------------------")


@testset "TEBD: comparison with exact evolution" begin
	J = 1.
	J2 = 1.2
	U = 0.7

	dt = 0.01
	tebd_sweeps = 20
	for L in (3, 4)
		ham, observers = hubbard_ladder(L, J, J2, U, spinal_fermion_site_ops_u1_u1())
		mpo = MPO(ham)
		state = initial_state_u1_u1(ComplexF64, L)
		canonicalize!(state)
		# exact evolution
		state1 = exact_timeevolution(mpo, -im*tebd_sweeps * dt, ExactMPS(state), ishermitian=true)
		obs1 = real([expectation(item, MPS(state1), iscanonical=false) for item in observers])

		# tebd
		circuit = trotter_propagator(ham, (0., -im * tebd_sweeps*dt), stepsize=dt, order=4)
		state2 = apply!(circuit, copy(state), trunc=truncdimcutoff(D=100, ϵ=1.0e-8))
		obs2 = real([expectation(item, state2, iscanonical=true) for item in observers])
		@test max_error(obs2, obs1) < 1.0e-6

		# 
		state3 = apply!(fuse_gates(circuit), copy(state), trunc=truncdimcutoff(D=100, ϵ=1.0e-8))
		obs3 = real([expectation(item, state3, iscanonical=true) for item in observers])
		@test max_error(obs3, obs1) < 1.0e-6

		# higher symmetry
		ham, observers = hubbard_ladder(L, J, J2, U, spinal_fermion_site_ops_u1_su2())
		state = initial_state_u1_su2(ComplexF64, L)
		circuit = fuse_gates(trotter_propagator(ham, (0., -im * tebd_sweeps*dt), stepsize=dt, order=4))
		state4 = apply!(circuit, copy(state), trunc=truncdimcutoff(D=100, ϵ=1.0e-8))
		obs4 = real([expectation(item, state4, iscanonical=true) for item in observers])
		@test max_error(obs4, obs1) < 1.0e-6
	end
	
end

