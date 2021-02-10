### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 1ad3e422-620c-11eb-1cfc-23098f1aa1f2
begin
	using HDF5
	include("Evaluation.jl"); import .Evaluation
end

# ╔═╡ f3391cb6-5c7a-11eb-1267-db40587a741e
begin
	datafile = h5open("data/dr16q_superset.hdf5", "r")
	y = read(datafile, "z_va")
	y_pca = read(datafile, "z_pca_va")
	y_pipe = read(datafile, "z_pipe_va")
	y_qn = read(datafile, "z_qn_va")
	close(datafile)
end

# ╔═╡ 49d4f36c-5c7e-11eb-2ee2-6737c5d9cb2c
Evaluation.rmse(y, y_pca), Evaluation.catastrophic_redshift_ratio(y, y_pca)

# ╔═╡ 6eb55366-5c7e-11eb-272b-a13c9b77744b
Evaluation.rmse(y, y_pipe), Evaluation.catastrophic_redshift_ratio(y, y_pipe)

# ╔═╡ 787ec2c4-5c7e-11eb-1b55-850c597ba857
Evaluation.rmse(y, y_qn), Evaluation.catastrophic_redshift_ratio(y, y_qn)

# ╔═╡ Cell order:
# ╠═1ad3e422-620c-11eb-1cfc-23098f1aa1f2
# ╠═f3391cb6-5c7a-11eb-1267-db40587a741e
# ╠═49d4f36c-5c7e-11eb-2ee2-6737c5d9cb2c
# ╠═6eb55366-5c7e-11eb-272b-a13c9b77744b
# ╠═787ec2c4-5c7e-11eb-1b55-850c597ba857
