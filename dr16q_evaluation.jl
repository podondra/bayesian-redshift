### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ bdb521fa-acd6-11eb-0d41-2d68a7abecb2
begin
	using BSON, CUDA, Flux, HDF5, Printf, StatsPlots
	include("Evaluation.jl"); using .Evaluation
	include("Neural.jl"); using .Neural
	include("Utils.jl"); using .Utils
	CUDA.versioninfo()
end

# ╔═╡ a13527ac-4670-4a2f-a390-17700d707705
begin
	dr16q_file = h5open("data/dr16q_superset.hdf5", "r")
	id_va = read(dr16q_file, "id_va")
	X_va = read(dr16q_file, "X_va") |> gpu
	y_va = read(dr16q_file, "z_va")
	y_10k = read(dr16q_file, "z_10k_va")
	y_pca = read(dr16q_file, "z_pca_va")
	y_pipe = read(dr16q_file, "z_pipe_va")
	y_qn = read(dr16q_file, "z_qn_va")
	y_vi = read(dr16q_file, "z_vi_va")
	source = read(dr16q_file, "source_z_va")
	size(X_va), size(y_va)
end

# ╔═╡ dc462c4d-ec37-4459-87e6-87428f2229da
Evaluation.rmse(y_va, y_pca), Evaluation.cat_z_ratio(y_va, y_pca)

# ╔═╡ cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
Evaluation.rmse(y_va[y_qn .> -1], y_qn[y_qn .> -1]), Evaluation.cat_z_ratio(y_va[y_qn .> -1], y_qn[y_qn .> -1])

# ╔═╡ 475552fc-2ef1-4614-9b61-d2c872d8bf11
begin
	density(y_va, label="Reference", xlabel="z", ylabel="Density")
	density!(y_qn[y_qn .> -1], label="QuasarNet")
end

# ╔═╡ a0e071cd-7f06-4533-b9d4-95665d2f28df
begin
	Core.eval(Main, :(import Flux, NNlib))
	model = BSON.load("models/classification_model.bson")[:model] |> gpu
	ŷ_va = Neural.classify(model, X_va) |> cpu
	model
end

# ╔═╡ 0211a7d5-f744-4331-a55c-6860716c2109
Evaluation.rmse(y_va, ŷ_va), Evaluation.cat_z_ratio(y_va, ŷ_va)

# ╔═╡ d1770126-cb47-47ae-844a-268210927dfb
begin
	density(y_va, label="Reference", xlabel="z", ylabel="Density")
	density!(ŷ_va, label="Model")
end

# ╔═╡ 61f99ee5-46c0-403a-8fca-ab33f491fad8
begin
	Δv = Evaluation.compute_delta_v(y_va, ŷ_va)
	# random
	j = rand(1:size(id_va, 2))
	# cat. z
	#j = rand((1:size(id_va, 2))[Δv .> 3000])
	# absolute error
	#j = sortperm(abs.(y_va - ŷ_va))[end]

	title = @sprintf(
		"z = %.3f; ẑ = %.2f; Δv = %.3f; source: %s",
		y_va[j], ŷ_va[j], Δv[j], source[j])
	Utils.plot_spectrum(X_va[:, j], title=title, legend=:none)
	Utils.plot_spectral_lines!(y_va[j])
	prep_spec = Utils.plot_spectral_lines!(ŷ_va[j], color=:red, location=:bottom)
	loglam, flux = Utils.get_spectrum("DR16Q_Superset_v3", id_va[:, j]...)
	orig_spec = plot(10 .^ loglam, flux, legend=:none)
	plot(prep_spec, orig_spec, layout=@layout [a; b])
end

# ╔═╡ Cell order:
# ╠═bdb521fa-acd6-11eb-0d41-2d68a7abecb2
# ╠═a13527ac-4670-4a2f-a390-17700d707705
# ╠═dc462c4d-ec37-4459-87e6-87428f2229da
# ╠═cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
# ╠═0211a7d5-f744-4331-a55c-6860716c2109
# ╠═475552fc-2ef1-4614-9b61-d2c872d8bf11
# ╠═a0e071cd-7f06-4533-b9d4-95665d2f28df
# ╠═d1770126-cb47-47ae-844a-268210927dfb
# ╠═61f99ee5-46c0-403a-8fca-ab33f491fad8
