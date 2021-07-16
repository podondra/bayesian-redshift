### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ bdb521fa-acd6-11eb-0d41-2d68a7abecb2
using DataFrames, HDF5, Printf, Random, Statistics, StatsBase, StatsPlots

# ╔═╡ 5c29de0c-4c3f-44ea-aea3-dbc0449a4a22
include("Evaluation.jl"); import .Evaluation

# ╔═╡ 58bb6e14-6261-45ff-b647-de2d63a4b129
include("Neural.jl"); import .Neural

# ╔═╡ 62fd8899-7fe0-4d54-9326-79008c60140b
include("Utils.jl"); import .Utils

# ╔═╡ fce2913d-6c91-492b-9b98-81f5c886c467
md"# Generalisation to DR16Q Superset"

# ╔═╡ 8c971f02-0dab-41d0-928b-7937052f7542
plotly()

# ╔═╡ a13527ac-4670-4a2f-a390-17700d707705
begin
	datafile = h5open("data/dr16q_superset.hdf5", "r")
	id = read(datafile, "id")
	n = size(id, 2)
	X = read(datafile, "X")
	df = DataFrame(
		plate=id[1, :],
		mjd=id[2, :],
		fiberid=id[3, :],
		z=read(datafile, "z"),
		source=read(datafile, "source_z"),
		z_pred=read(datafile, "z_pred"),
		z_pred_std=read(datafile, "z_pred_std"),
		z_10k = read(datafile, "z_10k"),
		pipe_corr_10k = read(datafile, "pipe_corr_10k"),
		z_pca = read(datafile, "z_pca"),
		z_pipe = read(datafile, "z_pipe"),
		z_vi=read(datafile, "z_vi"),
		z_qn=read(datafile, "z_qn"),
		entropy=read(datafile, "entropy"),
		entropy_std=read(datafile, "entropy_std"),
		mutual_information=read(datafile, "mutual_information"),
		variation_ratio=read(datafile, "variation_ratio"),
		sn=read(datafile, "sn_median_all"))
	zs_pred = read(datafile, "zs_pred")
	close(datafile)
	df
end

# ╔═╡ d1770126-cb47-47ae-844a-268210927dfb
begin
	idx_z = df.z .> -1
	density(df.z[idx_z], label="Reference", xlabel="z", ylabel="Density")
	density!(df.z_pred[idx_z], label="MC Dropout")
end

# ╔═╡ b6bc407f-2fd6-4739-ae1d-da96ff984526
histogram(df.sn, xlabel="S/N", ylabel="Count")

# ╔═╡ 076b15f3-12a7-4151-a52d-682edbb5dc7d
function preview_idx(i)
	Utils.plot_spectrum(X[:, i], legend=:none,
		title=@sprintf(
			"z = %.3f; source = %s; ẑ = %.2f; E = %.1f",
			df[i, :z], df[i, :source], df[i, :z_pred], df[i, :entropy]))
	Utils.plot_spectral_lines!(df[i, :z])
	Utils.plot_spectral_lines!(df[i, :z_pred], color=:red, location=:bottom)
end

# ╔═╡ 3a28c77a-5e9f-4ea0-9cb5-78dc32f2ad40
md"## Correct Prediction"

# ╔═╡ fc1e339e-9043-4336-aa50-91408025d330
begin
	Δv = Evaluation.compute_Δv(df.z, df.z_pred)
	i_rnd = rand((1:n)[(Δv .< 3000) .& (df.sn .> 50) .& (df.z_pred .> 0.1)])
	preview_idx(i_rnd)
end

# ╔═╡ de5b8936-7b64-43e0-8ab5-057da3f015fc
md"## $z > 6.445$"

# ╔═╡ 66851320-f094-4b66-a006-c9cfccc1a816
begin
	idx_high_z = df.z .> 6.445
	countmap(df.source[idx_high_z])
end

# ╔═╡ 2f6d398e-5a9f-4f23-82a0-274603456444
preview_idx(rand((1:n)[idx_high_z]))

# ╔═╡ 44c86007-8ff8-4d29-b96f-4490d5e1b8fb
begin
	idx_vi = df.source .== "VI"
	first_vi, second_vi = (1:n)[idx_high_z .& idx_vi]
	id[:, idx_high_z .& idx_vi], df.entropy[idx_high_z .& idx_vi]
end

# ╔═╡ 85c11afd-00ed-4bb2-bd7a-5522d2b40132
preview_idx(first_vi)

# ╔═╡ 2de2465a-4470-4afa-94eb-785e8df97752
preview_idx(second_vi)

# ╔═╡ dca014ef-0caf-4af3-96ef-10215164fdf0
begin
	# seed from random.org
	Random.seed!(49)
    idx_rnd_high_z = rand((1:n)[idx_high_z .& .~idx_vi])
	id[:, idx_rnd_high_z]
end

# ╔═╡ 222d621a-6078-4498-8594-d30455ec01c0
preview_idx(idx_rnd_high_z)

# ╔═╡ b662295b-9bfd-4765-b20c-bd9185acc7e6
md"## Random Visual Inspection of 10k Spectra"

# ╔═╡ 53e40953-7e3f-44ea-a629-7dca5d1834b1
begin
	idx_10k = df.z_10k .> -1
	idx_10k_all = df.pipe_corr_10k .>= 0 
	sum(idx_10k), sum(idx_10k_all)
end

# ╔═╡ 532e5b27-4595-49c6-a7b7-ad044fe8e62b
begin
	idx_10k_missing = df.z_10k .<= -1
	sum(idx_10k_all .& idx_10k_missing)
end

# ╔═╡ af07e531-cec1-4ee2-bb9c-916a8b038a7d
countmap(df.z_10k[idx_10k_all .& idx_10k_missing])

# ╔═╡ bda05a89-a68f-4067-8224-8994ee6943d1
median(df.entropy[idx_10k]), median(df.entropy[idx_10k_all .& idx_10k_missing])

# ╔═╡ 6e4b619e-ffad-4277-a6ef-5ba6ebd1bef0
begin
	z_10k = df.z_10k[idx_10k]
	z_pred = df.z_pred[idx_10k]
	z_pipe = df.z_pipe[idx_10k]
	z_pca = df.z_pca[idx_10k]
	z_qn = df.z_qn[idx_10k]
end

# ╔═╡ 0211a7d5-f744-4331-a55c-6860716c2109
Evaluation.rmse(z_10k, z_pred), Evaluation.median_Δv(z_10k, z_pred), Evaluation.cat_z_ratio(z_10k, z_pred)

# ╔═╡ cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
Evaluation.rmse(z_10k, z_pipe), Evaluation.median_Δv(z_10k, z_pipe), Evaluation.cat_z_ratio(z_10k, z_pipe)

# ╔═╡ dc462c4d-ec37-4459-87e6-87428f2229da
Evaluation.rmse(z_10k, z_pca), Evaluation.median_Δv(z_10k, z_pca), Evaluation.cat_z_ratio(z_10k, z_pca)

# ╔═╡ b69e11ff-1249-43fb-a423-61efa0945030
Evaluation.rmse(z_10k, z_qn), Evaluation.median_Δv(z_10k, z_qn), Evaluation.cat_z_ratio(z_10k, z_qn)

# ╔═╡ 573e6e02-a2a9-457f-b2fd-4acdd921c090
md"## Suggestions of Redshifts"

# ╔═╡ a3db9fd5-a9a1-46c3-9097-7573d48bc5df
zs_pred

# ╔═╡ c84f8e62-c1e4-4ea8-b69d-8d51175f10e3
begin
	i_sug = 54463
	id[:, i_sug], df.source[i_sug]
end

# ╔═╡ b7e9879d-8c41-4130-bbc6-e14ba62b8f0e
countmap(zs_pred[:, i_sug])

# ╔═╡ b8c77084-6ee8-4615-96c7-deb317614e0c
begin
	title = @sprintf(
		"z = %.3f; source = %s ẑ = %.2f; E = %.1f",
		df[i_sug, :z], df[i_sug, :source], df[i_sug, :z_pred], df[i_sug, :entropy])
	Utils.plot_spectrum(X[:, i_sug], legend=:none, title=title)
	Utils.plot_spectral_lines!(0.08)
end

# ╔═╡ 197b8de6-f7f5-4701-8e9e-220b781a0c1e
md"## On-Edge Predictions"

# ╔═╡ 94b7dc28-36d8-4a00-92fe-a7e1d65afdb0
begin
	idx_edge = zeros(Bool, n)
	for k in 1:n
		set = collect(Set((zs_pred[:, k])))
		if length(set) == 2
			idx_edge[k] = abs(set[1] - set[2]) <= 0.015
		end
	end
	sum(idx_edge)
end

# ╔═╡ 77108e12-ad9e-418c-bd02-194cb5a891c4
begin
	histogram(df.entropy, xlabel="Entropy", label="All")
	histogram!(df.entropy[idx_edge], label="On-Edge")
	histogram!(df.entropy[.~idx_edge], label="Without On-Edge")
end

# ╔═╡ e66137f7-2233-49bb-a6ed-a4aac6a3a319
sum(idx_edge .& idx_10k)

# ╔═╡ bc19f08c-839d-484c-98d0-4842a64799ee
md"## Spectra with the Highest Entropy"

# ╔═╡ 6648d01b-ce06-4fda-b339-022f68266bcc
begin
	entropy = df[:, :entropy]
	entropy[idx_edge] .= 0
	histogram(entropy)
end

# ╔═╡ bb0c2182-7309-4cf2-85f9-7462d41d4b22
begin
	i_high_entr = sortperm(entropy)[end - 2:end]
	id[:, i_high_entr]
end

# ╔═╡ c75d0c17-6d3e-4c69-8abf-3ddae4d1e35f
preview_idx(i_high_entr[1])

# ╔═╡ dfe23e93-904b-481f-b247-158637cd361e
preview_idx(i_high_entr[2])

# ╔═╡ 75767b4a-cfc6-45e0-a270-5f237f135bed
preview_idx(i_high_entr[3])

# ╔═╡ eb11930b-8f4f-4301-a462-a41fa54d980f
md"## Utilisation of Uncertainties"

# ╔═╡ 08a0f0ac-731c-41f5-909b-3b5d920567ec
begin
	entropy_10k = entropy[idx_10k]
	histogram(entropy_10k)
end

# ╔═╡ 24cb3a20-9f22-424a-bba8-3dca7fac410d
begin
	z_pred_std = df.z_pred_std[idx_10k]
	entropy_std = df.entropy_std
	histogram(entropy_std)
end

# ╔═╡ fa971ce4-7e96-4aab-9db5-392f0d5a1dfc
begin
	entropy_10k_std = entropy_std[idx_10k]
	histogram(entropy_10k_std)
end

# ╔═╡ d2c25b73-2a9d-40db-a439-98599e80a33c
begin
	ts = 0.001:0.01:maximum(entropy)

	coverages = [sum(entropy .< t) / n for t in ts]
	coverages_std = [sum(entropy_std .< t) / n for t in ts]

	cat_zs = 100 .* [Evaluation.cat_z_ratio(
			z_10k[entropy_10k .< t],
			z_pred[entropy_10k .< t])
		for t in ts]
	cat_zs_std = 100 .* [Evaluation.cat_z_ratio(
			z_10k[entropy_10k_std .< t],
			z_pred_std[entropy_10k_std .< t])
		for t in ts]

	plot(ts, coverages, ylabel="Coverage", label="MC Dropout")
	plot_coverages = plot!(ts, coverages_std, label="Std. Dropout")

	plot(ts, cat_zs, ylabel="Est. Cat. z Ratio", label="MC Dropout")
	plot_cat_zs = plot!(ts, cat_zs_std, xlabel="Threshold", label="Std. Dropout")

	plot(plot_coverages, plot_cat_zs, layout=@layout [a; b])
end

# ╔═╡ 01a4659d-4b10-4532-8086-0bf22fbf4825
n * 0.01, n * 0.05, n * 0.1

# ╔═╡ Cell order:
# ╟─fce2913d-6c91-492b-9b98-81f5c886c467
# ╠═bdb521fa-acd6-11eb-0d41-2d68a7abecb2
# ╠═8c971f02-0dab-41d0-928b-7937052f7542
# ╠═5c29de0c-4c3f-44ea-aea3-dbc0449a4a22
# ╠═58bb6e14-6261-45ff-b647-de2d63a4b129
# ╠═62fd8899-7fe0-4d54-9326-79008c60140b
# ╠═a13527ac-4670-4a2f-a390-17700d707705
# ╠═d1770126-cb47-47ae-844a-268210927dfb
# ╠═b6bc407f-2fd6-4739-ae1d-da96ff984526
# ╠═076b15f3-12a7-4151-a52d-682edbb5dc7d
# ╟─3a28c77a-5e9f-4ea0-9cb5-78dc32f2ad40
# ╠═fc1e339e-9043-4336-aa50-91408025d330
# ╟─de5b8936-7b64-43e0-8ab5-057da3f015fc
# ╠═66851320-f094-4b66-a006-c9cfccc1a816
# ╠═2f6d398e-5a9f-4f23-82a0-274603456444
# ╠═44c86007-8ff8-4d29-b96f-4490d5e1b8fb
# ╠═85c11afd-00ed-4bb2-bd7a-5522d2b40132
# ╠═2de2465a-4470-4afa-94eb-785e8df97752
# ╠═dca014ef-0caf-4af3-96ef-10215164fdf0
# ╠═222d621a-6078-4498-8594-d30455ec01c0
# ╟─b662295b-9bfd-4765-b20c-bd9185acc7e6
# ╠═53e40953-7e3f-44ea-a629-7dca5d1834b1
# ╠═532e5b27-4595-49c6-a7b7-ad044fe8e62b
# ╠═af07e531-cec1-4ee2-bb9c-916a8b038a7d
# ╠═bda05a89-a68f-4067-8224-8994ee6943d1
# ╠═6e4b619e-ffad-4277-a6ef-5ba6ebd1bef0
# ╠═0211a7d5-f744-4331-a55c-6860716c2109
# ╠═cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
# ╠═dc462c4d-ec37-4459-87e6-87428f2229da
# ╠═b69e11ff-1249-43fb-a423-61efa0945030
# ╟─573e6e02-a2a9-457f-b2fd-4acdd921c090
# ╠═a3db9fd5-a9a1-46c3-9097-7573d48bc5df
# ╠═c84f8e62-c1e4-4ea8-b69d-8d51175f10e3
# ╠═b7e9879d-8c41-4130-bbc6-e14ba62b8f0e
# ╠═b8c77084-6ee8-4615-96c7-deb317614e0c
# ╟─197b8de6-f7f5-4701-8e9e-220b781a0c1e
# ╠═94b7dc28-36d8-4a00-92fe-a7e1d65afdb0
# ╠═77108e12-ad9e-418c-bd02-194cb5a891c4
# ╠═e66137f7-2233-49bb-a6ed-a4aac6a3a319
# ╟─bc19f08c-839d-484c-98d0-4842a64799ee
# ╠═6648d01b-ce06-4fda-b339-022f68266bcc
# ╠═bb0c2182-7309-4cf2-85f9-7462d41d4b22
# ╠═c75d0c17-6d3e-4c69-8abf-3ddae4d1e35f
# ╠═dfe23e93-904b-481f-b247-158637cd361e
# ╠═75767b4a-cfc6-45e0-a270-5f237f135bed
# ╟─eb11930b-8f4f-4301-a462-a41fa54d980f
# ╠═08a0f0ac-731c-41f5-909b-3b5d920567ec
# ╠═24cb3a20-9f22-424a-bba8-3dca7fac410d
# ╠═fa971ce4-7e96-4aab-9db5-392f0d5a1dfc
# ╠═d2c25b73-2a9d-40db-a439-98599e80a33c
# ╠═01a4659d-4b10-4532-8086-0bf22fbf4825
