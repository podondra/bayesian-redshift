### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ bdb521fa-acd6-11eb-0d41-2d68a7abecb2
using BSON, DataFrames, Flux, HDF5, Printf, Statistics, StatsBase, StatsPlots

# ╔═╡ 5c29de0c-4c3f-44ea-aea3-dbc0449a4a22
include("Evaluation.jl"); import .Evaluation

# ╔═╡ 58bb6e14-6261-45ff-b647-de2d63a4b129
include("Neural.jl"); import .Neural

# ╔═╡ 62fd8899-7fe0-4d54-9326-79008c60140b
include("Utils.jl"); import .Utils

# ╔═╡ fce2913d-6c91-492b-9b98-81f5c886c467
md"# SDSS DR16Q Superset Evaluation"

# ╔═╡ 330c18c3-3aba-48d0-ae9c-b3b859514235
Core.eval(Main, :(import Flux, NNlib))

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
		z_10k = read(datafile, "z_10k"),
		pipe_corr_10k = read(datafile, "pipe_corr_10k"),
		z_pca = read(datafile, "z_pca"),
		z_pipe = read(datafile, "z_pipe"),
		z_vi=read(datafile, "z_vi"),
		z_qn=read(datafile, "z_qn"),
		entropy=read(datafile, "entropy"),
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

# ╔═╡ 076b15f3-12a7-4151-a52d-682edbb5dc7d
function preview_idx(i)
	Utils.plot_spectrum(X[:, i], legend=:none,
		title=@sprintf(
			"z = %.3f; source = %s; ẑ = %.3f; E = %.2f",
			df[i, :z], df[i, :source], df[i, :z_pred], df[i, :entropy]))
	Utils.plot_spectral_lines!(df[i, :z])
	Utils.plot_spectral_lines!(df[i, :z_pred], color=:red, location=:bottom)
end

# ╔═╡ de5b8936-7b64-43e0-8ab5-057da3f015fc
md"## $z > 6.44$"

# ╔═╡ 66851320-f094-4b66-a006-c9cfccc1a816
begin
	idx_high_z = df.z .> 6.44
	sum(idx_high_z)
end

# ╔═╡ 2f6d398e-5a9f-4f23-82a0-274603456444
preview_idx(rand((1:n)[idx_high_z]))

# ╔═╡ 22b4f5bb-ee29-4ac8-9d9e-585e0a5151ca
countmap(df.source[idx_high_z])

# ╔═╡ 78795ab0-7dae-42a6-accd-d2d468a7099c
begin
	idx_vi = df.source .== "VI"
	sum(idx_vi)
end

# ╔═╡ 5dea6670-b950-409f-adba-ce696dd99199
sum(idx_high_z .& idx_vi)

# ╔═╡ 2c8fdfce-19f4-44cd-a43a-433a87ae962d
df.entropy[idx_high_z .& idx_vi]

# ╔═╡ be1213c3-01ac-47d8-b9d3-c80d70f6457f
preview_idx(rand((1:n)[idx_high_z .& idx_vi]))

# ╔═╡ 7127b6f3-b4ec-4560-b7c2-e7e0261276cb
md"## $\hat{z} = 0$"

# ╔═╡ 1efcba1d-5e1e-4b12-939f-c7431cc3c0e5
begin
	idx_z_pred_zero = df.z_pred .< 0.005
	idx_z_zero = df.z .< 0.005
	sum(idx_z_pred_zero), sum(idx_z_zero)
end

# ╔═╡ 8bc93fba-f600-4086-8f2f-393138246c67
preview_idx(rand((1:n)[idx_z_pred_zero .& .~idx_z_zero]))

# ╔═╡ b662295b-9bfd-4765-b20c-bd9185acc7e6
md"## Random Visual Inspection of 10k Spectra"

# ╔═╡ 53e40953-7e3f-44ea-a629-7dca5d1834b1
begin
	idx_10k = df.pipe_corr_10k .>= 0
	n_10k = sum(idx_10k)
	n_10k
end

# ╔═╡ 8c541607-4211-4338-a7b2-30cbe4be0c53
begin
	pipe_corr_10k = df.pipe_corr_10k[idx_10k]
	z_pred = df.z_pred[idx_10k]
	z_pipe = df.z_pipe[idx_10k]

	z_10k = df.z_10k[idx_10k]
	# the two spectra has -999 so overflow to negative number
	z_10k[z_10k .== -999.0] .= -1
	idx_missing = z_10k .<= -1

	# suppose we predict correctly those missing
	#z_10k[idx_missing] = z_pred[idx_missing]

	Δv = Evaluation.compute_Δv(z_10k, z_pred)
	X_10k = X[:, idx_10k]
end

# ╔═╡ 71fa953c-2a6d-4002-8529-7672730a655e
begin
	idx_failure = (Δv .>= 3000.0)
	idx_error = idx_missing .| idx_failure
	sum(idx_missing), sum(idx_failure), sum(idx_error)
end

# ╔═╡ 101e78ea-98a2-452d-bed3-a41f544c8e49
1 - sum(pipe_corr_10k) / n_10k

# ╔═╡ 0211a7d5-f744-4331-a55c-6860716c2109
Evaluation.cat_z_ratio(z_10k, z_pred)

# ╔═╡ cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
Evaluation.cat_z_ratio(z_10k, z_pipe)

# ╔═╡ dc462c4d-ec37-4459-87e6-87428f2229da
Evaluation.cat_z_ratio(z_10k, df.z_pca[idx_10k])

# ╔═╡ b69e11ff-1249-43fb-a423-61efa0945030
Evaluation.cat_z_ratio(z_10k, df.z_qn[idx_10k])

# ╔═╡ c6da5788-24fc-4cb0-add4-339965aa365c
sum(abs.(z_pipe[idx_failure] .- z_pred[idx_failure]) .> 0.05)

# ╔═╡ 197b8de6-f7f5-4701-8e9e-220b781a0c1e
md"## On Edge Predictions

Due to binning, there can be a situation where the redshift is on the edge.
Therefore, the model is not sure into which bin to put its redshift.
But, we can filter it."

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
	histogram!(df.entropy[.~idx_edge], label="W\\out On-Edge")
end

# ╔═╡ eb11930b-8f4f-4301-a462-a41fa54d980f
md"## Utilisation of Uncertainties"

# ╔═╡ 08a0f0ac-731c-41f5-909b-3b5d920567ec
begin
	entropy = df[:, :entropy]
	#entropy[idx_edge] .= 0
	entropy = entropy[idx_10k]
end

# ╔═╡ afc55a96-2c39-4155-835a-8d3f21840f87
function preview_10k_idx(i)
	Utils.plot_spectrum(X_10k[:, i], legend=:none,
		title=@sprintf(
			"z_10k = %.3f; z_pipe = %.3f; ẑ = %.2f; E = %.2f",
			z_10k[i], z_pipe[i], z_pred[i], entropy[i]))
	Utils.plot_spectral_lines!(z_10k[i])
	Utils.plot_spectral_lines!(z_pred[i], color=:red, location=:bottom)
end

# ╔═╡ 588c4f37-1691-4dcc-805d-9c169b04f1f0
preview_10k_idx(rand((1:n_10k)[idx_error]))

# ╔═╡ ad6b8a37-e5a7-4fe0-b63a-5ab1e2b600fa
preview_10k_idx(rand((1:n_10k)[idx_missing]))

# ╔═╡ 24cb3a20-9f22-424a-bba8-3dca7fac410d
begin
	model = BSON.load("models/classification_model.bson")[:model] |> gpu
	p = softmax(Neural.forward_pass(model, X |> gpu))
	std_entropy = p .* log.(p)
	std_entropy[isnan.(std_entropy)] .= 0
	std_entropy = dropdims(-sum(std_entropy, dims=1), dims=1)
end

# ╔═╡ ed607b71-b6c5-4175-a375-05278a8f3c72
histogram(std_entropy)

# ╔═╡ 1a4c427e-54db-4f27-9c60-c5f1c0459d37
n, 0.01 * n, 0.05 * n, 0.1 * n

# ╔═╡ 35d20cf4-61bc-44d5-8299-51e9b62b26d5
1 - (8581 / n)

# ╔═╡ d2c25b73-2a9d-40db-a439-98599e80a33c
begin
	thresholds = 0.001:0.01:maximum(df.entropy)
	coverages = [sum(df.entropy .< t) / n for t in thresholds]
	std_coverages = [sum(std_entropy .< t) / n for t in thresholds]
	cat_zs = 100 .* [
		Evaluation.cat_z_ratio(z_10k[entropy .< t],	z_pred[entropy .< t])
		for t in thresholds]
	std_cat_zs = 100 .* [
		Evaluation.cat_z_ratio(
			z_10k[std_entropy[idx_10k] .< t],
			z_pred[std_entropy[idx_10k] .< t])
		for t in thresholds]
	plot(thresholds, coverages, ylabel="Coverage", label="MC Dropout")
	plot_coverages = plot!(thresholds, std_coverages, label="Std Dropout")
	plot(thresholds, cat_zs, ylabel="Est. Cat. z Ratio", xlabel="Threshold", label="MC Dropout")
	plot_cat_zs = plot!(thresholds, std_cat_zs, label="Std Dropout")
	plot(plot_coverages, plot_cat_zs, layout=@layout [a; b])
end

# ╔═╡ 207a7bfe-c517-4fdf-b6a2-9da6a514e52d
histogram(df.entropy[idx_10k], xlabel="Entropy", legend=:none)

# ╔═╡ 8988a49e-e8ec-48de-b750-3106cfa4b0c1
md"## Signal to Noise"

# ╔═╡ 9085c100-8733-42a0-acde-f08864868eb0
histogram(df.sn)

# ╔═╡ 4efb1a5b-cb47-43dc-83c2-dc89a321624b
cor(df.entropy, df.sn)

# ╔═╡ 7838a5a5-a0b9-4c1b-a574-81f40d36e26f
marginalhist(df.entropy, df.sn)

# ╔═╡ 7fd52f8b-98b4-4229-9e48-294f4b762ee0
Δv_all = Evaluation.compute_Δv(df.z, df.z_pred)

# ╔═╡ 64b80d5c-baf5-4118-a72a-d0031d548a31
cor(df.entropy[df.z .> 0.0], Δv_all[df.z .> 0.0])

# ╔═╡ 759b4f08-ba9d-42f7-a89b-691beadb58d6
Δv_all[Δv_all .< 0], df.z[Δv_all .< 0]

# ╔═╡ 24cd4ec3-6a7a-4664-a4fb-ca9bbac3ed5c
marginalhist(Δv_all, df.sn)

# ╔═╡ Cell order:
# ╟─fce2913d-6c91-492b-9b98-81f5c886c467
# ╠═bdb521fa-acd6-11eb-0d41-2d68a7abecb2
# ╠═330c18c3-3aba-48d0-ae9c-b3b859514235
# ╠═8c971f02-0dab-41d0-928b-7937052f7542
# ╠═5c29de0c-4c3f-44ea-aea3-dbc0449a4a22
# ╠═58bb6e14-6261-45ff-b647-de2d63a4b129
# ╠═62fd8899-7fe0-4d54-9326-79008c60140b
# ╠═a13527ac-4670-4a2f-a390-17700d707705
# ╠═d1770126-cb47-47ae-844a-268210927dfb
# ╠═076b15f3-12a7-4151-a52d-682edbb5dc7d
# ╟─de5b8936-7b64-43e0-8ab5-057da3f015fc
# ╠═66851320-f094-4b66-a006-c9cfccc1a816
# ╠═2f6d398e-5a9f-4f23-82a0-274603456444
# ╠═22b4f5bb-ee29-4ac8-9d9e-585e0a5151ca
# ╠═78795ab0-7dae-42a6-accd-d2d468a7099c
# ╠═5dea6670-b950-409f-adba-ce696dd99199
# ╠═2c8fdfce-19f4-44cd-a43a-433a87ae962d
# ╠═be1213c3-01ac-47d8-b9d3-c80d70f6457f
# ╟─7127b6f3-b4ec-4560-b7c2-e7e0261276cb
# ╠═1efcba1d-5e1e-4b12-939f-c7431cc3c0e5
# ╠═8bc93fba-f600-4086-8f2f-393138246c67
# ╟─b662295b-9bfd-4765-b20c-bd9185acc7e6
# ╠═53e40953-7e3f-44ea-a629-7dca5d1834b1
# ╠═8c541607-4211-4338-a7b2-30cbe4be0c53
# ╠═71fa953c-2a6d-4002-8529-7672730a655e
# ╠═101e78ea-98a2-452d-bed3-a41f544c8e49
# ╠═0211a7d5-f744-4331-a55c-6860716c2109
# ╠═cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
# ╠═dc462c4d-ec37-4459-87e6-87428f2229da
# ╠═b69e11ff-1249-43fb-a423-61efa0945030
# ╠═afc55a96-2c39-4155-835a-8d3f21840f87
# ╠═588c4f37-1691-4dcc-805d-9c169b04f1f0
# ╠═ad6b8a37-e5a7-4fe0-b63a-5ab1e2b600fa
# ╠═c6da5788-24fc-4cb0-add4-339965aa365c
# ╟─197b8de6-f7f5-4701-8e9e-220b781a0c1e
# ╠═94b7dc28-36d8-4a00-92fe-a7e1d65afdb0
# ╠═77108e12-ad9e-418c-bd02-194cb5a891c4
# ╟─eb11930b-8f4f-4301-a462-a41fa54d980f
# ╠═08a0f0ac-731c-41f5-909b-3b5d920567ec
# ╠═24cb3a20-9f22-424a-bba8-3dca7fac410d
# ╠═ed607b71-b6c5-4175-a375-05278a8f3c72
# ╠═1a4c427e-54db-4f27-9c60-c5f1c0459d37
# ╠═35d20cf4-61bc-44d5-8299-51e9b62b26d5
# ╠═d2c25b73-2a9d-40db-a439-98599e80a33c
# ╠═207a7bfe-c517-4fdf-b6a2-9da6a514e52d
# ╟─8988a49e-e8ec-48de-b750-3106cfa4b0c1
# ╠═9085c100-8733-42a0-acde-f08864868eb0
# ╠═4efb1a5b-cb47-43dc-83c2-dc89a321624b
# ╠═7838a5a5-a0b9-4c1b-a574-81f40d36e26f
# ╠═7fd52f8b-98b4-4229-9e48-294f4b762ee0
# ╠═64b80d5c-baf5-4118-a72a-d0031d548a31
# ╠═759b4f08-ba9d-42f7-a89b-691beadb58d6
# ╠═24cd4ec3-6a7a-4664-a4fb-ca9bbac3ed5c
