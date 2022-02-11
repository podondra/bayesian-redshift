### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ bdb521fa-acd6-11eb-0d41-2d68a7abecb2
begin
	using CSV, DataFrames, HDF5, Printf, Random, Statistics, StatsBase, StatsPlots
	include("Evaluation.jl")
	include("Neural.jl")
	include("Utils.jl")
	import .Evaluation, .Neural, .Utils
end

# ╔═╡ fce2913d-6c91-492b-9b98-81f5c886c467
md"# Generalisation to DR16Q Superset"

# ╔═╡ a13527ac-4670-4a2f-a390-17700d707705
begin
	superset = h5open("data/dr16q_superset.hdf5", "r")
	id = read(superset, "id")
	n = size(id, 2)
	X = read(superset, "X")
	df_orig = DataFrame(
		plate=id[1, :],	mjd=id[2, :], fiberid=id[3, :],
		ẑ=read(superset, "z_pred"),
		ẑ_std=read(superset, "z_pred_std"),
		entropy=read(superset, "entropy"),
		entropy_std=read(superset, "entropy_std"),
		mutual_information=read(superset, "mutual_information"),
		variation_ratio=read(superset, "variation_ratio"),
		z=read(superset, "z"),
		source_z=read(superset, "source_z"),
		z_vi=read(superset, "z_vi"),
		z_pipe=read(superset, "z_pipe"),
		zwarning=read(superset, "zwarning"),
		z_dr12q=read(superset, "z_dr12q"),
		z_dr7q_sch=read(superset, "z_dr7q_sch"),
		z_dr6q_hw=read(superset, "z_dr6q_hw"),
		z_10k=read(superset, "z_10k"),
		pipe_corr_10k=read(superset, "pipe_corr_10k"),
		z_pca=read(superset, "z_pca"),
		z_qn=read(superset, "z_qn"),
		sn=read(superset, "sn_median_all"),
		is_qso_final=read(superset, "is_qso_final"),
		idx_10k=read(superset, "idx_10k"))
	ẑs = Float32.(read(superset, "zs_pred"))
	close(superset)
	df_orig
end

# ╔═╡ 649c5981-e53b-4258-80cf-946c9b7b7a28
function preview(df, i)
	spec = df[i, :]
	label = @sprintf("spec-%04d-%5d-%04d.fits", spec.plate, spec.mjd, spec.fiberid)
	title = @sprintf(
		"z = %.3f; source = %s; ẑ = %.2f; E = %.1f; QSO = %d",
		spec.z, spec.source_z, spec.ẑ, spec.entropy, spec.is_qso_final)
	Utils.plot_spectrum(X[:, i], legend=:bottomright, label=label, title=title)
	Utils.plot_spectral_lines!(spec.z)
	Utils.plot_spectral_lines!(spec.ẑ, color=:red, location=:bottom)
end

# ╔═╡ 5d06b901-48ac-4ffe-9435-1d8f777c7409
function evaluate(z, ẑ)
	Dict(
		"rmse" => Evaluation.rmse(z, ẑ),
		"mean Δv" => Evaluation.meanΔv(z, ẑ),
		"std Δv" => std(Evaluation.computeΔv(z, ẑ)),
		"median Δv" => Evaluation.medianΔv(z, ẑ),
		"cfr" => Evaluation.cfr(z, ẑ))
end

# ╔═╡ 42dea5de-1b8f-40a0-981f-0e0abd9cf75c
md"## On-Edge Predictions"

# ╔═╡ 8840bba3-e870-4cd9-a55f-dad3ee5205f2
begin
	idx_edge = zeros(Bool, n)
	for k in 1:n
		set = collect(Set((ẑs[:, k])))
		if length(set) == 2
			idx_edge[k] = abs(set[1] - set[2]) <= 0.015
		end
	end
	sum(idx_edge)
end

# ╔═╡ 427677f2-a2fb-4971-8a26-f6f680249afa
begin
	histogram(df_orig.entropy, label="All", xlabel="Entropy", ylabel="Count")
	histogram!(df_orig.entropy[idx_edge], label="On-Edge")
	histogram!(df_orig.entropy[.~idx_edge], label="Without On-Edge")
end

# ╔═╡ 14f51f9f-5d5e-42a0-b084-7973867dcef4
begin
	df = copy(df_orig)
	df.entropy[idx_edge] .= 0
	histogram(df.entropy)
end

# ╔═╡ de5b8936-7b64-43e0-8ab5-057da3f015fc
md"## $z > 6.445$"

# ╔═╡ 66851320-f094-4b66-a006-c9cfccc1a816
begin
	df_high_z = df[df.z .> 6.445, :]
	n_high_z = size(df_high_z, 1)
	countmap(df_high_z.source_z)
end

# ╔═╡ 82912482-45d7-46b1-84c5-0f0695a94954
first_vi, second_vi = (1:n_high_z)[df_high_z.source_z .== "VI"]

# ╔═╡ 85c11afd-00ed-4bb2-bd7a-5522d2b40132
preview(df_high_z, first_vi)

# ╔═╡ 2de2465a-4470-4afa-94eb-785e8df97752
preview(df_high_z, second_vi)

# ╔═╡ 020a43d8-57e7-4575-a5ce-0189f518a224
preview(df, 1412574)

# ╔═╡ b662295b-9bfd-4765-b20c-bd9185acc7e6
md"## Random Visual Inspection of 10k Spectra"

# ╔═╡ 00803c53-a440-482e-904b-29244ddbc596
df_10k = df[df.idx_10k, :]

# ╔═╡ 0211a7d5-f744-4331-a55c-6860716c2109
evaluate(df_10k.z_10k, df_10k.ẑ)

# ╔═╡ cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
evaluate(df_10k.z_10k, df_10k.z_pipe)

# ╔═╡ dc462c4d-ec37-4459-87e6-87428f2229da
evaluate(df_10k.z_10k, df_10k.z_pca)

# ╔═╡ b69e11ff-1249-43fb-a423-61efa0945030
begin
	idx_10k_qn = df_10k.z_qn .> -1
	evaluate(df_10k.z_10k[idx_10k_qn], df_10k.z_qn[idx_10k_qn])
end

# ╔═╡ c6efefc5-3105-4d3f-aed1-215ac7fb3f3d
begin
	Δv_pred = Evaluation.computeΔv(df_10k.z_10k, df_10k.ẑ)
	Δv_pipe = Evaluation.computeΔv(df_10k.z_10k, df_10k.z_pipe)
	Δv_pca = Evaluation.computeΔv(df_10k.z_10k, df_10k.z_pca)
	Δv_qn = Evaluation.computeΔv(df_10k.z_10k[idx_10k_qn], df_10k.z_qn[idx_10k_qn])
	stop = max(maximum(Δv_pred), maximum(Δv_pipe), maximum(Δv_pca), maximum(Δv_qn))
	bins = range(0, stop=stop, length=128)
	plot(
		Δv_pred, seriestype=:stephist, bins=bins,
		yaxis=:log, label="Bayesian SZNet", xlabel="Δv", ylabel="Count")
	plot!(
		Δv_pipe, seriestype=:stephist, bins=bins,
		yaxis=:log, label="Pipeline")
end

# ╔═╡ 9f7f4a61-16f8-4920-88e6-ac5364c603b1
@df df_10k histogram2d(
	:z_10k, :ẑ, bins=Neural.N_LABELS,
	xlabel="Visual Redshift", ylabel="SZNet Redshift")

# ╔═╡ 0ca11ff7-c0af-475c-be54-35c6b8fa5c84
@df df_10k histogram2d(
	:z_10k, :z_pipe, bins=Neural.N_LABELS,
	xlabel="Visual Redshift", ylabel="Pipeline Redshift")

# ╔═╡ 77542e57-362c-45d0-8612-b71b473f21cd
@df df_10k histogram2d(
	:z_10k, :z_pca, bins=Neural.N_LABELS,
	xlabel="Visual Redshift", ylabel="PCA Redshift")

# ╔═╡ 04895dc3-5bc4-4468-86be-b823f64a6608
@df df_10k histogram2d(
	:z_10k[idx_10k_qn], :z_qn[idx_10k_qn], bins=Neural.N_LABELS,
	xlabel="Visual Redshift", ylabel="QuasarNET Redshift")

# ╔═╡ 573e6e02-a2a9-457f-b2fd-4acdd921c090
md"## Suggestions of Redshifts"

# ╔═╡ a3db9fd5-a9a1-46c3-9097-7573d48bc5df
ẑs

# ╔═╡ c84f8e62-c1e4-4ea8-b69d-8d51175f10e3
begin
	i_sug = 54463
	countmap(ẑs[:, i_sug])
end

# ╔═╡ b8c77084-6ee8-4615-96c7-deb317614e0c
begin
	title = @sprintf(
		"z = %.3f; source = %s ẑ = %.2f; E = %.1f",
		df[i_sug, :z], df[i_sug, :source_z], df[i_sug, :ẑ], df[i_sug, :entropy])
	Utils.plot_spectrum(X[:, i_sug], legend=:none, title=title)
	Utils.plot_spectral_lines!(2.7)
end

# ╔═╡ bc19f08c-839d-484c-98d0-4842a64799ee
md"## Spectra with the Highest Entropy"

# ╔═╡ bb0c2182-7309-4cf2-85f9-7462d41d4b22
i_high_entr = sortperm(df.entropy, rev=true)[1:2]

# ╔═╡ dfe23e93-904b-481f-b247-158637cd361e
preview(df, i_high_entr[1])

# ╔═╡ 75767b4a-cfc6-45e0-a270-5f237f135bed
preview(df, i_high_entr[2])

# ╔═╡ eb11930b-8f4f-4301-a462-a41fa54d980f
md"## Utilisation of Uncertainties"

# ╔═╡ d2c25b73-2a9d-40db-a439-98599e80a33c
begin
	ts = 0.001:0.01:maximum(df.entropy)

	coverages = [sum(df.entropy .< t) / n for t in ts]
	coverages_std = [sum(df.entropy_std .< t) / n for t in ts]

	cfrs = 100 .* [
		Evaluation.cfr(
			df_10k.z_10k[df_10k.entropy .< t],
			df_10k.ẑ[df_10k.entropy .< t])
		for t in ts]
	cfrs_std = 100 .* [
		Evaluation.cfr(
			df_10k.z_10k[df_10k.entropy_std .< t],
			df_10k.ẑ_std[df_10k.entropy_std .< t])
		for t in ts]

	plot(ts, coverages, label="MC Dropout", xlabel="Threshold", ylabel="Coverage")
	plot_coverages = plot!(ts, coverages_std, label="Standard Dropout")

	plot(ts, cfrs, ylabel="Est. Cat. z Ratio", label="MC Dropout")
	plot_cfrs = plot!(ts, cfrs_std, label="Standard Dropout")

	plot(plot_cfrs, plot_coverages, layout=@layout [a; b])
end

# ╔═╡ 01a4659d-4b10-4532-8086-0bf22fbf4825
ceil(Int, n * 0.01), ceil(Int, n * 0.05), ceil(Int, n * 0.1)

# ╔═╡ 406d2e59-6b1e-47ea-bb81-33a264e5134d
begin
	t_99 = 3.9602
	n - sum(df.entropy .< t_99),
	evaluate(df_10k.z_10k[df_10k.entropy .< t_99], df_10k.ẑ[df_10k.entropy .< t_99])
end

# ╔═╡ 85dae4e6-8455-45fa-93c0-9f47a2378386
begin
	t_99_std = 2.058565
	n - sum(df.entropy_std .< t_99_std),
	evaluate(
		df_10k.z_10k[df_10k.entropy_std .< t_99_std],
		df_10k.ẑ_std[df_10k.entropy_std .< t_99_std])
end

# ╔═╡ 8340562d-ae14-49ec-b950-aef0d69228ab
begin
	t_95 = 2.5564000
	n - sum(df.entropy .< t_95),
	evaluate(df_10k.z_10k[df_10k.entropy .< t_95], df_10k.ẑ[df_10k.entropy .< t_95])
end

# ╔═╡ 892dcd13-f929-4e15-aebe-46ed39b2ceb4
begin
	t_95_std = 1.1257656
	n - sum(df.entropy_std .< t_95_std),
	evaluate(
		df_10k.z_10k[df_10k.entropy_std .< t_95_std],
		df_10k.ẑ_std[df_10k.entropy_std .< t_95_std])
end

# ╔═╡ 427dc993-2271-4088-ac16-7b864587e737
begin
	t_90 = 1.72948
	n - sum(df.entropy .< t_90),
	evaluate(df_10k.z_10k[df_10k.entropy .< t_90], df_10k.ẑ[df_10k.entropy .< t_90])
end

# ╔═╡ cbf0aea7-8158-452c-bd43-cb5eb9d90aeb
begin
	t_90_std = 0.866325
	n - sum(df.entropy_std .< t_90_std),
	evaluate(
		df_10k.z_10k[df_10k.entropy_std .< t_90_std],
		df_10k.ẑ_std[df_10k.entropy_std .< t_90_std])
end

# ╔═╡ 0320f4d6-d0cf-440f-b3c1-c405da499edd
md"## Catalogues"

# ╔═╡ 0b9c530b-e7f3-4d07-b059-fe64f1d0cc0b
begin
	catalogue = DataFrame(
		plate=id[1, :],	mjd=id[2, :], fiberid=id[3, :],
		z_pred=df.ẑ,
		z=df.z,
		source_z=df.source_z,
		is_qso_final=df.is_qso_final,
		entropy=df.entropy,
		z_pred_1=ẑs[1, :],
		z_pred_2=ẑs[2, :],
		z_pred_3=ẑs[3, :],
		z_pred_4=ẑs[4, :],
		z_pred_5=ẑs[5, :],
		z_pred_6=ẑs[6, :],
		z_pred_7=ẑs[7, :],
		z_pred_8=ẑs[8, :],
		z_pred_9=ẑs[9, :],
		z_pred_10=ẑs[10, :],
		z_pred_11=ẑs[11, :],
		z_pred_12=ẑs[12, :],
		z_pred_13=ẑs[13, :],
		z_pred_14=ẑs[14, :],
		z_pred_15=ẑs[15, :],
		z_pred_16=ẑs[16, :],
		z_pred_17=ẑs[17, :],
		z_pred_18=ẑs[18, :],
		z_pred_19=ẑs[19, :],
		z_pred_20=ẑs[20, :],
		z_vi=df.z_vi,
		z_pipe=df.z_pipe,
		zwarning=df.zwarning,
		z_dr12q=df.z_dr12q,
		z_dr7q_sch=df.z_dr7q_sch,
		z_dr6q_hw=df.z_dr6q_hw,
		z_10k=df.z_10k,
		z_pca=df.z_pca,
		z_qn=df.z_qn)
end

# ╔═╡ 095af2bb-82aa-4659-b8a5-0bc7b47f174a
CSV.write("data/dr16q_superset_redshift.csv", sort(catalogue, :entropy))

# ╔═╡ 8e460124-460d-47e4-9e26-07e8e138f24f
begin
	Δv = Evaluation.computeΔv(df.z, df.ẑ)
	catalogue_failures = sort(catalogue[Δv .> 3000, :], :entropy)
end

# ╔═╡ 0c6cbade-cd23-4a05-9e99-ccecf8df3696
CSV.write("data/dr16q_superset_redshift_failure.csv", catalogue_failures)

# ╔═╡ be9905a4-abcd-4fbb-9210-e0265c0167a9
md"## Spectra for Appendix"

# ╔═╡ a0733021-a008-4f0c-8432-9643a0edd1e9
md"### Missed QSOs"

# ╔═╡ cf446cac-4894-48b1-ac7d-d1254a439bc4
(1:n)[(Δv .> 3000) .& (df.sn .> 25) .& (df.z .< 0.5) .& (df.entropy .< 1)]

# ╔═╡ 9c06858d-335d-4349-9e0e-b7cbf17e1535
preview(df, 170401)

# ╔═╡ 5397127b-4ca8-46aa-a9a9-b4ea1184dd8b
preview(df, 1327825)

# ╔═╡ 03aa6613-2543-4878-85d8-6e176a817c4e
preview(df, 1031686)

# ╔═╡ cddb53b6-d408-4bc9-b537-4f8f407aa140
preview(df, 114188)

# ╔═╡ 5ef30517-fc7f-40bc-9243-26618985c357
preview(df, 1113376)

# ╔═╡ d128bcf6-fcb4-476f-bda3-4419c466d18d
md"### Incorrect High $z$"

# ╔═╡ bb66f1f9-4458-4463-9ff4-1322a9d38904
(1:n)[(Δv .> 3000) .& (df.sn .> 25) .& (df.z .> 5) .& (df.entropy .< 1)]

# ╔═╡ 73301948-3fa8-4831-b5f0-8b4dcbf6f75f
preview(df, 1433645)

# ╔═╡ fb097f9a-9279-4688-ae6e-4144ab3d0dc5
preview(df, 1436878)

# ╔═╡ eb9573a0-de80-4ce9-a366-32e68aeb1343
preview(df, 160337)

# ╔═╡ 8fb11698-8aa7-4eef-a024-4e1fa4584174
md"### Stars"

# ╔═╡ 4ccf3dfb-5a63-4b66-a4c0-ea5a5aa9d20b
preview(df, 136790)

# ╔═╡ ffc1830c-9aa9-4e76-bf33-2164b0ad784b
df.sn[136790]

# ╔═╡ ec3254d3-4221-4b00-a6e7-88844fe0104b
(1:n)[(df.is_qso_final .== 1) .& (df.sn .> 25) .& (df.ẑ .< 0.005)]

# ╔═╡ 53d39274-90d5-4a33-b019-dfdd125bb670
preview(df, 343869)

# ╔═╡ 4bae3c7c-3959-4499-ade7-57b4b713fbd9
preview(df, 556255)

# ╔═╡ a308b6e5-f822-4cd4-8fb3-a34c6fdf71bc
(1:n)[(Δv .> 3000) .& (df.sn .> 25) .& (df.ẑ .< 0.005) .& (df.entropy .< 1)]

# ╔═╡ f697939c-282b-4771-a973-c0a2d9bb6aef
preview(df, 1433649)

# ╔═╡ e70acd20-97f3-49be-b294-33792389e5bd
preview(df, 202316)

# ╔═╡ 3e400232-36ae-45f7-80e0-5697860f6609
preview(df, 534267)

# ╔═╡ 49a5700d-c61a-4c35-a77a-13653787baea
preview(df, 267587)

# ╔═╡ 51f681c4-e67b-43c8-b656-c94a689a0305
preview(df, 1344651)

# ╔═╡ 16ecba58-89fc-443c-a1bc-05a84199e928
md"### Error with High Entropy"

# ╔═╡ 19b216d2-4e99-46d3-8123-fc462aeaaf94
(1:n)[(Δv .> 3000) .& (df.sn .> 25)]

# ╔═╡ 932bc26e-7d05-4866-929f-869ab7e4e6a6
preview(df, 66523)

# ╔═╡ ba4ea09a-5e02-41a0-803a-c5d31848b61e
preview(df, 1359912)

# ╔═╡ 7aa58872-eec9-4283-b357-18b6f502dfa8
preview(df, 44949)

# ╔═╡ Cell order:
# ╟─fce2913d-6c91-492b-9b98-81f5c886c467
# ╠═bdb521fa-acd6-11eb-0d41-2d68a7abecb2
# ╠═a13527ac-4670-4a2f-a390-17700d707705
# ╟─649c5981-e53b-4258-80cf-946c9b7b7a28
# ╟─5d06b901-48ac-4ffe-9435-1d8f777c7409
# ╟─42dea5de-1b8f-40a0-981f-0e0abd9cf75c
# ╠═427677f2-a2fb-4971-8a26-f6f680249afa
# ╠═8840bba3-e870-4cd9-a55f-dad3ee5205f2
# ╠═14f51f9f-5d5e-42a0-b084-7973867dcef4
# ╟─de5b8936-7b64-43e0-8ab5-057da3f015fc
# ╠═66851320-f094-4b66-a006-c9cfccc1a816
# ╠═82912482-45d7-46b1-84c5-0f0695a94954
# ╠═85c11afd-00ed-4bb2-bd7a-5522d2b40132
# ╠═2de2465a-4470-4afa-94eb-785e8df97752
# ╠═020a43d8-57e7-4575-a5ce-0189f518a224
# ╟─b662295b-9bfd-4765-b20c-bd9185acc7e6
# ╠═00803c53-a440-482e-904b-29244ddbc596
# ╠═0211a7d5-f744-4331-a55c-6860716c2109
# ╠═cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
# ╠═dc462c4d-ec37-4459-87e6-87428f2229da
# ╠═b69e11ff-1249-43fb-a423-61efa0945030
# ╠═c6efefc5-3105-4d3f-aed1-215ac7fb3f3d
# ╠═9f7f4a61-16f8-4920-88e6-ac5364c603b1
# ╠═0ca11ff7-c0af-475c-be54-35c6b8fa5c84
# ╠═77542e57-362c-45d0-8612-b71b473f21cd
# ╠═04895dc3-5bc4-4468-86be-b823f64a6608
# ╟─573e6e02-a2a9-457f-b2fd-4acdd921c090
# ╠═a3db9fd5-a9a1-46c3-9097-7573d48bc5df
# ╠═c84f8e62-c1e4-4ea8-b69d-8d51175f10e3
# ╠═b8c77084-6ee8-4615-96c7-deb317614e0c
# ╟─bc19f08c-839d-484c-98d0-4842a64799ee
# ╠═bb0c2182-7309-4cf2-85f9-7462d41d4b22
# ╠═dfe23e93-904b-481f-b247-158637cd361e
# ╠═75767b4a-cfc6-45e0-a270-5f237f135bed
# ╟─eb11930b-8f4f-4301-a462-a41fa54d980f
# ╠═d2c25b73-2a9d-40db-a439-98599e80a33c
# ╠═01a4659d-4b10-4532-8086-0bf22fbf4825
# ╠═406d2e59-6b1e-47ea-bb81-33a264e5134d
# ╠═85dae4e6-8455-45fa-93c0-9f47a2378386
# ╠═8340562d-ae14-49ec-b950-aef0d69228ab
# ╠═892dcd13-f929-4e15-aebe-46ed39b2ceb4
# ╠═427dc993-2271-4088-ac16-7b864587e737
# ╠═cbf0aea7-8158-452c-bd43-cb5eb9d90aeb
# ╟─0320f4d6-d0cf-440f-b3c1-c405da499edd
# ╠═0b9c530b-e7f3-4d07-b059-fe64f1d0cc0b
# ╠═095af2bb-82aa-4659-b8a5-0bc7b47f174a
# ╠═8e460124-460d-47e4-9e26-07e8e138f24f
# ╠═0c6cbade-cd23-4a05-9e99-ccecf8df3696
# ╟─be9905a4-abcd-4fbb-9210-e0265c0167a9
# ╟─a0733021-a008-4f0c-8432-9643a0edd1e9
# ╠═cf446cac-4894-48b1-ac7d-d1254a439bc4
# ╠═9c06858d-335d-4349-9e0e-b7cbf17e1535
# ╠═5397127b-4ca8-46aa-a9a9-b4ea1184dd8b
# ╠═03aa6613-2543-4878-85d8-6e176a817c4e
# ╠═cddb53b6-d408-4bc9-b537-4f8f407aa140
# ╠═5ef30517-fc7f-40bc-9243-26618985c357
# ╟─d128bcf6-fcb4-476f-bda3-4419c466d18d
# ╠═bb66f1f9-4458-4463-9ff4-1322a9d38904
# ╠═73301948-3fa8-4831-b5f0-8b4dcbf6f75f
# ╠═fb097f9a-9279-4688-ae6e-4144ab3d0dc5
# ╠═eb9573a0-de80-4ce9-a366-32e68aeb1343
# ╟─8fb11698-8aa7-4eef-a024-4e1fa4584174
# ╠═4ccf3dfb-5a63-4b66-a4c0-ea5a5aa9d20b
# ╠═ffc1830c-9aa9-4e76-bf33-2164b0ad784b
# ╠═ec3254d3-4221-4b00-a6e7-88844fe0104b
# ╠═53d39274-90d5-4a33-b019-dfdd125bb670
# ╠═4bae3c7c-3959-4499-ade7-57b4b713fbd9
# ╠═a308b6e5-f822-4cd4-8fb3-a34c6fdf71bc
# ╠═f697939c-282b-4771-a973-c0a2d9bb6aef
# ╠═e70acd20-97f3-49be-b294-33792389e5bd
# ╠═3e400232-36ae-45f7-80e0-5697860f6609
# ╠═49a5700d-c61a-4c35-a77a-13653787baea
# ╠═51f681c4-e67b-43c8-b656-c94a689a0305
# ╟─16ecba58-89fc-443c-a1bc-05a84199e928
# ╠═19b216d2-4e99-46d3-8123-fc462aeaaf94
# ╠═932bc26e-7d05-4866-929f-869ab7e4e6a6
# ╠═ba4ea09a-5e02-41a0-803a-c5d31848b61e
# ╠═7aa58872-eec9-4283-b357-18b6f502dfa8
