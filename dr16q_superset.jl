### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 3621f840-5fcd-11eb-3be8-15ac4ab1b566
using DataFrames, FITSIO, StatsPlots

# ╔═╡ 05685b0c-5fcf-11eb-1f7f-c77372b9790e
md"# DR16Q Superset Catalogue

> The superset contains 1,440,615 observations of quasars, stars, and galaxies that were all targeted as quasars (or appeared in previous quasar catalogs)."

# ╔═╡ 5e927e1c-5fcd-11eb-39cd-d19c696d924e
begin
	superset_fits = FITS("data/DR16Q_Superset_v3.fits")
	superset_cols = [
		:PLATE, :MJD, :FIBERID, :IS_QSO_QN, :Z_QN,
		:RANDOM_SELECT, :Z_10K, :Z_CONF_10K, :PIPE_CORR_10K, :IS_QSO_10K,
        :Z_VI, :Z_CONF, :CLASS_PERSON, :IS_QSO_FINAL, :Z, :SOURCE_Z, :Z_PIPE,
		:ZWARNING, :Z_PCA, :ZWARN_PCA]
	superset = DataFrame(superset_fits[2])[:, superset_cols]
end

# ╔═╡ 9e7bd468-5fcf-11eb-24d2-a349a471fc8e
md"> For objects that have a redshift in the columns Z\_VI or Z\_10K and a confidence (Z\_CONF or Z\_CONF\_10K) of ≥ 2, Z records the corresponding redshift and SOURCE\_Z is set to VI.
> Otherwise, if an object has a redshift in the columns Z\_DR6Q\_HW or Z\_DR7Q\_SCH these values are used (with Z\_DR6Q\_HW overriding Z\_DR7Q\_SCH) and SOURCE\_Z is set to DR6Q\_HW or DR7QV\_SCH. As the ZDR7Q\_HW redshifts did not formally appear in the Shen et al. (2011) paper, these values are not used to populate the Z column.
> If no other visual inspection redshift is populated then Z\_DR12Q is used (and SOURCE\_Z is set to DR12QV).
> For objects with DR12Q redshifts, only the visual inspection redshifts are recorded; DR12Q pipeline redshifts  are not included.
> In the absence of any of these visual inspection redshifts, Z is populated with the automated pipeline redshift (and SOURCE\_Z is set to PIPE)."

# ╔═╡ c82279c0-5fcf-11eb-17b9-1136e2a2792d
begin
	source_idx = superset["SOURCE_Z"] .!= "PIPE"
	gt_zero_idx = superset["Z"] .> 0
	# TODO should I include epsilon?
	eq_zero_idx = superset["Z"] .== 0
	sum(source_idx), sum(gt_zero_idx), sum(eq_zero_idx)
end

# ╔═╡ 50475460-5fd0-11eb-0ec7-7301438e4cd2
source_subset = superset[source_idx .& (gt_zero_idx .| eq_zero_idx), :]

# ╔═╡ 282e7e62-5fd1-11eb-27ba-7d4d50fc1374
@df source_subset density(:Z, xlabel="z", ylabel="Density", legend=:none)

# ╔═╡ 868625f0-5fd1-11eb-00dd-1f68c388ceb6
md"## Wavelength Range"

# ╔═╡ 920b99f0-5fd1-11eb-2142-ab569810bf77
begin
	specobj_fits = FITS("data/specObj-dr16.fits")
	specobj_cols = [
		:PLATE, :MJD, :FIBERID, :WAVEMIN, :WAVEMAX, :WCOVERAGE, :RUN2D]
	specobj = DataFrame(specobj_fits[2])[:, specobj_cols]
end

# ╔═╡ 6cb7f21c-5fd2-11eb-324c-6755e1ad21db
wave_subset = leftjoin(source_subset, specobj, on=[:PLATE, :MJD, :FIBERID])

# ╔═╡ Cell order:
# ╟─05685b0c-5fcf-11eb-1f7f-c77372b9790e
# ╠═3621f840-5fcd-11eb-3be8-15ac4ab1b566
# ╠═5e927e1c-5fcd-11eb-39cd-d19c696d924e
# ╟─9e7bd468-5fcf-11eb-24d2-a349a471fc8e
# ╠═c82279c0-5fcf-11eb-17b9-1136e2a2792d
# ╠═50475460-5fd0-11eb-0ec7-7301438e4cd2
# ╠═282e7e62-5fd1-11eb-27ba-7d4d50fc1374
# ╟─868625f0-5fd1-11eb-00dd-1f68c388ceb6
# ╠═920b99f0-5fd1-11eb-2142-ab569810bf77
# ╠═6cb7f21c-5fd2-11eb-324c-6755e1ad21db
