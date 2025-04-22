import armament as ars
import utils
import argparse
from ovito import scene
from ovito.io import import_file
from ovito.vis import Viewport
from ovito.modifiers import ComputePropertyModifier
from ovito.modifiers import ColorCodingModifier
from ovito.modifiers import DeleteSelectedModifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pickle
import sys
import os

## Description
# this script reads a bath of DNAfold trajectories and calculates the correlation
  # of the staple hybridization times with some measure of the final product
  # quality (for now, either crystallinity or final number of bound staples).
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).
# the hybridization / product quality correlation (hybCorr) is a rather simple
  # concept when applied to individual beads; however, when extracting a
  # single correlation value for each strand, there are several options.
  # First, there are the options where some "composite" hyb time is calculed
  # for each strand, then this value is correlated with the product quality.
  # This composite value can be either the average, earliest, or latest of
  # the hyb times of the component beads. Second, there are the options where
  # the correlation is calulated for each component bead, then some
  # "composite" correlation is calculated. This composite value can be either 
  # the average or the max (absolute value) correlation.


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--cadFile',		type=str,	required=True,		help='name of caDNAno file')
	parser.add_argument('--copiesFile',		type=str,	required=True,		help='name of copies file, which contains a list of simulation folders')	
	parser.add_argument('--topFile',		type=str, 	default=None,		help='if using oxdna positions, name of topology file')
	parser.add_argument('--confFile',		type=str, 	default=None,		help='if using oxdna positions, name of conformation file')
	parser.add_argument('--corr_var',		type=str,	default="final_S",	help='hat to correlate with the strands (values: final_S, final_n_hyb)')
	parser.add_argument('--corr_type',		type=str,	default="hyb_last",	help='how to average correlation across strand (values: hyb_avg, hyb_first, hyb_last, corr_avg, corr_max')
	parser.add_argument('--values_report',	type=str,	default="none",		help='what staple results to report (values: none, staple, corr, all')
	parser.add_argument('--loadResults',	type=bool,	default=False,		help='whether to load the results from a pickle file')
	parser.add_argument('--nstep_skip',		type=float,	default=0,			help='if not loading results, number of recorded initial steps to skip')
	parser.add_argument('--nstep_max',		type=float,	default=0,			help='max number of recorded steps to use (0 for all)')
	parser.add_argument('--coarse_time',	type=int,	default=1,			help='if not loading results, coarse factor for time steps')
	parser.add_argument('--mov_avg_stride',	type=int,	default=1,			help='if not loading results, stride length for moving average')

	### set arguments
	args = parser.parse_args()
	cadFile = args.cadFile
	copiesFile = args.copiesFile
	topFile = args.topFile
	confFile = args.confFile
	corr_var = args.corr_var
	corr_type = args.corr_type
	values_report = args.values_report
	loadResults = args.loadResults
	nstep_skip = int(args.nstep_skip)
	nstep_max = int(args.nstep_max)
	coarse_time = args.coarse_time
	mov_avg_stride = args.mov_avg_stride

	### check input
	if topFile is not None and confFile is not None:
		position_src = "oxdna"
	else:
		position_src = "cadnano"
		if topFile is not None:
			print("Flag: oxDNA topology file provided without configuration file, using caDNAno positions.")
		if confFile is not None:
			print("Flag: oxDNA configuration file provided without topology file, using caDNAno positions.")


################################################################################
### Heart

	### read data and calculate results
	if not loadResults:

		### get simulation folders
		simFolds, nsim = utils.getSimFolds(copiesFile)

		### get pickled data
		connFile = "analysis/connectivity_vars.pkl"
		ars.testFileExist(connFile,"connectivity")
		with open(connFile,"rb") as f:
			[strands, bonds_backbone, complements, n_scaf, nbead] = pickle.load(f)[:5]

		### get minimum number of steps
		nstep_allSim = np.zeros(nsim,dtype=int)
		for i in range(nsim):
			datFile = simFolds[i] + "analysis/trajectory_centered.dat"
			nstep_allSim[i] = ars.getNstep(datFile, nstep_skip, coarse_time)
		nstep_min = int(min(nstep_allSim))
		nstep_use = nstep_min if nstep_max == 0 else min([nstep_min,nstep_max])

		### loop through simulations
		first_hyb_times_scaled_allSim = np.zeros((nsim,n_scaf))
		n_hyb_frac_allSim = np.zeros((nsim,nstep_use))
		S_allSim = np.zeros((nsim,nstep_use))
		for i in range(nsim):

			### calculate hyb times
			hybFile = simFolds[i] + "analysis/hyb_status.dat"
			hyb_status = utils.readHybStatus(hybFile, nstep_skip, coarse_time, nstep_use)
			dump_every = utils.getDumpEveryHyb(hybFile)*coarse_time
			first_hyb_times_scaled_allSim[i] = utils.calcFirstHybTimes(hyb_status, complements, n_scaf, dump_every)[1]
			n_hyb_frac_allSim[i] = np.sum(hyb_status==1,axis=1)/nbead

			### calculate crystallinity
			datFile = simFolds[i] + "analysis/trajectory_centered.dat"
			bdis_scaf = list(range(1,n_scaf+1))
			points, _, dbox = ars.readAtomDump(datFile, nstep_skip, coarse_time, bdis=bdis_scaf, nstep_max=nstep_use); print()
			S = utils.calcCrystallinity(points, dbox)
			S_allSim[i] = ars.movingAvg(S, mov_avg_stride)

		### store results
		resultsFile = "analysis/hyb_correlation_results.pkl"
		with open(resultsFile, 'wb') as f:
			pickle.dump([first_hyb_times_scaled_allSim, n_hyb_frac_allSim, S_allSim, strands, bonds_backbone, complements, n_scaf], f)

	### load results
	else:
		resultsFile = "analysis/hyb_correlation_results.pkl"
		ars.testFileExist(resultsFile,"results")
		with open(resultsFile, 'rb') as f:
			[first_hyb_times_scaled_allSim, n_hyb_frac_allSim, S_allSim, strands, bonds_backbone, complements, n_scaf] = pickle.load(f)


################################################################################
### Results

	### select product quality variable for hybridization correlation
	if corr_var == "final_n_hyb":
		product_quality = n_hyb_frac_allSim[:,-1]
	elif corr_var == "final_S":
		product_quality =  S_allSim[:,-1]
	else:
		print("Error: Unknown correlation variable.")
		sys.exit()

	### calculate and visualize correlation
	hybCorr, hybCorr_strand = calcHybCorr(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type)

	### prepare position data
	if position_src == "cadnano":
		r = utils.initPositionsCaDNAno(cadFile)[0]
	if position_src == "oxdna":
		r = utils.initPositionsOxDNA(cadFile, topFile, confFile)[0]
	r, charges, dbox3 = prepGeoData(r, strands, complements, hybCorr, hybCorr_strand)

	### write geometry and ovito files
	outGeoFile = "analysis/geometry_hybCorr.in"
	ars.writeGeo(outGeoFile, dbox3, r, types=strands, charges=charges, bonds=bonds_backbone)
	ovitoFile = "analysis/vis_hybCorr.ovito"
	writeOvito(ovitoFile, outGeoFile)

	### report best staples
	staple_labels = np.arange(2,max(strands)+1)
	sorted_results = sorted(zip(hybCorr_strand[1:],staple_labels), reverse=True)
	for corr, s in sorted_results:
		if values_report == "staple":
			print(f"{s:2.0f}")
		if values_report == "corr":
			print(f"{corr:0.4f}")
		if values_report == "all":
			print(f"Strand {s:2.0f}: {corr:0.4f}")

	### correlation statistics
	plotCorrHist(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type)
	plt.show()


################################################################################
### Plotters

### plot histogram of hybridization correlations
def plotCorrHist(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type):

	### actual correlation
	hybCorr_strand = calcHybCorr(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type)[1]

	### random correlation
	nrand = 100
	nstrand = max(strands)
	hybCorr_strand_rand = np.zeros((nrand,nstrand))
	for i in range(nrand):
		rng = np.random.default_rng()
		rng.shuffle(product_quality)
		hybCorr_strand_rand[i] = calcHybCorr(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type)[1]
	hybCorr_strand_rand = hybCorr_strand_rand.reshape(nrand*nstrand)

	### plot
	nbin = 10
	Alim = [-1,1]
	ars.magicPlot()
	ars.plotHist(hybCorr_strand_rand,nbin=nbin*2,Alim_bin=Alim,plotAsLine=True)
	ars.plotHist(hybCorr_strand,nbin=nbin,Alim_bin=Alim)
	plt.legend(["Randomized","Original Data"])


### write session state vito file that visualizes the geometry
def writeOvito(ovitoFile, outGeoFile):

	### initialize pipeline
	pipeline = import_file(outGeoFile, atom_style="full")
	pipeline.add_to_scene()

	### prepare basic DNAfold scene
	pipeline = utils.setOvitoBasics(pipeline)

	### set scaffold and staple particle radii and bond widths
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Radius',expressions=['(ParticleType==1)?0.6:1']))
	pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds',output_property='Width',expressions=['(@1.ParticleType==1)?1.2:2']))

	### set color coding
	pipeline.modifiers.append(ColorCodingModifier(property='Charge',start_value=-1,end_value=1,gradient=ColorCodingModifier.BlueWhiteRed()))
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Color',expressions=['(ParticleType==1)?1:Color.R','(ParticleType==1)?1:Color.G','(ParticleType==1)?1:Color.B']))

	### add option to delete staples
	pipeline.modifiers.append(ComputePropertyModifier(enabled=False,output_property='Selection',expressions=['ParticleType!=1']))
	pipeline.modifiers.append(DeleteSelectedModifier())

	### write ovito file
	scene.save(ovitoFile)
	pipeline.remove_from_scene()


################################################################################
### Calculation Managers

### calculate correlation between hybridization times (h) and product quality (q)
def calcHybCorr(h, q, strands, complements, corr_type):
	nsim = h.shape[0]
	n_scaf = h.shape[1]
	nstrand = max(strands)

	### correlation by bead
	corr = np.zeros(n_scaf)
	for bi in range(n_scaf):
		if len(set(h[:,bi])) != 1:
			corr[bi] = np.corrcoef(h[:,bi],q)[0,1]

	### group correlation by strands, pre-correlation methods
	if corr_type == "hyb_avg" or corr_type == "hyb_first" or corr_type == "hyb_last":
		h_grouped = [ np.zeros((nsim,0)) for i in range(nstrand) ]
		h_grouped[0] = np.zeros((nsim,1))
		for bi in range(n_scaf):
			if len(complements[bi]) > 0:
				strand = strands[complements[bi][0]-1]
				A = h_grouped[strand-1]
				B = h[:,bi].reshape(nsim,1)
				h_grouped[strand-1] = np.concatenate((A,B),axis=1)
		corr_strand = np.zeros(nstrand)
		for si in range(1,nstrand):
			if corr_type == "hyb_avg":
				h_singleStrand = np.mean(h_grouped[si],axis=1)
			elif corr_type == "hyb_first":
				h_singleStrand = np.min(h_grouped[si],axis=1)
			elif corr_type == "hyb_last":
				h_singleStrand = np.max(h_grouped[si],axis=1)
			if len(set(h_singleStrand)) != 1:
				corr_strand[si] = np.corrcoef(h_singleStrand,q)[0,1]

	### group correlation by strands, post-correlation methods
	elif corr_type == "corr_avg" or corr_type == "corr_max":
		corr_grouped = [ [] for si in range(nstrand) ]
		corr_grouped[0] = [0]
		for bi in range(n_scaf):
			if len(complements[bi]) > 0:
				strand = strands[complements[bi][0]-1]
				corr_grouped[strand-1].append(corr[bi])
		if corr_type == "corr_avg":
			corr_strand = np.array([ np.mean(corr_grouped[si]) for si in range(nstrand) ])
		elif corr_type == "corr_max":
			corr_strand = np.zeros(nstrand)
			for si in range(nstrand):
				if all(c > 0 for c in corr_grouped[si]):
					corr_strand[si] = max(corr_grouped[si])
				elif all(c < 0 for c in corr_grouped[si]):
					corr_strand[si] = min(corr_grouped[si])
				else:
					corr_strand[si] = 0

	### error message
	else:
		print("Error: Unknown correlation type.")
		sys.exit()

	### results
	return corr, corr_strand


################################################################################
### Utility Functions

### get geometry data ready for visualization
def prepGeoData(r, strands, complements, hybCorr, hybCorr_strand):
	n_ori = len(strands)
	n_scaf = np.sum(strands==1)

	### box diameter
	dbox3 = [ max(abs(r[:,0]))+2.72, max(abs(r[:,1]))+2.4, max(abs(r[:,2]))+2.4 ]
	dbox3 = [ 2*i for i in dbox3 ]

	### assign charge
	charges = np.zeros(n_ori)
	charges[:n_scaf] = hybCorr
	for bi in range(n_scaf,n_ori):
		charges[bi] = hybCorr_strand[strands[bi]-1]

	### return results
	return r, charges, dbox3


### run the script
if __name__ == "__main__":
	main()
	print()

