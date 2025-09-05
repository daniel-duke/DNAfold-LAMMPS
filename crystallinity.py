import arsenal as ars
import utils
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pickle
import sys
import os

## Description
# this script reads a DNAfold trajectory (or a batch of trajectories) and
  # calculates the crystallinity of the scaffold throughout the simulation.
# other product quality measurements (RMSD and ACN) can also be calculated.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).

## To Do
# write a separate script that analyzes a single "started from caDNAno" simulation, 
  # gets the mean structure, then saves the positions to a pickle file, then add
  # the option in this script to read that file and use it as the RMSD reference.


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',		type=str,	default=None,	help='name of copies file (first column - simulation folder names)')
	parser.add_argument('--simFold',		type=str,	default=None,	help='name of simulation folder, used if no copies file, defaults to current folder')
	parser.add_argument('--clusterFile',	type=str,	default=None,	help='name of cluster file, only used for crystallinity')	
	parser.add_argument('--nstep_skip',		type=float,	default=0,		help='if not loading results, number of recorded initial steps to skip')
	parser.add_argument('--coarse_time',	type=int,	default=1,		help='if not loading results, coarse factor for time steps')
	parser.add_argument('--doRMSD',			type=int,	default=False,	help='whether to calculate RMSD')		
	parser.add_argument('--cadFile',		type=str,	default=None,	help='if calculating RMSD, name of caDNAno file')	
	parser.add_argument('--topFile',		type=str, 	default=None,	help='if calculating RMSD and using oxdna positions, name of topology file')
	parser.add_argument('--confFile',		type=str, 	default=None,	help='if calculating RMSD and using oxdna positions, name of conformation file')
	parser.add_argument('--doACN',			type=int,	default=False,	help='whether to calculate ACN')		
	parser.add_argument('--loadResults',	type=int,	default=False,	help='whether to load the results from a pickle file')
	parser.add_argument('--nstep_max',		type=float,	default=0,		help='max number of extracted steps to use, excluding initial frame (0 for all)')
	parser.add_argument('--mov_avg_stride',	type=int,	default=1,		help='stride length for moving average')
	parser.add_argument('--saveFig',		type=int,	default=False,	help='whether to save pdf of figures')

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	simFold = args.simFold
	clusterFile = args.clusterFile
	nstep_skip = int(args.nstep_skip)
	coarse_time = args.coarse_time
	doRMSD = args.doRMSD
	cadFile = args.cadFile
	topFile = args.topFile
	confFile = args.confFile
	doACN = args.doACN
	loadResults = args.loadResults
	nstep_max = int(args.nstep_max)
	mov_avg_stride = args.mov_avg_stride
	saveFig = args.saveFig

	### check arguments
	if copiesFile is not None and clusterFile is not None:
		clusterFile = None
		print("Flag: Both copies and cluster file provided, so ignoring cluster file.")
	if doRMSD and clusterFile is not None:
		print("Flag: skipping RMSD calculation, clusters not supported yet.")
		doRMSD = False
	if doACN and clusterFile is not None:
		print("Flag: skipping ACN calculation, clusters not supported yet.")
		doACN = False
	if mov_avg_stride == parser.get_default('mov_avg_stride'):
		print("Flag: Using the default moving average stide (1), which may not be ideal.")

	### check for files required for RMSD
	if doRMSD:
		if cadFile is None:
			print("Error: caDNAno file required for RMSD calculation.\n")
			sys.exit()
		if topFile is not None and confFile is not None:
			position_src = 'oxdna'
		else:
			position_src = 'cadnano'
			if topFile is not None:
				print("Flag: oxDNA topology file provided without configuration file, using caDNAno positions.")
			if confFile is not None:
				print("Flag: oxDNA configuration file provided without topology file, using caDNAno positions.")


################################################################################
### Heart

	### read data and calculate results
	if not loadResults:

		### get simulation folders
		simFolds, nsim = utils.getSimFolds(copiesFile, simFold)

		### get pickled data
		connFile = "analysis/connectivity_vars.pkl"
		n_scaf, scaf_shift = readConn(connFile)

		### get ideal positions
		if doRMSD: r_ideal = utils.initPositionsCaDNAno(cadFile, scaf_shift)[0][:n_scaf]

		### get minimum number of steps
		nstep_allSim = np.zeros(nsim,dtype=int)
		for i in range(nsim):
			datFile = simFolds[i] + "analysis/trajectory_centered.dat"
			nstep_allSim[i] = ars.getNstep(datFile, nstep_skip, coarse_time)
		nstep_min = int(min(nstep_allSim))
		nstep_use = nstep_min if nstep_max == 0 else min([nstep_min,nstep_max+1])

		### whole scaffold analysis
		if clusterFile is None:

			### loop through simulations
			S_allSim = np.zeros((nsim,nstep_use))
			RMSD_allSim = np.zeros((nsim,nstep_use))
			ACN_allSim = np.zeros(nsim)
			for i in range(nsim):

				### calculations
				datFile = simFolds[i] + "analysis/trajectory_centered.dat"; print()
				points, _, dbox, used_every = ars.readAtomDump(datFile, nstep_skip, coarse_time, bdis=-n_scaf, nstep_max=nstep_use, getUsedEvery=True)
				S_allSim[i] = utils.calcCrystallinity(points, dbox)
				if doRMSD: RMSD_allSim[i] = utils.calcRMSD(points, r_ideal)
				if doACN: ACN_allSim[i] = calcACN(points[-1])

		### clustered scaffold analysis
		else:

			### read clusters
			bdis = ars.readCluster(clusterFile)
			ncluster = len(bdis)

			### read trajectory
			datFile = simFolds[0] + "analysis/trajectory_centered.dat"; print()
			points, _, dbox, molecules, used_every = ars.readAtomDump(datFile, nstep_skip, coarse_time, bdis=bdis, nstep_max=nstep_use, getUsedEvery=True)
			points = ars.sortPointsByMolecule(points, molecules)

			### loop through clusters
			S_allSim = np.zeros((ncluster,nstep_use))
			RMSD_allSim = np.zeros((ncluster,nstep_use))
			ACN_allSim = np.zeros(ncluster)
			for i in range(ncluster):

				### calculate crystallinity
				S_allSim[i] = utils.calcCrystallinity(points[i], dbox)

		### store results
		resultsFile = "analysis/crystallinity_results.pkl"
		with open(resultsFile, 'wb') as f:
			pickle.dump([S_allSim, RMSD_allSim, ACN_allSim, used_every], f)

	### load results
	else:
		resultsFile = "analysis/crystallinity_results.pkl"
		cucumber = ars.unpickle(resultsFile, [2,2,1,0])
		[S_allSim, RMSD_allSim, ACN_allSim, used_every] = cucumber

		### trim steps
		nstep_pkl = S_allSim.shape[1]
		nstep_use = nstep_pkl if nstep_max == 0 else min([nstep_pkl,nstep_max+1])
		S_allSim = S_allSim[:,:nstep_use]
		RMSD_allSim = RMSD_allSim[:,:nstep_use]


################################################################################
### Results

	### apply moving average
	for i in range(len(S_allSim)):
		S_allSim[i] = ars.movingAvg(S_allSim[i], mov_avg_stride)
		RMSD_allSim[i] = ars.movingAvg(RMSD_allSim[i], mov_avg_stride)

	### report best simulations
	if copiesFile is not None:
		copyNames = utils.readCopies(copiesFile)[0]
		sorted_results = sorted(zip(S_allSim[:,-1], RMSD_allSim[:,-1], ACN_allSim, copyNames), reverse=True)
		print("\nProduct qualities:")
		for S_final, RMSD, ACN, copyName in sorted_results:
			if doRMSD and doACN:
				print(f"{copyName}: {S_final:0.4f}, {RMSD:0.4f}, {ACN:0.4f}")
			elif doRMSD:
				print(f"{copyName}: {S_final:0.4f}, {RMSD:0.4f}")
			elif doACN:
				print(f"{copyName}: {S_final:0.4f}, {ACN:0.4f}")
			else:
				print(f"{copyName}: {S_final:0.4f}")

	### create figures folder
	if saveFig: ars.createSafeFold("analysis/figures")

	### RMSD
	if doRMSD:
		plotRMSD(RMSD_allSim, used_every)
		if saveFig: plt.savefig("analysis/figures/RMSD.pdf")
		if nsim > 1:
			plotRMSDvS(RMSD_allSim, S_allSim)
			if saveFig: plt.savefig("analysis/figures/RMSDvS.pdf")

	### ACN
	if doACN and nsim > 1:
		plotACNvS(ACN_allSim, S_allSim)
		if saveFig: plt.savefig("analysis/figures/ACNvS.pdf")

	### crystallinity
	plotCrystallinity(S_allSim, used_every)
	if saveFig: plt.savefig("analysis/figures/crystallinity.pdf")

	### display
	if not saveFig: plt.show()


################################################################################
### Plotters

### plot average Landau-De Gennes crystallinity parameter
def plotCrystallinity(S_allSim, used_every):
	nsim = S_allSim.shape[0]
	nstep = S_allSim.shape[1]
	plotMedian = False
	plotSEM = False

	### calculations
	S_avg = np.median(S_allSim, axis=0)
	S_sem = np.zeros(nstep)
	for i in range(nstep):
		S_sem[i] = ars.calcSEM(S_allSim[:,i])
	time = utils.getTime(nstep, used_every)

	### plot prep
	cmap = cm.viridis
	norm = mcolors.Normalize(vmin=1, vmax=nsim)
	ranks = [ sorted(S_allSim[:,-1]).index(x) + 1 for x in S_allSim[:,-1] ]

	### configure plot
	ars.magicPlot()
	plt.figure("Crystallinity")
	plt.ylim(0,0.8)
	plt.xlabel("Time [$s$]")
	plt.ylabel("Crystallinity")

	### plot the data
	if nsim == 1:
		plt.plot(time, S_avg, color='k')
	else:
		for i in range(nsim):
			color = cmap(norm(ranks[i]))
			plt.plot(time, S_allSim[i], color=color, linewidth=2, alpha=0.6)
		if plotMedian:
			plt.plot(time, S_avg, color='k', linewidth=4)
		for i in range(nsim):
			color = cmap(norm(ranks[i]))
			plt.plot(time[-1], S_allSim[i,-1], 'o', color=color)
		if plotMedian:
			plt.plot(time[-1], S_avg[-1], 'o', color='k')
		if plotSEM:
			plt.fill_between(time, S_avg-S_sem, S_avg+S_sem, color='k', alpha=0.3, edgecolor='none')


### plot simulation and average crystallinity
def plotRMSD(RMSD_allSim, used_every):
	nsim = RMSD_allSim.shape[0]
	nstep = RMSD_allSim.shape[1]
	plotMedian = False
	plotSEM = False

	### calculations
	RMSD_avg = np.mean(RMSD_allSim, axis=0)
	RMSD_sem = np.zeros(nstep)
	for i in range(nstep):
		RMSD_sem[i] = ars.calcSEM(RMSD_allSim[:,i])
	time = utils.getTime(nstep, used_every)

	### plot prep
	cmap = cm.viridis
	norm = mcolors.Normalize(vmin=1, vmax=nsim)
	ranks = [ sorted(RMSD_allSim[:,-1]).index(x) + 1 for x in RMSD_allSim[:,-1] ]

	### configure plot
	ars.magicPlot()
	plt.figure("RMSD")
	plt.xlabel("Time [$s$]")
	plt.ylabel("$RMSD$")

	### plot the data
	if nsim == 1:
		plt.plot(time, RMSD_avg, color='k')
	else:
		for i in range(nsim):
			color = cmap(norm(ranks[i]))
			plt.plot(time,RMSD_allSim[i], color=color, linewidth=2, alpha=0.6)
		if plotMedian:
			plt.plot(time, RMSD_avg, color='k', linewidth=4)
		for i in range(nsim):
			color = cmap(norm(ranks[i]))
			plt.plot(time[-1], RMSD_allSim[i,-1], 'o', color=color)
		if plotMedian:
			plt.plot(time[-1],RMSD_avg[-1], 'o', color='k')
		if plotSEM:
			plt.fill_between(time, RMSD_avg-RMSD_sem, RMSD_avg+RMSD_sem, color='k', alpha=0.3, edgecolor='none')


### plot average crossing number against crystallinity
def plotACNvS(ACN_allSim, S_allSim):
	ars.magicPlot()
	plt.figure("ACN vs S")
	plt.scatter(S_allSim[:,-1], ACN_allSim, color='k')
	plt.xlabel("Final Crystallinity")
	plt.ylabel("Average Crossing Number")


### plot RMSD against crystallinity
def plotRMSDvS(RMSD_allSim, S_allSim):
	ars.magicPlot()
	plt.figure("RMSD vs S")
	plt.scatter(S_allSim[:,-1], RMSD_allSim[:,-1], color='k')
	plt.xlabel("Final Crystallinity")
	plt.ylabel("RMSD")


################################################################################
### File Handlers

### get connectivity variables
def readConn(connFile):
	ars.testFileExist(connFile, "connectivity")
	with open(connFile, 'rb') as f:
		params = pickle.load(f)
	n_scaf = params['n_scaf']
	circularScaf = params['circularScaf']
	scaf_shift = 0 if circularScaf else params['scaf_shift']
	return n_scaf, scaf_shift


################################################################################
### Average Crossing Number

### used to calculate ACN
def generateRandomProjection():
	vec = np.random.randn(3)
	vec /= np.linalg.norm(vec)
	if np.allclose(vec, [0, 0, 1]):
		tmp = np.array([0, 1, 0])
	else:
		tmp = np.array([0, 0, 1])
	right = np.cross(tmp, vec)
	right /= np.linalg.norm(right)
	up = np.cross(vec, right)
	return np.stack([right, up])


### used to calculate ACN
def segmentsIntersect(a1, a2, b1, b2):
	def orientation(p, q, r):
		return np.sign((q[0] - p[0]) * (r[1] - p[1]) -
					   (q[1] - p[1]) * (r[0] - p[0]))

	o1 = orientation(a1, a2, b1)
	o2 = orientation(a1, a2, b2)
	o3 = orientation(b1, b2, a1)
	o4 = orientation(b1, b2, a2)
	return (o1 != o2) and (o3 != o4)


### used to calculate ACN
def countCrossings(proj_chain):
	n = len(proj_chain)
	crossings = 0
	for i in range(n):
		p1, p2 = proj_chain[i], proj_chain[(i + 1) % n]
		for j in range(i + 2, n):
			if abs(i - j) == 1 or (i == 0 and j == n - 1):
				continue
			q1, q2 = proj_chain[j], proj_chain[(j + 1) % n]
			if segmentsIntersect(p1, p2, q1, q2):
				crossings += 1
	return crossings


### calculate average crossing number
def calcACN(chain, num_projections=100):
	print("Estimating ACN...")
	total_crossings = 0
	for i in range(num_projections):
		proj_matrix = generateRandomProjection()
		proj_chain = chain @ proj_matrix.T
		total_crossings += countCrossings(proj_chain)
	return total_crossings / num_projections


### run the script
if __name__ == "__main__":
	main()
	print()

