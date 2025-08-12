import armament as ars
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
	parser.add_argument('--copiesFile',		type=str,	default=None,	help='name of copies file, which contains a list of simulation folders')	
	parser.add_argument('--simFold',		type=str,	default=None,	help='name of simulation folder, should exist within current directory')
	parser.add_argument('--rseed',			type=int,	default=1,		help='random seed, used to find simFold if necessary')
	parser.add_argument('--clusterFile',	type=str,	default=None,	help='name of cluster file, only used for crystallinity')	
	parser.add_argument('--calcRMSD',		type=str,	default=False,	help='whether to calculate RMSD')		
	parser.add_argument('--cadFile',		type=str,	default=None,	help='if calculating RMSD, name of caDNAno file')	
	parser.add_argument('--calcACN',		type=str,	default=False,	help='whether to calculate ACN')		
	parser.add_argument('--loadResults',	type=int,	default=False,	help='whether to load the results from a pickle file')
	parser.add_argument('--nstep_skip',		type=float,	default=0,		help='if not loading results, number of recorded initial steps to skip')
	parser.add_argument('--nstep_max',		type=float,	default=0,		help='if not loading results, max number of recorded steps to use (0 for all)')
	parser.add_argument('--coarse_time',	type=int,	default=1,		help='if not loading results, coarse factor for time steps')
	parser.add_argument('--mov_avg_stride',	type=int,	default=1,		help='stride length for moving average')
	parser.add_argument('--saveFig',		type=int,	default=False,	help='whether to save pdf of figures')

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	simFold = args.simFold
	rseed = args.rseed
	clusterFile = args.clusterFile
	calcRMSD = args.calcRMSD
	cadFile = args.cadFile
	calcACN = args.calcACN
	loadResults = args.loadResults
	nstep_skip = int(args.nstep_skip)
	nstep_max = int(args.nstep_max)
	coarse_time = args.coarse_time
	mov_avg_stride = args.mov_avg_stride
	saveFig = args.saveFig

	### check arguments
	if copiesFile is not None and clusterFile is not None:
		clusterFile = None
		print("Flag: Both copies and cluster file provided, so ignoring cluster file.")
	if calcRMSD and cadFile is None:
		print("Error: caDNAno file required for RMSD calculation.")
		sys.exit()
	if calcRMSD and clusterFile is not None:
		print("Flag: skipping RMSD calculation, clusters not supported yet")
		calcRMSD = False


################################################################################
### Heart

	### read data and calculate results
	if not loadResults:

		### get simulation folders
		simFolds, nsim = utils.getSimFolds(copiesFile, simFold, rseed)

		### get pickled data
		connFile = "analysis/connectivity_vars.pkl"
		n_scaf = readConn(connFile)

		### get ideal positions
		if calcRMSD:
			r = utils.initPositionsCaDNAno(cadFile)[0]
			r_ideal = r[:n_scaf]

		### get minimum number of steps
		nstep_allSim = np.zeros(nsim,dtype=int)
		for i in range(nsim):
			datFile = simFolds[i] + "analysis/trajectory_centered.dat"
			nstep_allSim[i] = ars.getNstep(datFile, nstep_skip, coarse_time)
		nstep_min = int(min(nstep_allSim))
		nstep_use = nstep_min if nstep_max == 0 else min([nstep_min,nstep_max])

		### get trajectory dump frequency
		datFile = simFolds[0] + "analysis/trajectory_centered.dat"
		dump_every = ars.getDumpEvery(datFile)*coarse_time

		### whole scaffold analysis
		if clusterFile is None:
			bdis = list(range(1,n_scaf+1))

			### loop through simulations
			S_allSim = np.zeros((nsim,nstep_use))
			RMSD_allSim = np.zeros((nsim,nstep_use))
			ACN_allSim = np.zeros(nsim)
			for i in range(nsim):

				### calculate crustallinity
				datFile = simFolds[i] + "analysis/trajectory_centered.dat"
				points, _, dbox = ars.readAtomDump(datFile, nstep_skip, coarse_time, bdis=bdis, nstep_max=nstep_use); print()
				S_allSim[i] = utils.calcCrystallinity(points, dbox)
				if calcRMSD:
					RMSD_allSim[i] = utils.calcRMSD(points, r_ideal)
				if calcACN:
					ACN_allSim[i] = estimate_acn(points[-1])

		### clustered scaffold analysis
		else:
			bdis = ars.readCluster(clusterFile)
			ncluster = len(bdis)

			### read trajectory
			datFile = simFolds[0] + "analysis/trajectory_centered.dat"
			points, _, dbox, molecules = ars.readAtomDump(datFile, nstep_skip, coarse_time, bdis=bdis, nstep_max=nstep_use); print()
			points = ars.sortPointsByMolecule(points, molecules)

			### loop through clusters
			S_allSim = np.zeros((ncluster,nstep_use))
			RMSD_allSim = None
			ACN_allSIm = None
			for i in range(ncluster):

				### calculate crystallinity
				S_allSim[i] = utils.calcCrystallinity(points[i], dbox)

		### store results
		resultsFile = "analysis/crystallinity_vars.pkl"
		with open(resultsFile, 'wb') as f:
			pickle.dump([S_allSim, RMSD_allSim, ACN_allSim, dump_every], f)

	### load results
	else:
		resultsFile = "analysis/crystallinity_vars.pkl"
		ars.testFileExist(resultsFile,"results")
		with open(resultsFile, 'rb') as f:
			[S_allSim, RMSD_allSim, ACN_allSim, dump_every] = pickle.load(f)


################################################################################
### Results

	### apply moving average
	for i in range(len(S_allSim)):
		S_allSim[i] = ars.movingAvg(S_allSim[i], mov_avg_stride)

	### report best simulations
	if copiesFile is not None:
		copyNames = ars.readCopies(copiesFile)[0]
		sorted_results = sorted(zip(S_allSim[:,-1], ACN_allSim, copyNames), reverse=True)
		for S_final, ACN, copyName in sorted_results:
			print(f"{copyName}: {S_final:0.4f}, {ACN:0.4f}")

	### RMSD
	if calcRMSD:
		plotRMSD(RMSD_allSim, dump_every)
		if saveFig: plt.savefig("analysis/RMSD.pdf")

	### knotty
	if calcACN:
		plotACNvsS(ACN_allSim, S_allSim)
		if saveFig: plt.savefig("analysis/ACN.pdf")

	### crystallinity
	plotCrystallinity(S_allSim, dump_every)
	if saveFig: plt.savefig("analysis/crystallinity.pdf")

	### display
	if not saveFig: plt.show()


################################################################################
### Plotters

### plot average Landau-De Gennes crystallinity parameter
def plotCrystallinity(S_allSim, dump_every):
	nsim = S_allSim.shape[0]
	nstep = S_allSim.shape[1]
	plotSEM = False

	### calculations
	S_avg = np.mean(S_allSim, axis=0)
	S_sem = np.zeros(nstep)
	for i in range(nstep):
		S_sem[i] = ars.calcSEM(S_allSim[:,i])

	### calculate time
	dt = 0.01
	scale = 5200
	time = np.arange(nstep)*dump_every*dt*scale*1E-9

	### plot prep
	cmap = cm.viridis
	norm = mcolors.Normalize(vmin=1, vmax=nsim)
	ranks = [ sorted(S_allSim[:,-1]).index(x) + 1 for x in S_allSim[:,-1] ]

	### plot
	ars.magicPlot()
	plt.figure("S")
	if nsim > 1:
		for i in range(nsim):
			color = cmap(norm(ranks[i]))
			plt.plot(time,S_allSim[i],color=color,linewidth=2,alpha=0.6)
			plt.plot(time[-1],S_allSim[i,-1],'o',color=color)
		if plotSEM: plt.fill_between(time,S_avg-S_sem,S_avg+S_sem,color='k',alpha=0.4,edgecolor='none')
	plt.plot(time,S_avg,'k',linewidth=4)
	plt.plot(time[-1],S_avg[-1],'o',color='k')
	plt.ylim(0,0.8)
	plt.xlabel("Time [$s$]")
	plt.ylabel("Crystallinity")


### plot average crossing number
def plotACNvsS(ACN_allSim, S_allSim):
	ars.magicPlot()
	plt.figure("ACN")
	plt.scatter(ACN_allSim,S_allSim[:,-1])
	plt.xlabel("Average Crossing Number")
	plt.ylabel("Final Crystallinity")


### plot average Landau-De Gennes crystallinity parameter
def plotRMSD(RMSD_allSim, dump_every):
	nsim = RMSD_allSim.shape[0]
	nstep = RMSD_allSim.shape[1]
	plotSEM = False

	### calculations
	RMSD_avg = np.mean(RMSD_allSim, axis=0)
	RMSD_sem = np.zeros(nstep)
	for i in range(nstep):
		RMSD_sem[i] = ars.calcSEM(RMSD_allSim[:,i])

	### calculate time
	dt = 0.01
	scale = 5200
	time = np.arange(nstep)*dump_every*dt*scale*1E-9

	### plot prep
	cmap = cm.viridis
	norm = mcolors.Normalize(vmin=1, vmax=nsim)
	ranks = [ sorted(RMSD_allSim[:,-1]).index(x) + 1 for x in RMSD_allSim[:,-1] ]

	### plot
	ars.magicPlot()
	plt.figure("RMSD")
	if nsim > 1:
		for i in range(nsim):
			color = cmap(norm(ranks[i]))
			plt.plot(time,RMSD_allSim[i],color=color,linewidth=2,alpha=0.6)
			plt.plot(time[-1],RMSD_allSim[i,-1],'o',color=color)
		if plotSEM: plt.fill_between(time,RMSD_avg-RMSD_sem,RMSD_avg+RMSD_sem,color='k',alpha=0.4,edgecolor='none')
	plt.plot(time,RMSD_avg,'k',linewidth=4)
	plt.plot(time[-1],RMSD_avg[-1],'o',color='k')
	plt.xlabel("Time [$s$]")
	plt.ylabel("$RMSD$")


################################################################################
### File Handlers

def readConn(connFile):
	ars.testFileExist(connFile, "connectivity")
	with open(connFile, 'rb') as f:
		params = pickle.load(f)[0]
	n_scaf = params['n_scaf']
	return n_scaf


################################################################################
### Knotty

def generate_random_projection():
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

def project_chain(chain, proj_matrix):
	return chain @ proj_matrix.T

def fast_segments_intersect(a1, a2, b1, b2):
	def orientation(p, q, r):
		return np.sign((q[0] - p[0]) * (r[1] - p[1]) -
					   (q[1] - p[1]) * (r[0] - p[0]))

	o1 = orientation(a1, a2, b1)
	o2 = orientation(a1, a2, b2)
	o3 = orientation(b1, b2, a1)
	o4 = orientation(b1, b2, a2)

	return (o1 != o2) and (o3 != o4)

def count_crossings(proj_chain):
	n = len(proj_chain)
	crossings = 0
	for i in range(n):
		p1, p2 = proj_chain[i], proj_chain[(i + 1) % n]
		for j in range(i + 2, n):
			if abs(i - j) == 1 or (i == 0 and j == n - 1):
				continue
			q1, q2 = proj_chain[j], proj_chain[(j + 1) % n]
			if fast_segments_intersect(p1, p2, q1, q2):
				crossings += 1
	return crossings

def estimate_acn(chain, num_projections=100):
	total_crossings = 0
	for _ in range(num_projections):
		proj_matrix = generate_random_projection()
		proj_chain = project_chain(chain, proj_matrix)
		total_crossings += count_crossings(proj_chain)
	return total_crossings / num_projections


### run the script
if __name__ == "__main__":
	main()
	print()

