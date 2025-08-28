import armament as ars
import utils
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

## Description
# this script reads a batch of hybridization time trajectories, calculates
  # the free staple kinetics and number of hybridizations, and plots them.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',		type=str,	default=None,	help='name of copies file (first column - simulation folder names)')
	parser.add_argument('--simFold',		type=str,	default=None,	help='name of simulation folder, used if no copies file, defaults to current folder')
	parser.add_argument('--nstep_skip',		type=float,	default=0,		help='number of recorded initial steps to skip')
	parser.add_argument('--coarse_time',	type=int,	default=1,		help='coarse factor for time steps')
	parser.add_argument('--nstep_max',		type=float,	default=0,		help='max number of recorded steps to use, excluding initial frame (0 for all)')
	parser.add_argument('--mov_avg_stride',	type=int,	default=1,		help='stride length for moving average, used for derivative')
	parser.add_argument('--misFile',		type=str,	default=None,	help='name of misbinding file, which contains cutoffs and energies')
	parser.add_argument('--saveFig',		type=int,	default=False,	help='whether to save pdf of figures')

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	simFold = args.simFold
	nstep_skip = int(args.nstep_skip)
	coarse_time = args.coarse_time
	nstep_max = int(args.nstep_max)
	misFile = args.misFile
	saveFig = args.saveFig

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(copiesFile, simFold)


################################################################################
### Heart

	### get pickled data
	connFile = "analysis/connectivity_vars.pkl"
	strands, complements, n_scaf, nbead = readConn(connFile)

	### determine whether to analyze kinetics
	doKinetics = True
	stap_copies = max([ len(x) for x in complements[:n_scaf] ])
	if stap_copies != 1:
		print("Flag: multiple staple copies detected, skipping free staple kinetics analysis.")
		doKinetics = False

	### get minimum number of steps
	nstep_allSim = np.zeros(nsim,dtype=int)
	for i in range(nsim):
		datFile = simFolds[i] + "analysis/trajectory_centered.dat"
		nstep_allSim[i] = ars.getNstep(datFile, nstep_skip, coarse_time)
	nstep_min = int(min(nstep_allSim))
	nstep_use = nstep_min if nstep_max == 0 else min([nstep_min,nstep_max+1])

	### initialization
	hyb_status_allSim = np.zeros((nsim,nstep_use,nbead))

	### loop over simulations
	for i in range(nsim):

		### analyze hybridizations
		hybFile = simFolds[i] + "analysis/hyb_status.dat"; print()
		hyb_status_allSim[i], used_every_indiv = utils.readHybStatus(hybFile, nstep_skip, coarse_time, nstep_max=nstep_use, mis_status='mis', getUsedEvery=True)

		### check consistency of used step frequency
		if i == 0: used_every = used_every_indiv
		if dump_every_indiv != dump_every:
			print("Error: inconsistent dump frequencies between simulation copies.")
			sys.exit()

	### create figures folder
	if saveFig: ars.createSafeFold("analysis/figures")

	### plot native complement kinetics
	if doKinetics:
		plotKinetics(hyb_status_allSim, strands, n_scaf, dump_every)
		if saveFig: plt.savefig("analysis/figures/kinetics_native.pdf")

	### plot number of native complement scaffold hybridizations
	plotNhyb(hyb_status_allSim, n_scaf, dump_every)
	if saveFig: plt.savefig("analysis/figures/n_hyb_native.pdf")

	### misbinding analysis
	if misFile is not None:

		### plot misbond-inclusive kinetics
		if doKinetics:
			plotKinetics(hyb_status_allSim, strands, n_scaf, dump_every, True)
			if saveFig: plt.savefig("analysis/figures/kinetics_all.pdf")

		### plot total number of scaffold hybridizations
		plotNhyb(hyb_status_allSim, n_scaf, dump_every, True)
		if saveFig: plt.savefig("analysis/figures/n_hyb_all.pdf")

		### plot number of scaffold misbonds
		plotNmis(hyb_status_allSim, n_scaf, dump_every, misFile)
		if saveFig: plt.savefig("analysis/figures/n_mis.pdf")

	### display
	if not saveFig: plt.show()


################################################################################
### Plotters

### plot free staple kinetics
def plotKinetics(hyb_status_allSim, strands, n_scaf, dump_every, includeMis=False):
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]

	### figure out figure label
	figLabel = "Kinetics" if not includeMis else "Mis Kinetics"

	### calculations
	conc_avg, conc_sem, conc_min = calcKinetics(hyb_status_allSim, strands, n_scaf)
	error_min = np.maximum(conc_avg - conc_sem, conc_min*np.ones(nstep))
	error_max = conc_avg + conc_sem
	time = utils.getTime(nstep, dump_every)

	### plot
	ars.magicPlot()
	plt.figure(figLabel)
	plt.plot(time, np.log(conc_avg), label='Mean')
	if nsim > 1: plt.fill_between(time, np.log(error_min), np.log(error_max), alpha=0.3, label='SEM')
	plt.xlabel("Time [s]")
	plt.ylabel("$\\ln(C/C_0)$")
	plt.ylim(np.log(conc_min)*1.05, -np.log(conc_min)*0.05)
	plt.title("Free Staple Kinetics")
	plt.legend()


### plot number of hybridizations
def plotNhyb(hyb_status_allSim, n_scaf, dump_every, includeMis=False):
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]

	### figure out figure label
	figLabel = "N Hyb" if not includeMis else "Mis N Hyb"

	### calculations
	n_hyb_avg, n_hyb_sem = calcNhyb(hyb_status_allSim, n_scaf, includeMis)
	error_min = n_hyb_avg - n_hyb_sem
	error_max = n_hyb_avg + n_hyb_sem
	time = utils.getTime(nstep, dump_every)

	### plot
	ars.magicPlot()
	plt.figure(figLabel)
	plt.plot(time, n_hyb_avg, label='Mean')
	if nsim > 1: plt.fill_between(time, error_min, error_max, alpha=0.3, label='SEM')
	plt.axhline(y=n_scaf, color='k', linestyle='--', label='N\\textsubscript{scaffold}')
	plt.xlabel("Time [s]")
	plt.ylabel("N\\textsubscript{hyb}")
	plt.ylim(-n_scaf*0.05, n_scaf*1.05)
	plt.title("Number of Scaffold Hybridizations")
	plt.legend()


### plot number of hybridizations
def plotNmis(hyb_status_allSim, n_scaf, dump_every, misFile):
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]

	### get misbinding energies
	Us_mis = utils.readMis(misFile)[1]
	nmisBond = len(Us_mis)

	### calculate free staple concentrations
	n_mis_avg, n_mis_sem = calcNmis(hyb_status_allSim, n_scaf, nmisBond)
	error_min = n_mis_avg - n_mis_sem
	error_max = n_mis_avg + n_mis_sem
	time = utils.getTime(nstep, dump_every)

	### plot
	ars.magicPlot()
	plt.figure("N Mis")
	for m in range(nmisBond):
		plt.plot(time, n_hyb_avg[m], label=f'$U = {Us_mis[m]:0.2f}$')
		if nsim > 1: plt.fill_between(time, error_min[m], error_max[m], alpha=0.3)
	plt.xlabel("Time [s]")
	plt.ylabel("N\\textsubscript{mis}")
	plt.title("Number of Scaffold Misbonds")
	plt.legend()


################################################################################
### File Handlers

### get connectivity variables
def readConn(connFile):
	ars.testFileExist(connFile, "connectivity")
	with open(connFile, 'rb') as f:
		params = pickle.load(f)
	strands = params['strands']
	complements = params['complements']
	n_scaf = params['n_scaf']
	nbead = params['nbead']
	return strands, complements, n_scaf, nbead


################################################################################
### Calculation Managers

### calculate free staple concentrations
def calcKinetics(hyb_status_allSim, strands, n_scaf, includeMis=False):
	if includeMis: hyb_status_allSim[hyb_status_allSim>1] = 1
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]
	nbead = hyb_status_allSim.shape[2]
	nstap = max(strands)-1
	conc_min = 1/nstap
	conc_avg = np.zeros(nstep)
	conc_sem = np.zeros(nstep)
	for i in range(nstep):
		stap_freedom = np.ones((nstap,nsim),dtype=int)
		for j in range(n_scaf,nbead):
			for k in range(nsim):
				if hyb_status_allSim[k,i,j] == 1:
					stap_freedom[strands[j]-2,k] = 0
		conc_indiv = np.mean(stap_freedom,axis=1)
		conc_avg[i] = max(np.mean(conc_indiv), conc_min)
		conc_sem[i] = ars.calcSEM(conc_indiv)
	return conc_avg, conc_sem, conc_min


### calculate number of hybridizations
def calcNhyb(hyb_status_allSim, n_scaf, includeMis=False):
	if includeMis: hyb_status_allSim[hyb_status_allSim>1] = 1
	nstep = hyb_status_allSim.shape[1]
	n_hyb_avg = np.zeros(nstep)
	n_hyb_sem = np.zeros(nstep)
	for i in range(nstep):
		n_hyb_indiv = np.sum(hyb_status_allSim[:,i,:n_scaf]==1,axis=1)
		n_hyb_avg[i] = np.mean(n_hyb_indiv)
		n_hyb_sem[i] = ars.calcSEM(n_hyb_indiv)
	return n_hyb_avg, n_hyb_sem


### calculate number of misbonds
def calcNmis(hyb_status_allSim, n_scaf, nmisBond):
	nstep = hyb_status_allSim.shape[1]
	n_mis_avg = np.zeros((nmisBond,nstep))
	n_mis_sem = np.zeros((nmisBond,nstep))
	for m in range(nmisBond):
		for i in range(nstep):
			hyb_val = 1+(m+1)/100
			n_mis_indiv = np.sum(hyb_status_allSim[:,i,:n_scaf]==hyb_val,axis=1)
			n_mis_avg[m,i] = np.mean(n_mis_indiv)
			n_mis_sem[m,i] = ars.calcSEM(n_mis_indiv)
	return n_mis_avg, n_mis_sem


### run the script
if __name__ == "__main__":
	main()
	print()

