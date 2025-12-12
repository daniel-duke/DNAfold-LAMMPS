import armament as ars
import utils
import utilsLocal
import parameters
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import brentq
from collections import Counter
import pickle
import itertools
import argparse
import copy
import json
import sys
import os

## Description
# this script takes a caDNAno json file, creates the interaction and geometry
  # arrays necessary for the dnafold model, and writes the geometry and input
  # files necessary to simulate the system in lammps.
# all indexing starts at 0, then is increased by 1 when written to lammps files.
# the strand indexing for a reserved staples file starts at 1, but since the
  # first strand is always the scaffold, the indexing effectively starts at 2.
# in addition to the packages listed above, the nupack package is required if
  # including misbinding with either energy function or position optimization.
# required arguments from input file: cadFile, nstep, dbox

## To Do
# add cpu thermo output
# distinguish comm cutoff from force cutoff


################################################################################
### Parameters

def main():

	### where to get files
	useDanielFiles = False

	### special code to make Daniel happy
	if useDanielFiles:

		### chose design
		desID = "2HBx4"				# design identification
		simTag = ""					# appended to desID to get name of output folder
		simType = "experiment"		# prepended to desID to get name of output folder within standard location
		rstapTag = None				# tag for reserved staples file (None for not reserving staples)
		confTag = None				# if starting bound, tag for oxDNA configuration file (None for caDNAno positions)
		rseed = 1					# random seed, used for initializing positions and LAMMPS thermostat, also used for naming simulation folders
		nsim = 1					# number of simulations, starting with given random seed and incrementing by 1 for each simulation

		### choose parameters
		nstep			= 1E7		# steps		- number of production steps
		nstep_relax		= 1E5		# steps		- number of relaxation steps
		dump_every		= 1E4		# steps		- number of steps between positions dumps
		dbox			= 40		# nm		- periodic boundary diameter
		stap_copies		= 1 		# int		- number of copies for each staple
		circularScaf	= True		# bool		- whether the scaffold is circular
		scaf_shift		= 0			# int		- if linear scaffold, bead shift for cut location (3' end chasing 5' end)
		forceBind		= False		# bool		- whether to force hybridization (not applied if >1 staple copies)
		startBound		= False		# bool		- whether to start at caDNAno positions
		nmisBond		= 0			# int		- number of misbinding levels (0 for no misbinding)
		rseed_mis 		= 1			# int		- random seed for misbinding (None to match general random seed)

		### get input files
		cadFile = utilsLocal.getCadFile(desID)
		rstapFile = utilsLocal.getRstapFile(desID, rstapTag) if rstapTag is not None else None
		oxFiles = utilsLocal.getOxFiles(desID, confTag) if confTag is not None else None

		### set parameters
		cadFile, rstapFile, oxFiles, p = readInput(None, rseed, cadFile, rstapFile, oxFiles, nstep, nstep_relax, dump_every, dbox, stap_copies, circularScaf, scaf_shift, forceBind, startBound, nmisBond, rseed_mis)
		
		### set output folder
		outFold = utilsLocal.getSimHomeFold(desID, simTag, simType)
		ars.createSafeFold(outFold)

		### set simulation folders
		outSimFolds = utilsLocal.writeCopies(outFold, p.rseed, nsim)

		### set random seeds
		rseeds = np.arange(rseed,rseed+nsim)

		### copy design files to output folder
		outCadFile = outFold + desID + ".json"
		os.system(f"cp \"{cadFile}\" \"{outCadFile}\"")
		if p.reserveStap:
			outRstapFile = outFold + "rstap_" + desID + rstapTag + ".txt"
			os.system(f"cp \"{rstapFile}\" \"{outRstapFile}\"")

	### regular code for the general populace
	if not useDanielFiles:

		### get arguments
		parser = argparse.ArgumentParser()
		parser.add_argument('--inFile',		type=str,	required=True,	help='name of input file, which contains file names and parameters')
		parser.add_argument('--copiesFile',	type=str,	default=None,	help='name of copies file (first column - simulation folder names; second (optional) column - random seeds)')
		parser.add_argument('--simFold',	type=str,	default=None,	help='name of simulation folder, only used if no copies file, defaults to current folder')
		parser.add_argument('--rseed',		type=int,	default=None,	help='random seed, for initializing positions and LAMMPS thermostat, only used if copies file does not contain random seeds')
		
		### set arguments
		args = parser.parse_args()
		inFile = args.inFile
		copiesFile = args.copiesFile
		simFold = args.simFold
		rseed = args.rseed

		### set parameters
		cadFile, rstapFile, oxFiles, p = readInput(inFile, rseed)

		### set output folder
		outFold = "./"

		### set simulation folders
		outSimFolds, nsim = utils.getSimFolds(copiesFile, simFold)

		### set random seeds
		rseeds = utils.getRseeds(copiesFile, rseed)
		

################################################################################
### Heart

	### read caDNAno file and build DNAfold model
	strands, backbone_neighbors, complements, is_crossover, p = buildDNAfoldModel(cadFile, p)
	complements, is_crossover = shiftScaffold(complements, is_crossover, p)
	
	### read reserved staples file
	is_reserved_strand = readRstap(rstapFile, p)

	### calculate complementary factors
	comp_factors, mis_d2_cuts = calcCompFactors(complements, p)

	### record parameters
	paramsFile = outFold + "parameters.txt"
	p.record(paramsFile, rseeds, rseed_mis)

	### record misbinding cutoffs and energies
	outMisFile = outFold + "misbinding.txt"
	writeMisbinding(outMisFile, mis_d2_cuts, p)
	
	### loop over simulations
	for i in range(nsim):

		### adjust random seed
		p.rseed = rseeds[i]
		p.rng = np.random.default_rng(p.rseed)

		### adjust misbinding random seed
		if rseed_mis == None:
			p.rseed_mis = rseeds[i]
			p.rng_mis = np.random.default_rng(p.rseed_mis)

		### create simulation folder
		outSimFold = outSimFolds[i]; print()
		ars.createSafeFold(outSimFold)

		### write geometry files
		r, nhyb, nangle = composeGeo(outSimFold, strands, backbone_neighbors, complements, is_crossover, is_reserved_strand, comp_factors, cadFile, oxFiles, p)
		composeGeoVis(outSimFold, strands, backbone_neighbors, r, p)

		### create reaction folder
		outReactFold = outSimFold + "react/"
		ars.createEmptyFold(outReactFold)

		### write react files
		nreact_bondHyb = writeReactHybBond(outReactFold, mis_d2_cuts, p)
		nreact_bondMis = writeReactMisBond(outReactFold, mis_d2_cuts, p)
		nreact_angleAct = writeReactAngleAct(outReactFold, backbone_neighbors, complements, is_crossover, p)
		nreact_angleDeact = writeReactAngleDeact(outReactFold, backbone_neighbors, complements, is_crossover, p)
		nreact_bridge, nreact_unbridge = writeReactBridge(outReactFold, backbone_neighbors, complements, is_crossover, p)

		### write lammps input file
		writeInput(outSimFold, nhyb, nangle, nreact_bondHyb, nreact_bondMis, nreact_angleAct, nreact_angleDeact, nreact_bridge, nreact_unbridge, p)

	### reminder of any flags
	if p.paramFlag:
		print("\n!!! Warning !!!")
		print("At least one flag was raised when setting parameters.")


################################################################################
### File Handlers

### read input file
def readInput(inFile=None, rseed=1, cadFile=None, rstapFile=None, oxFiles=None, nstep=None, nstep_relax=1E5, dump_every=1E4, dbox=None, stap_copies=1, circularScaf=True, scaf_shift=0, forceBind=False, startBound=False, nmisBond=0, rseed_mis=1):

	### set oxDNA files
	topFile = oxFiles[0] if oxFiles is not None else None
	confFile = oxFiles[1] if oxFiles is not None else None

	### list keys that can have 'None' as their final value
	allow_none_default = {'rstapFile','topFile','confFile','rseed_mis'}

	### define parameters with their default values
	param_defaults = {
		'cadFile':			cadFile,		# str			- name of caDNAno file (required)
		'rstapFile':		rstapFile,		# str			- name of reserved staples file
		'topFile':			topFile,		# str			- name of topology file
		'confFile':			confFile,		# str			- name of configuration file
		'nstep': 			nstep,			# steps			- number of production steps (required)
		'nstep_relax': 		nstep_relax,	# steps			- number of relaxation steps
		'dump_every': 		dump_every,		# steps			- number of steps between positions dumps
		'dt': 				0.01,			# ns			- integration time step
		'dbox': 			dbox,			# nm			- periodic boundary diameter (required)
		'stap_copies': 		stap_copies,	# int			- number of copies for each staple
		'circularScaf':		circularScaf,	# bool			- whether the scaffold is circular
		'scaf_shift':		scaf_shift,		# int			- if linear scaffold, bead shift for cut location (3' end chasing 5' end)
		'forceBind': 		forceBind,		# bool			- whether to force hybridization (not applied if >1 staple copies)
		'startBound': 		startBound,		# bool			- whether to start at caDNAno positions
		'nmisBond': 		nmisBond,		# int			- number of misbinding levels (0 for no misbinding)
		'ncompFactor':		2,				# int			- number of complementary factors (set to 1 if no misbinding)
		'optCompFactors': 	False,			# bool			- whether to optimize complementary factors for misbinding
		'optCompEfunc': 	False,			# bool			- whether to optimize energy function for misbinding
		'rseed_mis': 		rseed_mis,		# int			- random seed for misbinding (None to match general random seed)
		'bridgeEnds':		False,			# bool			- whether to include end bridging reactions (if applicable)
		'dehyb': 			True,			# bool			- whether to include dehybridization reactions
		'debug': 			False,			# bool			- whether to include debugging output
		'T':				300,			# K				- temperature
		'T_relax':			600,			# K				- temperature for relaxation (set to 300 if starting bound)
		'r_h_bead':			1.28,			# nm			- hydrodynamic radius of single bead
		'visc':				0.8472,			# mPa/s			- viscosity (units equivalent to pN*ns/nm^2) (default value for 300K)
		'sigma':			2.14,			# nm			- WCA distance parameter
		'epsilon':			4.0,			# kcal/mol		- WCA energy parameter
		'r12_eq':			2.72,			# nm			- equilibrium bead separation
		'k_x': 				120.0,			# kcal/mol/nm2	- backbone spring constant (standard definition)
		'r12_cut_hyb':		2.0,			# nm			- hybridization potential cutoff radius
		'U_hyb':			10.0,			# kcal/mol		- depth of hybridization potential
		'U_mis_max':		12.0,			# kcal/mol		- strongest magnitude of misbinding energy allowed
		'U_mis_min':		6.0,			# kcal/mol		- weakest magnitude of misbinding energy to consider
		'U_mis_shift':		2.0,			# kcal/mol		- energy shift towards weaker misbinding
		'dsLp': 			50.0,			# nm			- persistence length of dsDNA
	}

	### define types for each parameter
	param_types = {
		'cadFile':			str,
		'rstapFile':		str,
		'topFile':			str,
		'confFile':			str,
		'nstep':			lambda x: int(float(x)),
		'nstep_relax':		lambda x: int(float(x)),
		'dump_every':		lambda x: int(float(x)),
		'dt':				float,
		'dbox':				float,
		'stap_copies':		int,
		'circularScaf':		lambda x: x.lower() == 'true',
		'scaf_shift':		int,
		'forceBind':		lambda x: x.lower() == 'true',
		'startBound':		lambda x: x.lower() == 'true',
		'nmisBond':			int,
		'ncompFactor':		int,
		'optCompFactors':	lambda x: x.lower() == 'true',
		'optCompEfunc':		lambda x: x.lower() == 'true',
		'rseed_mis':		int,
		'bridgeEnds':		lambda x: x.lower() == 'true',
		'dehyb':			lambda x: x.lower() == 'true',
		'debug':			lambda x: x.lower() == 'true',
		'T':				float,
		'T_relax':			float,
		'r_h_bead':			float,
		'visc':				float,
		'sigma':			float,
		'epsilon':			float,
		'r12_eq':			float,
		'k_x':				float,
		'r12_cut_hyb':		float,
		'U_hyb':			float,
		'U_mis_max':		float,
		'U_mis_min':		float,
		'U_mis_shift':		float,
		'dsLp':				float,
	}

	### store parsed parameters
	params = {}

	### read parameters from file
	if inFile is not None:
		ars.checkFileExist(inFile,'input')
		with open(inFile, 'r') as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith('#'):
					continue
				if '=' not in line:
					print(f"Error: Invalid line in input file: {line}.\n")
					sys.exit()
				key, value = map(str.strip, line.split('=', 1))
				if key not in param_defaults:
					print(f"Error: Unknown parameter: {key}.\n")
					sys.exit()
				try:
					params[key] = param_types[key](value)
				except:
					print(f"Error: Cannot parse value for '{key}': {value}.\n")
					sys.exit()

	### apply defaults and check for required values
	for key, default in param_defaults.items():
		if key not in params:
			if default is None:
				if key not in allow_none_default:
					print(f"Error: Missing required parameter: {key}.\n")
					sys.exit()
			params[key] = default

	### get caDNAno file
	cadFile = params['cadFile']
	del params['cadFile']
	ars.checkFileExist(cadFile,"caDNAno")

	### get reserved staples file
	rstapFile = params['rstapFile']
	del params['rstapFile']
	ars.checkFileExist(cadFile,"reserved staples")
	params['reserveStap'] = False if rstapFile is None else True

	### get topology file
	topFile = params['topFile']
	del params['topFile']
	if topFile is not None and not startBound:
		print("Flag: oxDNA topology file given but not used.")

	### get conformation file
	confFile = params['confFile']
	del params['confFile']
	if confFile is not None and not startBound:
		print("Flag: oxDNA configuration file given but not used.")

	### set oxDNA files
	if topFile is not None and confFile is not None:
		oxFiles = [topFile, confFile]
	elif topFile is not None:
		print("Flag: oxDNA topology file given without configuration file, not using.")
	elif confFile is not None:
		print("Flag: oxDNA configuration file given without topology file, not using.")

	### edit default if no misbinding
	if params['nmisBond'] == 0:
		params['ncompFactor'] = 1

	### edit default if starting bound
	if params['startBound'] == True:
		params['T_relax'] = 300

	### add random seed
	params['rseed'] = rseed

	### convert into parameters class and return
	p = parameters.parameters(params)
	return cadFile, rstapFile, oxFiles, p


### read reserved staples file
def readRstap(rstapFile, p):
	is_reserved_strand = [ False for i in range(p.nstrand) ]

	### skip if not reserving staples
	if not p.reserveStap:
		return is_reserved_strand

	### read staples
	ars.checkFileExist(rstapFile,"reserved staples")
	with open(rstapFile, 'r') as f:
		reserved_strands = [ int(line.strip())-1 for line in f ]
	for si in range(len(reserved_strands)):
		is_reserved_strand[reserved_strands[si]] = True

	### return strand reservations status
	return is_reserved_strand


### write lammps geometry file, for simulation
def composeGeo(outSimFold, strands, backbone_neighbors, complements, is_crossover, is_reserved_strand, comp_factors, cadFile, oxFiles, p):
	print("Writing simulation geometry file...")

	### initailize positions
	if p.startBound:
		stap_offset = 0.01
		if oxFiles is not None:
			r = utils.initPositionsOxDNA(cadFile, oxFiles[0], oxFiles[1])[0]
		else:
			r = utils.initPositionsCaDNAno(cadFile)[0]
		r[p.n_scaf:] += stap_offset
		if p.reserveStap:
			r = randomStapPositions(r, strands, is_reserved_strand, p)
	else:
		r = initPositions(strands, p)
	r = np.append(r,np.zeros((1,3)),axis=0)

	### initialize
	molecules = np.ones(p.nbead+1,dtype=int)
	types = np.ones(p.nbead+1,dtype=int)
	charges = np.ones(p.nbead+1)
	bonds = np.zeros((0,3),dtype=int)
	angles = np.zeros((0,4),dtype=int)

	### scaffold atoms
	for bi in range(p.n_scaf):
		charges[bi] = is_crossover[bi] + 1

	### staple atoms
	for ci in range(p.stap_copies):
		for obi in range(p.n_scaf,p.n_ori):
			rbi = obi + ci*p.n_stap
			molecules[rbi] = strands[obi] + ci*(p.nstrand-1) + 1
			types[rbi] = 2
			if is_reserved_strand[strands[obi]] and not (p.startBound or p.forceBind):
				types[rbi] = 3
				r[rbi] = [0,0,0]

	### dummy atom
	molecules[p.nbead] = 0
	types[p.nbead] = 3
	charges[p.nbead] = 0

	### scaffold backbone bonds
	for bi in range(p.n_scaf-1):
		type = 1
		atom1 = bi + 1
		atom2 = bi + 2
		bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### scaffold end-to-end bond
	if p.circularScaf:
		bonds = np.append(bonds,[[1,p.n_scaf,1]],axis=0)

	### staple backbone bonds
	for ci in range(p.stap_copies):
		for obi in range(p.n_scaf,p.n_ori):
			if backbone_neighbors[obi][1] != -1:
				if is_reserved_strand[strands[obi]] and not (p.startBound or p.forceBind):
					type = 3 + p.nmisBond
				else:
					type = 1
				atom1 = obi + ci*p.n_stap + 1
				atom2 = backbone_neighbors[obi][1] + ci*p.n_stap + 1
				bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### hybridization bonds
	nhyb = 0
	for bi in range(p.n_scaf):
		if complements[bi] != -1:
			nhyb += 1
			if not is_reserved_strand[strands[complements[bi]]] and (p.forceBind or p.startBound):
				type = 2
				atom1 = bi + 1
				atom2 = complements[bi] + 1
				bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)
				charges[complements[bi]] = 2

	### angles
	nangle = 0
	for bi in range(p.n_scaf):
		bi_5p,bi_3p = getAssembledNeighbors(bi, backbone_neighbors, complements, p)
		if bi_5p != -1 and bi_3p != -1:
			if complements[bi_5p] != -1 and complements[bi] != -1 and complements[bi_3p] != -1:
				nangle += 1
				if p.startBound or p.forceBind:
					if not is_reserved_strand[strands[complements[bi_5p]]] and not is_reserved_strand[strands[complements[bi]]] and not is_reserved_strand[strands[complements[bi_3p]]]:
						if p.circularScaf or (bi != 0 and bi != p.n_scaf-1):
							type = is_crossover[bi] + 1
							atom1 = bi_5p + 1
							atom2 = bi + 1
							atom3 = bi_3p + 1
							angles = np.append(angles,[[type,atom1,atom2,atom3]],axis=0)
							charges[bi] = type + 2

	### count bond types
	nbondType = 3 + p.nmisBond

	### write geometry file
	outGeoFile = outSimFold + "geometry.in"
	ars.writeGeo(outGeoFile, p.dbox, r, molecules, types, bonds, angles, nbondType=nbondType, nangleType=2, charges=charges, extras=comp_factors)

	### return positions (without dummy atom)
	r = r[0:p.nbead]
	return r, nhyb, nangle


### write lammps geometry file, for visualization
def composeGeoVis(outSimFold, strands, backbone_neighbors, r, p):

	### initialize
	molecules = np.ones(p.nbead,dtype=int)
	types = np.ones(p.nbead,dtype=int)
	bonds = np.zeros((0,3),dtype=int)

	### compile atom information
	for ci in range(p.stap_copies):
		for obi in range(p.n_scaf,p.n_ori):
			rbi = obi + ci*p.n_stap
			molecules[rbi] = strands[obi] + ci*(p.nstrand-1) + 1
			types[rbi] = 2

	### scaffold backbone bonds
	for bi in range(p.n_scaf-1):
		type = 1
		atom1 = bi + 1
		atom2 = bi + 2
		bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### scaffold end-to-end bond
	if p.circularScaf:
		bonds = np.append(bonds,[[1,p.n_scaf,1]],axis=0)

	### staple backbone bonds
	for ci in range(p.stap_copies):
		for obi in range(p.n_scaf,p.n_ori):
			if backbone_neighbors[obi][1] != -1:
				type = 1
				atom1 = obi + ci*p.n_stap + 1
				atom2 = backbone_neighbors[obi][1] + ci*p.n_stap + 1
				bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### write file
	outGeoFile = outSimFold + "geometry_vis.in"
	ars.writeGeo(outGeoFile, p.dbox, r, molecules, types, bonds)


### write file that contains misbinding cutoffs
def writeMisbinding(outMisFile, mis_d2_cuts, p):
	if p.nmisBond == 0:
		return
	with open(outMisFile, 'w') as f:
		for i in range(p.nmisBond):
			U_hyb = p.U_mis_max - (i+0.5)*(p.U_mis_max-p.U_mis_min)/p.nmisBond - p.U_mis_shift
			f.write(f"{mis_d2_cuts[i+1]:0.4f} {U_hyb/6.96:0.2f}\n")


### write input file for lammps
def writeInput(outSimFold, nhyb, nangle, nreact_bondHyb, nreact_bondMis, nreact_angleAct, nreact_angleDeact, nreact_bridge, nreact_unbridge, p):
	print("Writing input file...")

	### computational parameters
	verlet_skin				= 4		# nm			- width of neighbor list skin (= r12_cut - sigma)
	neigh_every				= 10	# steps			- how often to consider updating neighbor list
	bond_res 				= 0.1	# nm			- distance between tabular bond interpolation points
	react_every_bondHyb		= 1E2	# steps			- how often to check for new hybridization bonds
	react_every_bondMis		= 1E4	# steps			- how often to check for misbinding
	react_every_angleAct	= 1E4	# steps			- how often to check for new angles
	react_every_angleDeact	= 1E2	# steps			- how often to check for removing angles
	react_every_bondDehyb	= 1E2	# steps			- how often to check for removing hybridization bonds
	react_every_bridge		= 1E2	# steps			- how often to check for bridge creation or destruction
	r12_cut_react_hybBond	= 3		# nm			- cutoff radius for potential hybridization bonds
	r12_cut_react_bridge	= 3		# nm			- cutoff radius for bridging reactions
	comm_cut				= 12	# nm			- communication cutoff (relevant for parallelization)
	U_barrier_comm			= 10	# kcal/mol		- energy barrier to exceeding communication cutoff
	F_forceBind				= 1		# kcal/mol/nm	- force to apply for forced binding

	### check neighbor list cutoff
	if p.r12_cut_WCA + verlet_skin < r12_cut_react_hybBond:
		print("Flag: Hybridization reaction cutoff exceeds neighbor list cutoff.")
	if p.r12_cut_WCA + verlet_skin < r12_cut_react_bridge:
		print("Flag: Bridging cutoff exceeds neighbor list cutoff.")

	### adjust comm_cut for forced binding
	if p.forceBind:
		comm_cut = int(np.sqrt(3)*p.dbox+1)

	### number of bond dybridization reactions
	nreact_bondDehyb = nreact_bondHyb
	if not p.dehyb:
		nreact_bondDehyb = 0

	### write table for full hybridization bond
	npoint_hybBond = writeHybBond(outSimFold, "hybFull", p.U_hyb, F_forceBind, comm_cut, U_barrier_comm, bond_res, p)

	### write table for partial hybridization bonds
	for i in range(p.nmisBond):
		U_hyb = p.U_mis_max - (i+0.5)*(p.U_mis_max-p.U_mis_min)/p.nmisBond - p.U_mis_shift
		writeHybBond(outSimFold, f"hybPart{i+1}", U_hyb, F_forceBind, comm_cut, U_barrier_comm, bond_res, p)

	### count digits
	len_nrbM = len(str(nreact_bondMis))
	len_nraH = len(str(nreact_angleAct))

	### open file
	outLammpsFile = outSimFold + "lammps.in"
	with open(outLammpsFile, 'w') as f:

		### header
		f.write(
			"\n#------ Begin Input ------#\n"
			"# Written by dnafold_lmp.py\n\n")

		### initialize environment
		f.write(
			"## Environment\n"
			"units           nano\n"
			"dimension       3\n"
			"boundary        p p p\n"
			"atom_style      full\n\n")

		### read geometry data
		f.write(
			"## Geometry\n"
			"fix CFs         all property/atom")
		for i in range(p.ncompFactor): f.write(f" d_CF{i+1}")
		f.write("\n"
			"read_data       geometry.in fix CFs NULL Extras &\n"
			"                extra/bond/per/atom 10 &\n"
			"                extra/angle/per/atom 10 &\n"
			"                extra/special/per/atom 100\n\n")

		### neighbor list
		f.write(
			"## Parameters\n"
		   f"neighbor        {verlet_skin:0.2f} bin\n"
		   f"neigh_modify    every {int(neigh_every)}\n"
		   f"pair_style      hybrid zero 0.0 lj/cut {p.r12_cut_WCA:0.2f}\n"
			"pair_modify     pair lj/cut shift yes\n"
		   f"pair_coeff      * * lj/cut {p.epsilon:0.2f} {p.sigma:0.2f} {p.r12_cut_WCA:0.2f}\n"
		   f"pair_coeff      * 3 zero\n"
			"special_bonds   lj 0.0 1.0 1.0\n"
		   f"comm_modify     cutoff {comm_cut}\n")

		### basic bonded interactions
		f.write(
		   f"bond_style      hybrid zero harmonic table linear {npoint_hybBond}\n"
		   f"bond_coeff      1 harmonic {p.k_x/2:0.2f} {p.r12_eq:0.2f}\n"
		   f"bond_coeff      2 table bond_hybFull.txt hybFull\n")

		### misbinding bonds
		for i in range(p.nmisBond): f.write(
		   f"bond_coeff      {i+3} table bond_hybPart{i+1}.txt hybPart{i+1}\n")

		### dummy bond (for reserved staples)
		f.write(
		   f"bond_coeff      {p.nmisBond+3} zero\n")

		### angled interactions
		f.write(
		   f"angle_style     harmonic\n"
		   f"angle_coeff     1 {p.k_theta/2:0.2f} 180\n"
		   f"angle_coeff     2 {p.k_theta/2:0.2f} 90\n")

		### group atoms
		for i in range(p.ncompFactor): f.write(
		   f"variable        varCF{i+1} atom d_CF{i+1}\n")
		f.write(
			"variable        varQ atom q\n"
			"variable        varID atom id\n"
		   f"group           scaf type == 1\n"
		   f"group           real id <= {p.nbead}\n"
			"group           mobile type 1 2\n"
		   f"thermo          {int(p.dump_every*10)}\n\n")

		### relax everything
		if p.nstep_relax > 0: f.write(
			"## Relaxation\n"
		   f"fix             tstat1 mobile langevin {p.T_relax} {p.T_relax} {1/p.gamma_t:0.4f} {p.rseed}\n"
			"fix             tstat2 mobile nve/limit 0.1\n"
		   f"timestep        {p.dt}\n"
		   f"run             {int(p.nstep_relax)}\n"
		   f"fix             tstat1 mobile langevin {p.T_relax} {p.T} {1/p.gamma_t:0.4f} {p.rseed}\n"
		   f"run             {int(p.nstep_relax)}\n"
			"unfix           tstat1\n"
			"unfix           tstat2\n"
			"reset_timestep  0\n\n")

		#-------- molecule templates --------#

		### molecule template header
		f.write(
			"## Molecules\n")

		### hyb bond templates
		for ri in range(nreact_bondHyb): f.write(
		   f"molecule        hybBond{ri+1}_mol_bondNo react/hybBond{ri+1}_mol_bondNo.txt\n"
		   f"molecule        hybBond{ri+1}_mol_bondYa react/hybBond{ri+1}_mol_bondYa.txt\n")

		### misbinding templates
		for ri in range(nreact_bondMis): f.write(
		   f"molecule        misBond{ri+1:0>{len_nrbM}}_mol react/misBond{ri+1:0>{len_nrbM}}_mol.txt\n")

		### angle activation templates
		for ri in range(nreact_angleAct): f.write(
		   f"molecule        angleAct{ri+1:0>{len_nraH}}_mol react/angleAct{ri+1:0>{len_nraH}}_mol.txt\n")

		### angle deactivation templates
		for ri in range(nreact_angleDeact): f.write(
		   f"molecule        angleDeact{ri+1}_mol react/angleDeact{ri+1}_mol.txt\n")

		### bridge templates
		for ri in range(nreact_bridge): f.write(
		   f"molecule        bridge_mol_bondNo react/bridge_mol_bondNo.txt\n"
		   f"molecule        bridge_mol_bondYa react/bridge_mol_bondYa.txt\n")

		### bridge breaking templates
		for ri in range(nreact_unbridge): f.write(
		   f"molecule        unbridge{ri+1}_mol_bondYa react/unbridge{ri+1}_mol_bondYa.txt\n"
		   f"molecule        unbridge{ri+1}_mol_bondNo react/unbridge{ri+1}_mol_bondNo.txt\n")
		f.write("\n")

		#-------- reactions --------#

		### reaction header
		f.write(
			"## Reactions\n"
			"fix             reactions all bond/react reset_mol_ids no")

		### bond hybridization reactions
		for ri in range(nreact_bondHyb): f.write(
		   f" &\n                react bondHyb{ri+1} all {int(react_every_bondHyb)} 0.0 {r12_cut_react_hybBond:.1f} hybBond{ri+1}_mol_bondNo hybBond{ri+1}_mol_bondYa react/bondHyb{ri+1}_map.txt custom_charges 2")
		
		### bond dehybridization reactions
		for ri in range(nreact_bondDehyb): f.write(
		   f" &\n                react bondDehyb{ri+1} all {int(react_every_bondDehyb)} {r12_cut_react_hybBond:.1f} {comm_cut} hybBond{ri+1}_mol_bondYa hybBond{ri+1}_mol_bondNo react/bondDehyb{ri+1}_map.txt custom_charges 2")
		
		### misbinding reactions
		for ri in range(nreact_bondMis): f.write(
		   f" &\n                react bondMis{ri+1:0>{len_nrbM}} all {int(react_every_bondMis)} 0.0 {r12_cut_react_hybBond:.1f} misBond{ri+1:0>{len_nrbM}}_mol misBond{ri+1:0>{len_nrbM}}_mol react/bondMis{ri+1:0>{len_nrbM}}_map.txt custom_charges 1")

		### angle activation reactions
		for ri in range(nreact_angleAct): f.write(
		   f" &\n                react angleAct{ri+1:0>{len_nraH}} all {int(react_every_angleAct)} 0.0 {p.r12_cut_hyb:.1f} angleAct{ri+1:0>{len_nraH}}_mol angleAct{ri+1:0>{len_nraH}}_mol react/angleAct{ri+1:0>{len_nraH}}_map.txt custom_charges 4")

		### angle deactivation reactions
		for ri in range(nreact_angleDeact): f.write(
		   f" &\n                react angleDeact{ri+1} all {int(react_every_angleDeact)} {p.r12_cut_hyb:.1f} {comm_cut} angleDeact{ri+1}_mol angleDeact{ri+1}_mol react/angleDeact{ri+1}_map.txt custom_charges 1")

		### bridge making reaction
		for ri in range(nreact_bridge): f.write(
		   f" &\n                react bridge all {int(react_every_bondHyb)} 0.0 {r12_cut_react_bridge:.2f} bridge_mol_bondNo bridge_mol_bondYa react/bridge_map.txt custom_charges 3")

		### bridge breaking reactions
		for ri in range(nreact_unbridge): f.write(
		   f" &\n                react unbridge{ri+1} all {int(react_every_bondDehyb)} {p.r12_cut_hyb:.1f} {comm_cut} unbridge{ri+1}_mol_bondYa unbridge{ri+1}_mol_bondNo react/unbridge{ri+1}_map.txt custom_charges 1")
		f.write("\n\n")

		#-------- end reactions --------#
	
		### binding updates
		f.write(
			"## Updates\n"
			"compute         bond_energies all bond/local engpot\n"
			"compute         angle_energies all angle/local eng\n"
		   f"fix             bond_hist all ave/histo {int(p.dump_every)} 1 {int(p.dump_every)} {-int(p.U_hyb)} -1E-6 1 c_bond_energies mode vector\n"
		   f"fix             angle_hist all ave/histo {int(p.dump_every)} 1 {int(p.dump_every)} 1E-6 1E6 1 c_angle_energies mode vector\n"
			"variable        hyb_count equal f_bond_hist[1]\n"
			"variable        angle_count equal f_angle_hist[1]\n"
			"variable        var_step equal step\n"
		   f"fix             print_hyb all print {int(p.dump_every)} &\n"
			"                \"Timestep ${var_step}  |  ${hyb_count}/"
						   f"{nhyb}" + " hybridizations  |  ${angle_count}/"
						   f"{nangle}" + " angles activated\"\n\n")

		### debug output
		if p.debug:
			f.write(
			"## Debugging\n"
			"compute         compDB1 all bond/local dist engpot\n"
			"compute         compDB2 all property/local btype batom1 batom2\n"
		   f"dump            dumpDB all local {int(p.dump_every)} dump_bonds.dat index c_compDB1[1] c_compDB1[2] c_compDB2[1] c_compDB2[2] c_compDB2[3] \n"
			"dump_modify     dumpDB append yes\n"
			"compute         compDA1 all angle/local theta eng\n"
			"compute         compDA2 all property/local atype aatom1 aatom2 aatom3\n"
		   f"dump            dumpDA all local {int(p.dump_every)} dump_angles.dat index c_compDA1[1] c_compDA1[2] c_compDA2[1] c_compDA2[2] c_compDA2[3] c_compDA2[4]\n"
			"dump_modify     dumpDA append yes\n"
		   f"dump            dumpDC all custom {int(p.dump_every)} dump_charges.dat id q\n"
			"dump_modify     dumpDC sort id append yes\n\n")

		### production
		f.write(
			"## Production\n"
		   f"fix             tstat1 mobile langevin {p.T} {p.T} {1/p.gamma_t:0.4f} {p.rseed}\n"
			"fix             tstat2 mobile nve\n"
		   f"timestep        {p.dt}\n"
		   f"dump            dumpT real custom {int(p.dump_every)} trajectory.dat id mol xs ys zs\n"
			"dump_modify     dumpT sort id append yes\n"
		   f"restart         {int(p.dump_every/2)} restart_binary1.out restart_binary2.out\n\n")

		### run the simulation
		f.write(
			"## Go Time\n"
		   f"run             {int(p.nstep)}\n"
			"write_data      restart_geometry.out\n\n")


### write reaction files for bond hybridization
def writeReactHybBond(outReactFold, mis_d2_cuts, p):
	print("Writing bond hyb/dehyb react files...")

	### no reactions necessary for forced binding with no reserved staples
	if p.forceBind and not p.reserveStap:
		return 0

	### template description
	# central scaffold, complimentary central staple, flanking scaffolds
	# initiated by central scaffold and staple (0 and 1)
	# one template always used, second template for linear scaffolds

	### fragment description
	# 1 - central scaffold, used to: (for hyb) ensure full or parital complimentarity, (for dehyb) ensure angle deactivation
	# 2 - central staple, used to: (for hyb) ensure full or partial complimentarity, (for hyb) ensure unbound hyb status, set hyb status with custom charges
	# 3 - flanking staple 1, used to: (for dehyb) ensure angle deactivation
	# 4 - flanking staple 2 (if present), used to: (for dehyb) ensure angle deactivation

	### initialize templates
	atoms_all = []
	bonds_all = []
	edges_all = []
	frags_all = []

	### two flanking scaffold beads
	atoms = [ [0,-1,0], [1,0,1], [0,-1,2], [0,-1,3] ]
	bonds = [ [0,2,0], [0,0,3] ]
	edges = [ 1, 2, 3 ]
	frags = [ [0], [1], [2], [3] ]
	atoms_all.append(atoms)
	bonds_all.append(bonds)
	edges_all.append(edges)
	frags_all.append(frags)

	### one flanking scaffold bead
	if not p.circularScaf:
		atoms = [ [0,-1,0], [1,0,1], [0,-1,2] ]
		bonds = [ [0,2,0] ]
		edges = [ 1, 2 ]
		frags = [ [0], [1], [2] ]
		atoms_all.append(atoms)
		bonds_all.append(bonds)
		edges_all.append(edges)
		frags_all.append(frags)

	### loop over reactions
	nreact = len(atoms_all)
	for ri in range(nreact):

		### make copies
		atoms = copy.deepcopy(atoms_all[ri])
		bonds = copy.deepcopy(bonds_all[ri])
		edges = copy.deepcopy(edges_all[ri])
		frags = copy.deepcopy(frags_all[ri])

		### pre-reaction template
		molFile = f"{outReactFold}hybBond{ri+1}_mol_bondNo.txt"
		writeMolecule(molFile, atoms, bonds, None, frags)

		### reaction adjustments (add hyb bond)
		bonds.append([1,0,1])
		atoms[1][1] = 1

		### post-reaction template
		molFile = f"{outReactFold}hybBond{ri+1}_mol_bondYa.txt"
		writeMolecule(molFile, atoms, bonds, None, frags)

		### count atoms and edges
		natom = len(atoms)
		nedge = len(edges)

		#-------- hybridization reaction map --------#

		### full hybridization map
		mapFile = f"{outReactFold}bondHyb{ri+1}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"2 constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write(f"1\n")
			f.write(f"2\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges[edgei]+1}\n")

			f.write("\nConstraints\n\n")
			f.write(f"custom \"round(rxnsum(v_varQ,2)) == 1\"\n")							# ensure unbound hyb status
			if p.nmisBond == 0:
				f.write(f"custom \"rxnsum(v_varCF1,1) == rxnsum(v_varCF1,2)\"\n")			# ensure full complimintarity (for no misbinding)
			else:
				f.write(f"custom \"")
				for i in range(p.ncompFactor):
					if i != 0:
						f.write(" + ")
					f.write(f"(rxnsum(v_varCF{i+1},1) - rxnsum(v_varCF{i+1},2))^2")
				f.write(f" <= {mis_d2_cuts[-1]:0.4f}\"\n")									# ensure full or partial complementarity

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

		#-------- dehybridization reaction map --------#

		### only if dehybridization
		if p.dehyb:

			### count constraints
			ncons = sum(1 for x in atoms if x[0] == 0)

			mapFile = f"{outReactFold}bondDehyb{ri+1}_map.txt"
			with open(mapFile, 'w') as f:

				f.write("## Hybridization\n")
				f.write(f"{natom} equivalences\n")
				f.write(f"{nedge} edgeIDs\n")
				f.write(f"{ncons} constraints\n")

				f.write(f"\nInitiatorIDs\n\n")
				f.write("1\n")
				f.write("2\n")

				f.write(f"\nEdgeIDs\n\n")
				for edgei in range(nedge):
					f.write(f"{edges[edgei]+1}\n")

				f.write("\nConstraints\n\n")
				for atomi in range(natom):
					if atoms[atomi][0] == 0:
						f.write(f"custom \"rxnsum(v_varQ,{atomi+1}) == 1 || rxnsum(v_varQ,{atomi+1}) == 2\"\n")		# ensure angle deactivation

				f.write("\nEquivalences\n\n")
				for atomi in range(natom):
					f.write(f"{atomi+1}\t{atomi+1}\n")

		#-------- end maps --------#

	### return reaction count
	return nreact


### write reaction files for hybridization bonds
def writeReactMisBond(outReactFold, mis_d2_cuts, p):
	print("Writing misbinding react files...")

	### no reactions necessary if not misbinding
	if p.nmisBond == 0:
		return 0

	### template description
	# central staple, complimentary central scaffold, flanking staples
	# initiated by central staple and scaffold (0 and 1)
	# three templates always used

	### fragment description
	# 1 - central staple, used to: ensure partial complimentarity, ensure fully bound hyb status, set hyb status with custom charges
	# 2 - central scaffold, use to: ensure partial complimentarity

	### initialize templates
	atoms_all = []
	bonds_all = []
	edges_all = []

	### two flanking staple beads
	atoms = [ [1,1,0], [0,-1,1], [1,-1,2], [1,-1,3] ]
	bonds = [ [1,0,1], [0,2,0], [0,0,3] ]
	edges = [ 1, 2, 3 ]
	atoms_all.append(atoms)
	bonds_all.append(bonds)
	edges_all.append(edges)

	### one flanking staple bead
	atoms = [ [1,1,0], [0,-1,1], [1,-1,2] ]
	bonds = [ [1,0,1], [0,2,0] ]
	edges = [ 1, 2 ]
	atoms_all.append(atoms)
	bonds_all.append(bonds)
	edges_all.append(edges)

	### no flanking staple bead
	atoms = [ [1,1,0], [0,-1,1] ]
	bonds = [ [1,0,1] ]
	edges = [ 1 ]
	atoms_all.append(atoms)
	bonds_all.append(bonds)
	edges_all.append(edges)

	### define fragments
	frags = [ [0], [1] ]

	### count reactions
	ntemplate = len(atoms_all)
	nreact = ntemplate*p.nmisBond
	len_nreact = len(str(nreact))

	### loop over reactions
	react_count = 0
	for ti in range(ntemplate):
		for mi in range(p.nmisBond):

			### get reaction index
			react_count += 1

			### make copies
			atoms = copy.deepcopy(atoms_all[ti])
			bonds = copy.deepcopy(bonds_all[ti])
			edges = copy.deepcopy(edges_all[ti])

			### reaction adjustments (turn full hyb bond into partial hyb bond)
			bonds[0][0] = mi+2
			atoms[0][1] = mi+2

			### template (pre and post reaction)
			molFile = f"{outReactFold}misBond{react_count:0>{len_nreact}}_mol.txt"
			writeMolecule(molFile, atoms, bonds, None, frags)

			### count atoms and edges
			natom = len(atoms)
			nedge = len(edges)

			#-------- misbinding reaction map --------#

			mapFile = f"{outReactFold}bondMis{react_count:0>{len_nreact}}_map.txt"
			with open(mapFile, 'w') as f:

				f.write("## Hybridization\n")
				f.write(f"{natom} equivalences\n")
				f.write(f"{nedge} edgeIDs\n")
				f.write(f"3 constraints\n")

				f.write(f"\nInitiatorIDs\n\n")
				f.write("1\n")
				f.write("2\n")

				f.write(f"\nEdgeIDs\n\n")
				for edgei in range(nedge):
					f.write(f"{edges[edgei]+1}\n")

				f.write("\nConstraints\n\n")
				f.write(f"custom \"round(rxnsum(v_varQ,1)) == 2\"\n")					# ensure fully bound hyb status
				f.write(f"custom \"")
				for i in range(p.ncompFactor):
					if i != 0:
						f.write(" + ")
					f.write(f"(rxnsum(v_varCF{i+1},1) - rxnsum(v_varCF{i+1},2))^2")
				f.write(f" > {mis_d2_cuts[mi]:0.4f}\"\n")								# ensure correct partial complimentarity
				f.write(f"custom \"")
				for i in range(p.ncompFactor):
					if i != 0:
						f.write(" + ")
					f.write(f"(rxnsum(v_varCF{i+1},1) - rxnsum(v_varCF{i+1},2))^2")
				f.write(f" <= {mis_d2_cuts[mi+1]:0.4f}\"\n")							# ensure correct partial complimentarity

				f.write("\nEquivalences\n\n")
				for atomi in range(natom):
					f.write(f"{atomi+1}\t{atomi+1}\n")

			#-------- end map --------#

	### return reaction count
	return nreact


### write reaction files for angle activation
def writeReactAngleAct(outReactFold, backbone_neighbors, complements, is_crossover, p):
	print("Writing angle activation react files...")
	debug_angle = False

	### template desciprion
	# 3 core scaffolds, 3 core staples, flanking scaffolds, connected staples
	# initiated by central scaffold and central staple (2 and 5)
	# up to 124 templates, usually no more than 80 or so
	# to catch symmetry, staple bonds always list the core staple first

	### fragment description
	# 1 - left core scaffold, used to: ensure angle type and status
	# 2 - central scaffold, used to: ensure angle type and status
	# 3 - right core scaffold, used to: ensure angle type and status
	# 4 - core scaffolds and staples, used to: set angle status and hyb status with custom charges

	### initialize templates
	atoms_all = []
	bonds_all = []
	angls_all = []
	edges_all = []

	#-------- template loop --------#

	### loop over all beads
	for bi in range(p.n_scaf):

		### get neighbors to central bead
		bi_5p,bi_3p = getAssembledNeighbors(bi, backbone_neighbors, complements, p)

		### skip if core scaffold is not present
		if bi_5p == -1 or bi_3p == -1:
			continue

		### skip if core scaffold is not fully complimentary
		if complements[bi_5p] == -1 or complements[bi] == -1 or complements[bi_3p] == -1:
			continue

		### loop over configurational options (bridged ends, 5p staple copied, 3p staple copied)
		for options in itertools.product([True, False], repeat=3):

			#-------- working 5' to 3' --------#

			### core atoms
			a = bi_5p
			b = bi
			c = bi_3p
			aC = complements[a]
			bC = complements[b]
			cC = complements[c]

			### core topology
			atoms_5to3 = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,1,cC], [1,1,bC], [1,1,aC] ]
			bonds_5to3 = [ [0,a,b], [0,b,c], [1,a,aC], [1,b,bC], [1,c,cC] ]
			angls_5to3 = [ [int(is_crossover[b]),a,b,c,1] ]
			edges_5to3 = [ cC, aC ]

			### add scaffold 5' side topology
			a_5p = getAssembledNeighbors(a, backbone_neighbors, complements, p)[0]
			if a_5p != -1:
				if backbone_neighbors[a][0] != -1 or options[0]:
					atoms_5to3.append([0,-1,a_5p])
					bonds_5to3.append([0,a_5p,a])
					angls_5to3.append([int(is_crossover[a]),a_5p,a,b,0])
					edges_5to3.append(a_5p)

			### add scaffold 3' side topology
			c_3p = getAssembledNeighbors(c, backbone_neighbors, complements, p)[1]
			if c_3p != -1:
				if backbone_neighbors[c][1] != -1 or options[0]:
					atoms_5to3.append([0,-1,c_3p])
					bonds_5to3.append([0,c,c_3p])
					angls_5to3.append([int(is_crossover[c]),b,c,c_3p,2])
					edges_5to3.append(c_3p)

			### initialize symmetric bonds
			bonds_sym = []

			### add central staple 5' side topology
			bC_5p = backbone_neighbors[bC][0]
			if bC_5p != -1:
				if [1,1,bC_5p] not in atoms_5to3:
					atoms_5to3.append([1,-1,bC_5p])
					edges_5to3.append(bC_5p)
					bonds_sym.append([0,bC,bC_5p])
				else:
					if p.stap_copies > 1 and options[1]:
						atoms_5to3.append([1,-1,bC_5p+p.n_stap])
						edges_5to3.append(bC_5p+p.n_stap)
						bonds_sym.append([0,bC,bC_5p+p.n_stap])
					else:
						bonds_5to3.append([0,bC,bC_5p])

			### add central staple 3' side topology
			bC_3p = backbone_neighbors[bC][1]
			if bC_3p != -1:
				if [1,1,bC_3p] not in atoms_5to3:
					atoms_5to3.append([1,-1,bC_3p])
					edges_5to3.append(bC_3p)
					bonds_sym.append([0,bC,bC_3p])
				else:
					if p.stap_copies > 1 and options[2]:
						atoms_5to3.append([1,-1,bC_3p+p.n_stap])
						edges_5to3.append(bC_3p+p.n_stap)
						bonds_sym.append([0,bC,bC_3p+p.n_stap])
					else:
						bonds_5to3.append([0,bC,bC_3p])

			### add symmetric bonds
			bonds_5to3 = bonds_5to3 + bonds_sym

			#-------- working 3' to 5' --------#

			### intialize atoms
			a = bi_3p
			b = bi
			c = bi_5p
			aC = complements[a]
			bC = complements[b]
			cC = complements[c]

			### core topology
			atoms_3to5 = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,1,cC], [1,1,bC], [1,1,aC] ]
			bonds_3to5 = [ [0,a,b], [0,b,c], [1,a,aC], [1,b,bC], [1,c,cC] ]
			angls_3to5 = [ [int(is_crossover[b]),a,b,c,1] ]
			edges_3to5 = [ cC, aC ]

			### add scaffold 3' side topology
			a_3p = getAssembledNeighbors(a, backbone_neighbors, complements, p)[1]
			if a_3p != -1:
				if backbone_neighbors[a][1] != -1 or options[0]:
					atoms_3to5.append([0,-1,a_3p])
					bonds_3to5.append([0,a_3p,a])
					angls_3to5.append([int(is_crossover[a]),a_3p,a,b,0])
					edges_3to5.append(a_3p)

			### add scaffold 5' side topology
			c_5p = getAssembledNeighbors(c, backbone_neighbors, complements, p)[0]
			if c_5p != -1:
				if backbone_neighbors[c][0] != -1 or options[0]:
					atoms_3to5.append([0,-1,c_5p])
					bonds_3to5.append([0,c,c_5p])
					angls_3to5.append([int(is_crossover[c]),b,c,c_5p,2])
					edges_3to5.append(c_5p)

			### initialize symmetric bonds
			bonds_sym = []

			### add central staple 3' side topology
			bC_3p = backbone_neighbors[bC][1]
			if bC_3p != -1:
				if [1,1,bC_3p] not in atoms_3to5:
					atoms_3to5.append([1,-1,bC_3p])
					edges_3to5.append(bC_3p)
					bonds_sym.append([0,bC,bC_3p])
				else:
					if p.stap_copies > 1 and options[2]:
						atoms_3to5.append([1,-1,bC_3p+p.n_stap])
						edges_3to5.append(bC_3p+p.n_stap)
						bonds_sym.append([0,bC,bC_3p+p.n_stap])
					else:
						bonds_3to5.append([0,bC,bC_3p])

			### add central staple 5' side topology
			bC_5p = backbone_neighbors[bC][0]
			if bC_5p != -1:
				if [1,1,bC_5p] not in atoms_3to5:
					atoms_3to5.append([1,-1,bC_5p])
					edges_3to5.append(bC_5p)
					bonds_sym.append([0,bC,bC_5p])
				else:
					if p.stap_copies > 1 and options[1]:
						atoms_3to5.append([1,-1,bC_5p+p.n_stap])
						edges_3to5.append(bC_5p+p.n_stap)
						bonds_sym.append([0,bC,bC_5p+p.n_stap])
					else:
						bonds_3to5.append([0,bC,bC_5p])

			### add symmetric bonds
			bonds_3to5 = bonds_3to5 + bonds_sym

			#-------- add templates to list --------#

			### renumber
			atoms_5to3,bonds_5to3,angls_5to3,edges_5to3 = renumberAtoms_angleTemplate(atoms_5to3,bonds_5to3,angls_5to3,edges_5to3)
			atoms_3to5,bonds_3to5,angls_3to5,edges_3to5 = renumberAtoms_angleTemplate(atoms_3to5,bonds_3to5,angls_3to5,edges_3to5)

			### test for symmetry
			templates = [[a,b,c,d] for a,b,c,d in zip([atoms_5to3,atoms_3to5],[bonds_5to3,bonds_3to5],[angls_5to3,angls_3to5],[edges_5to3,edges_3to5])]
			templates = removeDuplicateElements(templates)
			if len(templates) == 1:
				symmetric = True
			else:
				symmetric = False

			### add basic template to list
			atoms_all.append(atoms_5to3)
			bonds_all.append(bonds_5to3)
			angls_all.append(angls_5to3)
			edges_all.append(edges_5to3)
			atoms_all.append(atoms_3to5)
			bonds_all.append(bonds_3to5)
			angls_all.append(angls_3to5)
			edges_all.append(edges_3to5)

			### remove duplicates
			templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angls_all,edges_all)]
			templates = removeDuplicateElements(templates)
			if not symmetric:
				templates.pop()
			atoms_all,bonds_all,angls_all,edges_all = unzip4(templates)

			### for templates with at least two angles
			if len(angls_5to3) >= 2:

				### turn second angle on
				atoms_5to3_copy = copy.deepcopy(atoms_5to3)
				atoms_5to3_copy[angls_5to3[1][4]][1] += 2
				atoms_all.append(atoms_5to3_copy)
				bonds_all.append(bonds_5to3)
				angls_all.append(angls_5to3)
				edges_all.append(edges_5to3)
				atoms_3to5_copy = copy.deepcopy(atoms_3to5)
				atoms_3to5_copy[angls_3to5[1][4]][1] += 2
				atoms_all.append(atoms_3to5_copy)
				bonds_all.append(bonds_3to5)
				angls_all.append(angls_3to5)
				edges_all.append(edges_3to5)

				### remove duplicates
				templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angls_all,edges_all)]
				templates = removeDuplicateElements(templates)
				if not symmetric:
					templates.pop()
				atoms_all,bonds_all,angls_all,edges_all = unzip4(templates)

			### for templates with three angles
			if len(angls_5to3) >= 3:

				### turn just third angle on
				if not symmetric:
					atoms_5to3_copy = copy.deepcopy(atoms_5to3)
					atoms_5to3_copy[angls_5to3[2][4]][1] += 2
					atoms_all.append(atoms_5to3_copy)
					bonds_all.append(bonds_5to3)
					angls_all.append(angls_5to3)
					edges_all.append(edges_5to3)
					atoms_3to5_copy = copy.deepcopy(atoms_3to5)
					atoms_3to5_copy[angls_3to5[2][4]][1] += 2
					atoms_all.append(atoms_3to5_copy)
					bonds_all.append(bonds_3to5)
					angls_all.append(angls_3to5)
					edges_all.append(edges_3to5)

					### remove duplicates
					templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angls_all,edges_all)]
					templates = removeDuplicateElements(templates)
					templates.pop()
					atoms_all,bonds_all,angls_all,edges_all = unzip4(templates)

				### turn both angles on
				atoms_5to3_copy = copy.deepcopy(atoms_5to3)
				atoms_5to3_copy[angls_5to3[1][4]][1] += 2
				atoms_5to3_copy[angls_5to3[2][4]][1] += 2
				atoms_all.append(atoms_5to3_copy)
				bonds_all.append(bonds_5to3)
				angls_all.append(angls_5to3)
				edges_all.append(edges_5to3)
				atoms_3to5_copy = copy.deepcopy(atoms_3to5)
				atoms_3to5_copy[angls_3to5[1][4]][1] += 2
				atoms_3to5_copy[angls_3to5[2][4]][1] += 2
				atoms_all.append(atoms_3to5_copy)
				bonds_all.append(bonds_3to5)
				angls_all.append(angls_3to5)
				edges_all.append(edges_3to5)

				### remove duplicates
				templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angls_all,edges_all)]
				templates = removeDuplicateElements(templates)
				if not symmetric:
					templates.pop()
				atoms_all,bonds_all,angls_all,edges_all = unzip4(templates)

	#-------- end template loop --------#

	### for nice debug output
	if debug_angle: print()

	### define fragments
	frags = [ [0], [1], [2], [0,1,2,3,4,5] ]

	### loop over templates
	nreact = len(atoms_all)
	len_nreact = len(str(nreact))
	for ri in range(nreact):

		### make copies
		atoms = copy.deepcopy(atoms_all[ri])
		bonds = copy.deepcopy(bonds_all[ri])
		angls = copy.deepcopy(angls_all[ri])
		edges = copy.deepcopy(edges_all[ri])

		### debug output
		if debug_angle:
			print(f"Angle template {ri+1}:")
			print(atoms)
			print(bonds)
			print(angls)
			print(edges)
			print()

		### reaction adjustments (turn on central angle)
		atoms_write = copy.deepcopy(atoms)
		atoms_write[1][1] += 2

		### extract angles that are on
		angls_write = []
		for angli in range(len(angls)):
			if atoms_write[angls[angli][4]][1] > 1:
				angls_write.append(copy.deepcopy(angls[angli]))

		### template (pre and post reaction)
		molFile = f"{outReactFold}angleAct{ri+1:0>{len_nreact}}_mol.txt"
		writeMolecule(molFile, atoms_write, bonds, angls_write, frags)

		### count atoms and edges
		natom = len(atoms)
		nedge = len(edges)

		#-------- angle activation reaction map --------#

		### reaction map
		mapFile = f"{outReactFold}angleAct{ri+1:0>{len_nreact}}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"5 constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write("2\n")
			f.write("5\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges[edgei]+1}\n")
			
			f.write("\nConstraints\n\n")
			f.write(f"custom \"round(rxnsum(v_varQ,1)) == {atoms[0][1]+1}\"\n")		# ensure left angle type and status
			f.write(f"custom \"round(rxnsum(v_varQ,2)) == {atoms[1][1]+1}\"\n")		# ensure central angle type and status
			f.write(f"custom \"round(rxnsum(v_varQ,3)) == {atoms[2][1]+1}\"\n")		# ensure right angle type and status
			f.write("distance 1 6 0.0 2.0\n")										# ensure hyb status of left complimemts
			f.write("distance 3 4 0.0 2.0\n")										# ensure hyb status of right complimemts

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

		#-------- end map --------#

	### return reaction count
	return nreact


### write reaction files for angle deactivation
def writeReactAngleDeact(outReactFold, backbone_neighbors, complements, is_crossover, p):
	print("Writing angle deactivation react files...")

	### only necessary if including dehybridization
	if not p.dehyb:
		return 0

	### determine if there are end bridging reactions
	bridgeEndsReact = True
	if p.circularScaf or getAssembledNeighbors(0, backbone_neighbors, complements, p)[0] == -1:
		bridgeEndsReact = False

	### template description
	# central scaffold, central staple, flanking scaffolds
	# initiated by central scaffold and staple
	# up to 5 circular scaffold templates, up to 2 linear scaffold templates (max 7 in total)

	### fragment description
	# 1 - all beads, used to: set angle and hyb status with custom charges
	# 2 - central scaffold, used to: ensure angle type, ensure end evasion if bridging
	# 3 - flanking scaffold, used to: ensure angle type
	# 4 - flanking scaffold (if present), used to: ensure angle type

	### initialize circular scaffold templates
	atoms_all = []
	bonds_all = []
	edges_all = []

	### core topology (all 180)
	atoms = [ [0,0,0], [1,1,1], [0,0,2], [0,0,3] ]
	bonds = [ [1,0,1], [0,2,0], [0,0,3] ]
	edges = [ 1, 2, 3 ]

	### all 180
	if checkAngleDeactTemplateMatch(True, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 1")
		atoms_all.append(copy.deepcopy(atoms))
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))

	### side 90
	if checkAngleDeactTemplateMatch(False, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 2")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))

	### side, middle 90
	if checkAngleDeactTemplateMatch(False, False, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 3")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))

	### side, side 90
	if checkAngleDeactTemplateMatch(False, True, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 4")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[2][1] = 1
		atoms_copy[3][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))

	### all 90
	if checkAngleDeactTemplateMatch(False, False, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 5")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[2][1] = 1
		atoms_copy[3][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))

	### for nice debug output
	if p.debug: print()

	### define fragments
	frags = [ [0,1,2,3], [0], [2], [3] ]

	### loop over reactions
	nreact_circular = len(atoms_all)
	nreact = nreact_circular
	for ri in range(nreact_circular):

		### make copies
		atoms = copy.deepcopy(atoms_all[ri])
		bonds = copy.deepcopy(bonds_all[ri])
		edges = copy.deepcopy(edges_all[ri])

		### template (pre and post reaction)
		molFile = f"{outReactFold}angleDeact{ri+1}_mol.txt"
		writeMolecule(molFile, atoms, bonds, None, frags)

		### count atoms and edges
		natom = len(atoms)
		nedge = len(edges)

		### count constrains
		ncons = 5 if bridgeEndsReact else 3

		#-------- angle deactivation (circular scaffold) reaction map --------#

		mapFile = f"{outReactFold}angleDeact{ri+1}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"{ncons} constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write(f"1\n")
			f.write(f"2\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges[edgei]+1}\n")

			f.write("\nConstraints\n\n")
			f.write(f"custom \"round(rxnsum(v_varQ,2)) == {atoms[0][1]+1} || "
							 f"round(rxnsum(v_varQ,2)) == {atoms[0][1]+3}\"\n")		# ensure central angle type
			f.write(f"custom \"round(rxnsum(v_varQ,3)) == {atoms[2][1]+1} || "
							 f"round(rxnsum(v_varQ,3)) == {atoms[2][1]+3}\"\n")		# ensure left angle type
			f.write(f"custom \"round(rxnsum(v_varQ,4)) == {atoms[3][1]+1} || "
							 f"round(rxnsum(v_varQ,4)) == {atoms[3][1]+3}\"\n")		# ensure right angle type

			### avoid ends when appropriate
			if bridgeEndsReact:
				f.write(f"custom \"round(rxnsum(v_varID,2)) != {p.n_scaf}\"\n")
				f.write(f"custom \"round(rxnsum(v_varID,2)) != 1\"\n")
		
			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

		#-------- end map --------#

	### linear scaffold templates
	if not p.circularScaf:

		### initialize linear scaffold templates
		atoms_all = []
		bonds_all = []
		edges_all = []

		### center as 5' end
		if complements[0] != -1:

			atoms = [ [0,int(is_crossover[0]),0], [1,1,1], [0,int(is_crossover[1]),2] ]
			bonds = [ [1,0,1], [0,2,0] ]
			edges = [ 1, 2 ]
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))

		### center as 3' end
		if complements[p.n_scaf-1] != -1:

			atoms = [ [0,int(is_crossover[p.n_scaf-1]),0], [1,1,1], [0,int(is_crossover[p.n_scaf-2]),2] ]
			bonds = [ [1,0,1], [0,2,0] ]
			edges = [ 1, 2 ]
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))

		### remove duplicates (if any)
		templates = [[a,b,c] for a,b,c in zip(atoms_all,bonds_all,edges_all)]
		templates = removeDuplicateElements(templates)
		atoms_all,bonds_all,edges_all = unzip3(templates)

		### define fragments
		frags = [ [0,1,2], [0], [2] ]

		### loop over reactions
		nreact_linear = len(atoms_all)
		nreact += nreact_linear
		for ri in range(nreact_linear):

			### make copies
			atoms = copy.deepcopy(atoms_all[ri])
			bonds = copy.deepcopy(bonds_all[ri])
			edges = copy.deepcopy(edges_all[ri])

			### template (pre and post reaction)
			molFile = f"{outReactFold}angleDeact{nreact_circular+ri+1}_mol.txt"
			writeMolecule(molFile, atoms, bonds, None, frags)

			### count atoms and edges
			natom = len(atoms)
			nedge = len(edges)

			#-------- angle deactivation (linear scaffold) reaction map --------#

			mapFile = f"{outReactFold}angleDeact{nreact_circular+ri+1}_map.txt"
			with open(mapFile, 'w') as f:

				f.write("## Hybridization\n")
				f.write(f"{natom} equivalences\n")
				f.write(f"{nedge} edgeIDs\n")
				f.write(f"2 constraints\n")

				f.write(f"\nInitiatorIDs\n\n")
				f.write(f"1\n")
				f.write(f"2\n")

				f.write(f"\nEdgeIDs\n\n")
				for edgei in range(nedge):
					f.write(f"{edges[edgei]+1}\n")

				f.write("\nConstraints\n\n")
				f.write(f"custom \"round(rxnsum(v_varQ,2)) == {atoms[0][1]+3} || "
								 f"round(rxnsum(v_varQ,3)) == {atoms[2][1]+3}\"\n")		# ensure central angle type
				f.write(f"custom \"round(rxnsum(v_varQ,3)) == {atoms[2][1]+1} || "
								 f"round(rxnsum(v_varQ,3)) == {atoms[2][1]+3}\"\n")		# ensure left angle type
			
				f.write("\nEquivalences\n\n")
				for atomi in range(natom):
					f.write(f"{atomi+1}\t{atomi+1}\n")

			#-------- end map --------#

	### return reaction count
	return nreact


### write reaction files that connect and disconnect bridged scaffold ends
def writeReactBridge(outReactFold, backbone_neighbors, complements, is_crossover, p):
	print("Writing bridging react files...")

	### template description
	# central scaffold (5' end), central staple, flanking scaffold (3' end), flanking scaffold
	# initiated by central scaffold (5' end) and flanking scaffold (3' end)
	# only one template

	### fragment desciption
	# 1 - left central scaffold (3' end), used to: ensure 3' end identity
	# 2 - right central scaffold (5' end), used to: ensure 5' end identity
	# 3 - flanking scaffolds and staples, used to: set angle and hyb status with custom charges

	### determine if scaffold ends (if they exist) are bridged
	if p.circularScaf or getAssembledNeighbors(0, backbone_neighbors, complements, p)[0] == -1:
		return 0, 0

	### warning if no dehybridization
	if not p.dehyb:
		print("Flag: Scaffold ends are bridged, but dehybridization is is off, so end bridging is permenant.")

	### core beads
	a = p.n_scaf-1
	b = 0
	aC = complements[a]
	bC = complements[b]
	a_5p = p.n_scaf-2
	b_3p = 1

	### core topology
	atoms = [ [0,-1,a], [0,-1,b], [1,1,bC], [1,1,aC], [0,int(is_crossover[a_5p]),a_5p], [0,int(is_crossover[b_3p]),b_3p] ]
	bonds = [ [1,a,aC], [1,b,bC], [0,bC,aC], [0,a_5p,a], [0,b,b_3p] ]
	edges = [ bC, aC, a_5p, b_3p ]

	### renumber
	atoms,bonds,edges = renumberAtoms_bridgeTemplate(atoms,bonds,edges)

	### define fragments
	frags = [ [0], [1], [2,3,4,5] ]

	### pre reaction template
	molFile = f"{outReactFold}bridge_mol_bondNo.txt"
	writeMolecule(molFile,atoms,bonds,None,frags)

	### reaction adjustments (add bridging bond)
	bonds.append([0,0,1])

	### pre reaction template
	molFile = f"{outReactFold}bridge_mol_bondYa.txt"
	writeMolecule(molFile,atoms,bonds,None,frags)

	### count atoms and edges
	natom = len(atoms)
	nedge = len(edges)

	#-------- bridging reaction map --------#

	mapFile = f"{outReactFold}bridge_map.txt"
	with open(mapFile, 'w') as f:

		f.write("## Hybridization\n")
		f.write(f"{natom} equivalences\n")
		f.write(f"{nedge} edgeIDs\n")
		f.write(f"2 constraints\n")

		f.write(f"\nInitiatorIDs\n\n")
		f.write(f"1\n")
		f.write(f"2\n")

		f.write(f"\nEdgeIDs\n\n")
		for edgei in range(nedge):
			f.write(f"{edges[edgei]+1}\n")

		f.write("\nEquivalences\n\n")
		for atomi in range(natom):
			f.write(f"{atomi+1}\t{atomi+1}\n")

		f.write("\nConstraints\n\n")
		f.write(f"custom \"round(rxnsum(v_varID,1)) == {p.n_scaf}\"\n")		# ensure 3' end identity
		f.write(f"custom \"round(rxnsum(v_varID,2)) == 1\"\n")				# ensure 5' end identity

	#-------- end map --------#

	### no unbiridging if no dehybridization
	if not p.dehyb:
		return 1, 0

	### template description
	# central scaffold (end 1), central staple, flanking scaffold (end 2), flanking scaffold
	# initiated by central scaffold (end 1) and central staple
	# one or two templates

	### fragment desciption
	# 1 - all beads, used to: set angle and hyb status with custom charges
	# 2 - central scaffold (end 1), used to: ensure end identity, ensure angle type
	# 3 - flanking scaffold (end 2), used to: ensure end identity, ensure angle type
	# 4 - flanking scaffold, used to: ensure angle type

	### initialize unbridging templates
	atoms_all = []
	bonds_all = []
	edges_all = []

	### core beads (center as 5' end, working 5' to 3')
	a = 0
	aC = complements[a]
	a_5p = p.n_scaf-1
	a_3p = 1

	### core topology (center as 5' end)
	atoms = [ [0,int(is_crossover[a]),a], [1,1,aC], [0,int(is_crossover[a_5p]),a_5p], [0,int(is_crossover[a_3p]),a_3p] ]
	bonds = [ [1,a,aC], [0,a_5p,a], [0,a,a_3p] ]
	edges = [ aC, a_5p, a_3p ]

	### renumber, add to template list
	atoms,bonds,edges = renumberAtoms_bridgeTemplate(atoms,bonds,edges)
	atoms_all.append(copy.deepcopy(atoms))
	bonds_all.append(copy.deepcopy(bonds))
	edges_all.append(copy.deepcopy(edges))

	### core beads (center as 3' end, working 3' to 5')
	a = p.n_scaf-1
	aC = complements[a]
	a_3p = 0
	a_5p = p.n_scaf-2

	### core topology (center as 3' end)
	atoms = [ [0,int(is_crossover[a]),a], [1,1,aC], [0,int(is_crossover[a_3p]),a_3p], [0,int(is_crossover[a_5p]),a_5p] ]
	bonds = [ [1,a,aC], [0,a_3p,a], [0,a,a_5p] ]
	edges = [ aC, a_3p, a_5p ]

	### renumber, add to template list
	atoms,bonds,edges = renumberAtoms_bridgeTemplate(atoms,bonds,edges)
	atoms_all.append(copy.deepcopy(atoms))
	bonds_all.append(copy.deepcopy(bonds))
	edges_all.append(copy.deepcopy(edges))

	### remove possible duplicates
	templates = [[a,b,c] for a,b,c in zip(atoms_all,bonds_all,edges_all)]
	templates = removeDuplicateElements(templates)
	atoms_all,bonds_all,edges_all = unzip3(templates)

	### define fragments
	frags = [ [0,1,2,3], [0], [2], [3] ]

	### loop over reactions
	nreact_unbridge = len(atoms_all)
	for ri in range(nreact_unbridge):

		### make copies
		atoms = copy.deepcopy(atoms_all[ri])
		bonds = copy.deepcopy(bonds_all[ri])
		edges = copy.deepcopy(edges_all[ri])

		### pre reaction template
		molFile = f"{outReactFold}unbridge{ri+1}_mol_bondYa.txt"
		writeMolecule(molFile,atoms,bonds,None,frags)

		### reaction adjustments (remove bridging bond)
		bonds.pop(1)

		### post reaction template
		molFile = f"{outReactFold}unbridge{ri+1}_mol_bondNo.txt"
		writeMolecule(molFile,atoms,bonds,None,frags)

		### count atoms and edges
		natom = len(atoms)
		nedge = len(edges)

		#-------- unbridging reaction map --------#

		mapFile = f"{outReactFold}unbridge{ri+1}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"5 constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write(f"1\n")
			f.write(f"2\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges[edgei]+1}\n")

			f.write("\nConstraints\n\n")
			f.write(f"custom \"round(rxnsum(v_varID,2)) == 1 || "
							 f"round(rxnsum(v_varID,2)) == {p.n_scaf}\"\n")			# ensure end identity
			f.write(f"custom \"round(rxnsum(v_varID,3)) == 1 || "
							 f"round(rxnsum(v_varID,3)) == {p.n_scaf}\"\n")			# ensure end identity
			f.write(f"custom \"round(rxnsum(v_varQ,2)) == {atoms[0][1]+1} || "
							 f"round(rxnsum(v_varQ,2)) == {atoms[0][1]+3}\"\n")		# ensure end 1 angle type
			f.write(f"custom \"round(rxnsum(v_varQ,3)) == {atoms[2][1]+1} || "
							 f"round(rxnsum(v_varQ,3)) == {atoms[2][1]+3}\"\n")		# ensure end 2 angle type
			f.write(f"custom \"round(rxnsum(v_varQ,4)) == {atoms[3][1]+1} || "
							 f"round(rxnsum(v_varQ,4)) == {atoms[3][1]+3}\"\n")		# ensure flanking angle type
		
			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

		#-------- end map --------#

	### retrun reaction counts
	return 1, nreact_unbridge


### write molecule template file
def writeMolecule(molFile, atoms, bonds=None, angls=None, frags=None, includeCharge=True):
	natom = len(atoms)
	if bonds is not None and len(bonds) == 0: bonds = None
	if angls is not None and len(angls) == 0: angls = None
	if frags is not None and len(frags) == 0: frags = None

	with open(molFile, 'w') as f:
		f.write("## Molecule Template\n")
		f.write(f"{natom} atoms\n")
		if bonds is not None: f.write(f"{len(bonds)} bonds\n")	
		if angls is not None: f.write(f"{len(angls)} angles\n")			
		if frags is not None: f.write(f"{len(frags)} fragments\n")

		f.write("\nTypes\n\n")
		for atomi in range(natom):
			f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

		if bonds is not None:
			f.write("\nBonds\n\n")
			for bondi in range(len(bonds)):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

		if angls is not None:
			f.write("\nAngles\n\n")
			for angli in range(len(angls)):
				f.write(f"{angli+1}\t{angls[angli][0]+1}\t{angls[angli][1]+1}\t{angls[angli][2]+1}\t{angls[angli][3]+1}\n")

		if includeCharge:
			f.write("\nCharges\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

		if frags is not None:
			f.write("\nFragments\n\n")
			for fragi in range(len(frags)):
				f.write(f"{fragi+1}\t")
				for atomi in range(len(frags[fragi])):
					if atomi > 0:
						f.write(" ")
					f.write(f"{frags[fragi][atomi]+1}")
				f.write("\n")


### write table for hybridization bond
def writeHybBond(outSimFold, bondName, U_hyb, F_forceBind, comm_cut, U_barrier_comm, bond_res, p):
	bondFile = outSimFold + "bond_" + bondName + ".txt"
	npoint = int(comm_cut/bond_res+1)

	### write file
	with open(bondFile, 'w') as f:
		f.write(f"{bondName}\n")
		f.write(f"N {npoint}\n\n")
		f.write("# r E(r) F(r)\n")
		
		### loop over points
		for i in range(npoint):
			r12 = i * bond_res
			if r12 < p.r12_cut_hyb:
				U = U_hyb*(r12/p.r12_cut_hyb-1)
				F = -U_hyb/p.r12_cut_hyb
			elif p.forceBind:
				U = 6.96*F_forceBind*(r12-p.r12_cut_hyb)
				F = -6.96*F_forceBind
			elif r12 > comm_cut - 2:
				U = 6.96*U_barrier_comm*((r12-(comm_cut-2))/2)**2
				F = -6.96*U_barrier_comm*(r12-(comm_cut-2))
			else:
				U = 0
				F = 0

			### write point to file
			f.write(f"{i+1} {r12:.4f} {U:.4f} {F:.4f}\n")

	### result
	return npoint


################################################################################
### Calculation Managers

### calculate the positions of a swirl, given set parameters
def staticSwirl(nbead, amplitude, nperiod, radius, r12_eq, nsubstep):
	r = np.zeros((nbead,3))
	nsubstep = 20
	w = 0; z = 0
	for bi in range(nbead):
		for i in range(nsubstep):
			v = amplitude*nperiod/radius*np.cos(w*nperiod/radius)
			w += r12_eq/nsubstep/np.sqrt(1+v**2)
		r[bi,0] = radius*np.cos(w/radius)
		r[bi,1] = radius*np.sin(w/radius)
		r[bi,2] = amplitude*np.sin(nperiod*w/radius)
		mismatch = np.linalg.norm(r[0]-r[bi]) - r12_eq
		if bi > 1 and mismatch < 0:
			if bi < nbead/nperiod:
				print("Error: Too dense to initiate swirl, try again with larger box.\n")
				sys.exit()
			mismatch += r12_eq*(bi-nbead+1)
			return r, mismatch
	return r, mismatch


### initialize the positions of a swirl, adjusting the parameters to ensure end-to-end connectivity
def initPositionsSwirl(nbead, dbox, r12_eq, box_frac, nsubstep, max_iteration, tolerance):
	r = np.zeros((nbead,3))
	l = r12_eq*nbead
	if l/np.pi < box_frac*dbox:
		radius = l/np.pi/2
		for bi in range(nbead):
			r[bi,0] = radius*np.cos(bi/nbead*2*np.pi)
			r[bi,1] = radius*np.sin(bi/nbead*2*np.pi)
			r[bi,2] = 0
		print("Initialized scaffold as circle.")
		points = np.zeros((1,nbead,3))
		points[0] = r
	else:
		radius = box_frac*dbox/2
		pf = l/(np.pi*box_frac*dbox)
		nperiod = np.ceil((pf-0.4)/0.6)
		amplitude = (l/np.pi-0.4*box_frac*dbox)/(0.6*2*nperiod)
		r, mismatch = staticSwirl(nbead, amplitude, nperiod, radius, r12_eq, nsubstep)
		points = np.zeros((max_iteration,nbead,3))
		points[0] = r
		iteration = 0
		while abs(mismatch) > tolerance:
			if iteration == max_iteration:
				break
			else:
				iteration += 1
				amplitude += -mismatch/(np.pi*0.6*2*nperiod)
				r, mismatch = staticSwirl(nbead, amplitude, nperiod, radius, r12_eq, nsubstep)
				points[iteration] = r
		points = points[:iteration+1]
		if iteration < max_iteration:
			print(f"Swirl converged after {iteration} iteration.")
		else:
			print("Flag: Swirl did not converge.")
	return r, points


### initialize positions, keeping strands together
def initPositions(strands, p):
	print("Initializing positions...")

	### parameters
	max_nfail_strand = 20
	max_nfail_bead = 20

	### initializations
	nbead_placed = p.n_scaf
	nbead_locked = p.n_scaf
	nstrand_locked = 1
	r = np.zeros((p.nbead,3))

	### scaffold positions
	r[:p.n_scaf] = initPositionsSwirl(p.n_scaf, p.dbox, p.r12_eq, 0.8, 100, 100, p.r12_eq/4)[0]

	### loop over beads
	nfail_strand = 0
	while nbead_placed < p.nbead:
		rbi = nbead_placed
		obi = rbi2obi(rbi, p)

		### attempt to place bead
		nfail_bead = 0
		while True:

			### position linked to previous bead
			if strands[obi] == strands[obi-1]:
				r_propose = ars.applyPBC(r[rbi-1] + p.r12_eq*ars.randUnitVec(p.rng), p.dbox)

			### random position for new strand
			else:
				r_propose = ars.randPos(p.dbox, p.rng)

			### evaluate position, break loop if no overlap
			if not ars.checkOverlap(r_propose,r[:rbi],p.sigma,p.dbox):
				break

			### if loop not broken update bead fail count
			nfail_bead += 1

			### break bead loop if too much failure
			if nfail_bead == max_nfail_bead:
				break

		### set position if all went well
		if nfail_bead < max_nfail_bead:
			r[rbi] = r_propose
			nbead_placed += 1

			### update locked strands if end of strand
			if obi+1 == p.n_ori or strands[obi] < strands[obi+1]:

				### debug output
				if p.debug:
					print(f"Placed staple {nstrand_locked} after {nfail_strand} failures.")

				### updates
				nbead_locked += strands.count(strands[obi])
				nstrand_locked += 1
				nfail_strand = 0

		### reset strand if too much failure
		else:
			nfail_strand += 1
			nbead_placed = nbead_locked

			### give up if too many strand failures
			if nfail_strand == max_nfail_strand:
				print("Error: Could not place beads, try again with larger box.\n")
				sys.exit()

	### return positions
	return r


### initialize positions of given staples, keeping strands together
def randomStapPositions(r, strands, is_reserved_strand, p):
	print("Randomizing reserved staple positions...")

	### parameters
	max_nfail_strand = 20
	max_nfail_bead = 20

	### initializations
	nbead_placed = p.n_scaf
	nbead_locked = p.n_scaf
	nstrand_locked = 1

	### loop over beads
	nfail_strand = 0
	while nbead_placed < p.nbead:
		rbi = nbead_placed
		obi = rbi2obi(rbi, p)
		nfail_bead = 0

		### set to current position for regular beads
		if not is_reserved_strand[strands[obi]]:
			r_propose = r[rbi]

		### placement loop for reserved beads
		else:
			while True:

				### position linked to previous bead
				if strands[obi] == strands[obi-1]:
					r_propose = ars.applyPBC(r[rbi-1] + p.r12_eq*ars.randUnitVec(p.rng), p.dbox)

				### random position for new strand
				else:
					r_propose = ars.randPos(p.dbox, p.rng)

				### evaluate position, break loop if no overlap
				if not ars.checkOverlap(r_propose,r[:rbi],p.sigma,p.dbox):
					break

				### if loop not broken update bead fail count
				nfail_bead += 1

				### break bead loop if too much failure
				if nfail_bead == max_nfail_bead:
					break

		### set position if all went well
		if nfail_bead < max_nfail_bead:
			r[rbi] = r_propose
			nbead_placed += 1

			### update locked strands if end of strand
			if obi+1 == p.n_ori or strands[obi] < strands[obi+1]:

				### debug output
				if p.debug:
					print(f"Placed staple {nstrand_locked} after {nfail_strand} failures.")

				### updates
				nbead_locked += strands.count(strands[obi])
				nstrand_locked += 1
				nfail_strand = 0

		### reset strand if too much failure
		else:
			nfail_strand += 1
			nbead_placed = nbead_locked

			### give up if too many strand failures
			if nfail_strand == max_nfail_strand:
				print("Error: Could not place beads, try again with larger box.\n")
				sys.exit()

	### return positions
	return r


################################################################################
### Utilify Functions

### see if template exists anywhere in origami (not including bridging bonds)
def checkAngleDeactTemplateMatch(left180, mid180, right180, backbone_neighbors, complements, is_crossover, p):
	for bi in range(p.n_scaf):

		### check for core scaffold
		bi_5p = backbone_neighbors[bi][0]
		bi_3p = backbone_neighbors[bi][1]
		if bi_5p == -1 or bi_3p == -1:
			continue

		#-------- working 5' to 3' --------#
		pass_5p = True

		### get core scaffold
		a = bi_5p
		b = bi
		c = bi_3p

		### check for core staples
		if complements[b] == -1:
			pass_5p = False

		### check left angle
		if left180 == is_crossover[a]:
			pass_5p = False

		### check middle angle
		if mid180 == is_crossover[b]:
			pass_5p = False

		### check middle angle
		if right180 == is_crossover[c]:
			pass_5p = False

		#-------- working 3' to 5' --------#
		pass_3p = True

		### get core scaffold
		a = bi_3p
		b = bi
		c = bi_5p

		### check for core staples
		if complements[b] == -1:
			pass_3p = False

		### check left angle
		if left180 == is_crossover[a]:
			pass_3p = False

		### check middle angle
		if mid180 == is_crossover[b]:
			pass_3p = False

		### check middle angle
		if right180 == is_crossover[c]:
			pass_3p = False

		#-------- check for match --------#
		if pass_5p or pass_3p:
			return True
		else:
			continue

	### default result
	return False


### renumber atoms (starting from 0) in angle template starting
def renumberAtoms_angleTemplate(atoms, bonds, angls, edges):
	atoms_bi = [row[2] for row in atoms]
	for atomi in range(len(atoms)):
		atoms[atomi][2] = atomi
	for bondi in range(len(bonds)):
		for i in range(1,3):
			bonds[bondi][i] = atoms_bi.index(bonds[bondi][i])
	for angli in range(len(angls)):
		for i in range(1,4):
			angls[angli][i] = atoms_bi.index(angls[angli][i])
	for edgei in range(len(edges)):
		edges[edgei] = atoms_bi.index(edges[edgei])
	return atoms, bonds, angls, edges


### renumber atoms (starting from 0) in bridge template starting
def renumberAtoms_bridgeTemplate(atoms, bonds, edges):
	atoms_bi = [row[2] for row in atoms]
	for atomi in range(len(atoms)):
		atoms[atomi][2] = atomi
	for bondi in range(len(bonds)):
		for i in range(1,3):
			bonds[bondi][i] = atoms_bi.index(bonds[bondi][i])
	for edgei in range(len(edges)):
		edges[edgei] = atoms_bi.index(edges[edgei])
	return atoms, bonds, edges


### remove duplicate elements from array along first dimension
def removeDuplicateElements(array):
	array_seen = []
	for i in range(len(array)):
		new = True
		for j in range(len(array_seen)):
			if array[i] == array_seen[j]:
				new = False
				break
		if new:
			array_seen.append(array[i])
	return array_seen


### unzip four zipped arrays
def unzip4(zipped):
	a = []
	b = []
	c = []
	d = []
	for i in range(len(zipped)):
		a.append(zipped[i][0])
		b.append(zipped[i][1])
		c.append(zipped[i][2])
		d.append(zipped[i][3])
	return a,b,c,d

### unzip three zipped arrays
def unzip3(zipped):
	a = []
	b = []
	c = []
	for i in range(len(zipped)):
		a.append(zipped[i][0])
		b.append(zipped[i][1])
		c.append(zipped[i][2])
	return a,b,c


### return 5' and 3' neighbors, accounting for bridging bonds
def getAssembledNeighbors(bi, backbone_neighbors, complements, p):

	### for vast majority of cases, this is the result
	bi_5p = backbone_neighbors[bi][0]
	bi_3p = backbone_neighbors[bi][1]

	### only if attempting end bridging reactions
	if p.bridgeEnds:

		### check for bridging bond on 5' side
		if bi_5p == -1:
			if complements[bi] != -1:
				if backbone_neighbors[complements[bi]][1] != -1:
					if complements[backbone_neighbors[complements[bi]][1]] != -1:
						bi_5p = complements[backbone_neighbors[complements[bi]][1]]

		### check for bridging bond on 3' side
		if bi_3p == -1:
			if complements[bi] != -1:
				if backbone_neighbors[complements[bi]][0] != -1:
					if complements[backbone_neighbors[complements[bi]][0]] != -1:
						bi_3p = complements[backbone_neighbors[complements[bi]][0]]

	### return result
	return bi_5p,bi_3p


### get origami bead index from real bead index
def rbi2obi(rbi, p):
	if rbi < p.n_scaf:
		return rbi
	elif rbi < p.nbead:
		return (rbi-p.n_scaf)%p.n_stap + p.n_scaf
	else:
		print("Error: Origami bead index not defined for the dummy atom.\n")


################################################################################
### Misbinding

### calcualte factors used to determine complimentarity
def calcCompFactors(complements, p):

	### parameters
	nnt_bead = 8
	salt = 1

	### energy function parameters class
	efp = energyFuncParams(p)

	### initialize positions
	X = p.rng_mis.normal(size=(p.ncompFactor, p.n_scaf))

	### calculate interactions
	if p.nmisBond and (p.optCompFactors or p.optCompEfunc):

		### get sequences
		seqs = getSequences(nnt_bead, p)

		### calculate interaction energies
		E_mis_gt = calcInteractions(seqs, efp.e_cut, p.T, salt)

		### optimize positions
		if p.optCompFactors:
			X = fitPositionsToEnergies(X, E_mis_gt, efp)

		### energy function optimization
		if p.optCompEfunc:
			efp = fitEfuncToEnergies(X, E_mis_gt, efp, p.nmisBond)

		### debug output
		if p.debug:

			### reconstruct energies
			calcEnergy = defineEfuncReal(efp)
			E_mis_opt = calcEnergy(getDistances(X))

			### mean square error
			mask_int = (E_mis_gt<efp.e_cut) | (E_mis_opt<efp.e_cut)
			mse_all = np.mean((E_mis_gt - E_mis_opt) ** 2)
			mse_int = np.mean((E_mis_gt[mask_int] - E_mis_opt[mask_int]) ** 2)
			print("\nMisbinding MSE:")
			print(f"All pairs:         {mse_all:.6f}")
			print(f"Interacting pairs: {mse_int:.6f}")

			### energy histogram
			counts_gt = np.histogram(E_mis_gt[E_mis_gt<efp.e_cut], bins=p.nmisBond, range=(efp.e_min,efp.e_cut))[0]
			counts_opt = np.histogram(E_mis_opt[E_mis_opt<efp.e_cut], bins=p.nmisBond, range=(efp.e_min,efp.e_cut))[0]
			counts_gt[0] += sum(E_mis_gt<efp.e_min)
			print("\nMisbinding energy levels:")
			print(f"Ground Truth: {counts_gt}")
			print(f"Optimized:    {counts_opt}")

			### average misbinding energy
			e_levels = np.linspace(efp.e_min,efp.e_cut,p.nmisBond+1)[:-1]
			e_levels += (e_levels[1]-e_levels[0])/2
			e_mis_avg = np.sum(e_levels*counts_opt)/p.nbead
			print(f"\nAverage total misbinding energy per bead: {e_mis_avg:0.2f}")

	### cutoff distances
	mis_d2_cuts = np.zeros(p.nmisBond+1)
	if p.nmisBond:
		calcEnergy = defineEfuncReal(efp, vector=False)
		for i in range(p.nmisBond):
			e = efp.e_min + (i+1)*(efp.e_cut-efp.e_min)/p.nmisBond
			def calcEnergyRoot(d):
				return calcEnergy(d)-e
			mis_d2_cuts[i+1] = brentq(calcEnergyRoot, 0, efp.d_cut)**2

	### set complementary factors
	comp_factors = np.zeros((p.nbead+1, p.ncompFactor))
	comp_factors[:p.n_scaf] = X.T
	for rbi in range(p.n_scaf,p.nbead):
		obi = rbi2obi(rbi, p)
		if complements[obi] != -1:
			comp_factors[rbi] = comp_factors[complements[obi]]
		else:
			comp_factors[rbi] = p.rng_mis.normal(size=p.ncompFactor)

	### result
	return comp_factors, mis_d2_cuts


### generate random sequence
def getSequences(nnt_bead, p):
	useM13 = True
	m13file = "m13mp18.txt"

	### selecting from section of viral sequence
	if useM13:
		with open(m13file) as f:
			m13seq = f.readlines()[0]
		start = p.rng_mis.integers(len(m13seq)-p.n_scaf*nnt_bead+1)
		seqs = [None]*p.n_scaf
		for i in range(p.n_scaf):
			seqs[i] = m13seq[start+i*nnt_bead:start+(i+1)*nnt_bead]

	### selecting each base randomly
	else:
		bases = list('ATCG')
		seqs = [None]*p.n_scaf
		for i in range(p.n_scaf):
			seqs[i] = ''.join(p.rng_mis.choice(bases, nnt_bead))

	### result
	return seqs


### calcualte and process interaction energies from Santa Lucia parameters
def calcInteractions(seqs, e_cut, T, salt):
	print("Calculating interaction energies...")
	from nupack import Model
	from nupack import mfe
	nbead = len(seqs)

	### set nupack model
	modelDNA = Model(material='dna', celsius=T-273, sodium=salt)

	### calculate interactions
	E = np.zeros((nbead, nbead))
	for i in range(nbead):
		for j in range(i, nbead):
			scaf1_seq = seqs[i]
			scaf2_seq = seqs[j]
			stap1_seq = getComplement(scaf1_seq)
			stap2_seq = getComplement(scaf2_seq)
			E[i,j] = calcSantaLuciaNupack(scaf1_seq, stap2_seq, modelDNA, mfe)
			E[j,i] = calcSantaLuciaNupack(scaf2_seq, stap1_seq, modelDNA, mfe)

	### extract misbinding energies
	E_mis = ((E+E.T)/2)[np.triu_indices(nbead,1)]
	E_mis[E_mis>e_cut] = e_cut

	### result
	return E_mis


### calculate binding free energy from Santa Lucia parameters
def calcSantaLuciaNupack(seq1, seq2, model, mfe):
	mfe_structures = mfe(strands=[seq1,seq2], model=model)
	if len(mfe_structures) == 0: return 0
	return mfe_structures[0].energy


### set parameters and define energy function for end use
def defineEfuncReal(efp, vector=True):

	### check monotonicity of both cubic sections
	if not checkCubicMonotonic(efp.e_min, efp.e_swp, efp.m_min, efp.m_swp, efp.d_swp):
		return 0
	if not checkCubicMonotonic(efp.e_swp, efp.e_cut, efp.m_swp, efp.m_cut, efp.d_cut-efp.d_swp):
		return 0

	### calculate prefactors
	h1 = efp.d_swp
	a1 = 2*(efp.e_min-efp.e_swp)/h1**3 + (efp.m_min+efp.m_swp)/h1**2
	b1 = 3*(efp.e_swp-efp.e_min)/h1**2 - (2*efp.m_min+efp.m_swp)/h1
	h2 = efp.d_cut - efp.d_swp
	a2 = 2*(efp.e_swp-efp.e_cut)/h2**3 + (efp.m_swp+efp.m_cut)/h2**2
	b2 = 3*(efp.e_cut-efp.e_swp)/h2**2 - (2*efp.m_swp+efp.m_cut)/h2

	### define function
	if vector:
		def calcEnergy(D):
			E = np.ones(D.shape)*efp.e_cut
			mask1 = D < efp.d_swp
			dD1 = D[mask1]
			E[mask1] = a1*dD1**3 + b1*dD1**2 + efp.m_min*dD1 + efp.e_min
			mask2 = (D >= efp.d_swp) & (D < efp.d_cut)
			dD2 = D[mask2] - efp.d_swp
			E[mask2] = a2*dD2**3 + b2*dD2**2 + efp.m_swp*dD2 + efp.e_swp
			return E
	else:
		def calcEnergy(d):
			if d < efp.d_swp:
				return a1*d**3 + b1*d**2 + efp.m_min*d + efp.e_min
			elif d < efp.d_cut:
				return a2*(d-efp.d_swp)**3 + b2*(d-efp.d_swp)**2 + efp.m_swp*(d-efp.d_swp) + efp.e_swp
			else:
				return efp.e_cut

	### result
	return calcEnergy


### set parameters and define energy function for optimizing positions
def defineEfuncOpt(efp):

	### parameters for transition quadratics
	a_flat = efp.m_cut/(2*(efp.d_cut-efp.d_swc))
	b_flat = - (efp.m_cut*efp.d_swc)/(efp.d_cut-efp.d_swc)
	c_flat = efp.e_cut-a_flat*efp.d_cut**2-b_flat*efp.d_cut
	a_att = (efp.m_cut-efp.m_att)/(2*(efp.d_cut-efp.d_swc))
	b_att = efp.m_att - ((efp.m_cut-efp.m_att)*efp.d_swc)/(efp.d_cut-efp.d_swc)
	c_att = efp.e_cut-a_att*efp.d_cut**2-b_att*efp.d_cut
	e_flat = a_flat*efp.d_swc**2 + b_flat*efp.d_swc + c_flat
	e_att = a_att*efp.d_swc**2 + b_att*efp.d_swc + c_att

	### get real energy function
	calcEnergyReal = defineEfuncReal(efp)

	### vectorized energy function
	def calcEnergy(D, E_target):
		E = np.ones(D.shape)*e_flat
		mask1 = D < efp.d_cut
		D1 = D[mask1]
		E[mask1] = calcEnergyReal(D1)
		mask2 = (D >= efp.d_cut) & (D < efp.d_swc) & (E_target >= efp.e_cut)
		D2 = D[mask2]
		E[mask2] = a_flat*D2**2 + b_flat*D2 + c_flat
		mask3 = (D >= efp.d_cut) & (D < efp.d_swc) & (E_target < efp.e_cut)
		D3 = D[mask3]
		E[mask3] = a_att*D3**2 + b_att*D3 + c_att
		mask4 = (D >= efp.d_swc) & (E_target < efp.e_cut)
		D4 = D[mask4]
		E[mask4] = e_att + efp.m_att*(D4-efp.d_swc)
		return E

	### result
	return calcEnergy, e_flat, e_att


### find set of positions that satisfy the given pairwise energies
def fitPositionsToEnergies(X0, E_mis_gt, efp):
	print("Fitting misbinding positions...")
	nbead = X0.shape[1]
	ndim = X0.shape[0]

	### initial energy function
	calcEnergy, e_flat = defineEfuncOpt(efp)[:2]

	### target energy
	E_target = copy.deepcopy(E_mis_gt)
	E_target[E_mis_gt==efp.e_cut] = e_flat

	### optimize positions
	p_opt = minimize(lossPos, X0.flatten(), method='BFGS', args=(E_target, calcEnergy, ndim))
	X_opt = p_opt.x.reshape(ndim,-1)

	### result
	print(p_opt.message)
	return X_opt


### define loss function for optimizing positions
def lossPos(x, E_target, calcEnergy, ndim):
	X = x.reshape(ndim,-1)
	E_pred = calcEnergy(getDistances(X), E_target)
	return np.sum((E_pred - E_target)**2)


### find energy function parameters that satisfy the given pairwise energies
def fitEfuncToEnergies(X0, E_mis_gt, efp, nlevel):
	print("Fitting misbinding energy function...")
	nbead = X0.shape[1]
	ndim = X0.shape[0]

	### initial distances
	D0 = getDistances(X0)

	### target energy histogram
	E_hist_target = np.histogram(E_mis_gt[E_mis_gt<efp.e_cut], bins=nlevel, range=(efp.e_min,efp.e_cut))[0]
	E_hist_target[0] += sum(E_mis_gt<efp.e_min)

	### initial parameters to adjust
	x = [efp.d_swp, efp.d_cut, efp.e_swp, efp.m_min, efp.m_swp, efp.m_cut]
	bounds = [[0.1,10], [0.1,10], [-12,-6], [0,20], [0,20], [0,10]]

	### optimize energy function
	efp_opt = copy.deepcopy(efp)
	p_opt = differential_evolution(lossEfunc, bounds, args=(efp_opt, D0, E_hist_target, nlevel))

	### result
	print(p_opt.message)
	return efp_opt


### define loss function for optimizing energy function
def lossEfunc(x, efp, D0, E_hist_target, nlevel):
	efp.d_swp, efp.d_cut, efp.e_swp, efp.m_min, efp.m_swp, efp.m_cut = x
	calcEnergy = defineEfuncReal(efp)
	if calcEnergy == 0: return np.inf
	E_mis_pred = calcEnergy(D0)
	E_hist_pred = np.histogram(E_mis_pred[E_mis_pred<efp.e_cut], bins=nlevel, range=(efp.e_min,efp.e_cut))[0]
	return np.sum(((E_hist_pred - E_hist_target))**2)


### check if cubic function is monotonic over interval
def checkCubicMonotonic(y0, y1, m0, m1, d):
	A = 6*y0 + 3*d*m0 - 6*y1 + 3*d*m1
	B = -6*y0 - 4*d*m0 + 6*y1 - 2*d*m1
	C = d * m0

	def g(t):
		return A*t**2 + B*t + C

	values = [g(0), g(1)]
	if A != 0:
		t_vertex = -B / (2*A)
		if 0 <= t_vertex <= 1:
			values.append(g(t_vertex))
	return all(val >= 0 for val in values)


### get complimentary sequence
def getComplement(seq0):
	comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
	return ''.join(comp[b] for b in reversed(seq0))


### calculate interaction distances
def getDistances(X):
	i_idx, j_idx = np.triu_indices(X.shape[1], 1)
	return np.linalg.norm(X[:,i_idx] - X[:,j_idx], axis=0)


### parameters for energy function
class energyFuncParams:

	### initialize
	def __init__(self, p):

		### energy bounds
		self.e_min = -p.U_mis_max/6.96
		self.e_cut = -p.U_mis_min/6.96

		### efunc opt
		self.d_swp = 0.26
		self.d_cut = 0.92
		self.e_swp = -8.71
		self.m_min = 11.44
		self.m_swp = 7.37
		self.m_cut = 2.46

		### pos opt
		self.d_swc = 1.2
		self.m_att = 1

		if p.ncompFactor == 3:
			### efunc opt
			self.d_swp = 0.48
			self.d_cut = 1.38
			self.e_swp = -9.07
			self.m_min = 5.87
			self.m_swp = 5.02
			self.m_cut = 2.04

			### pos opt
			self.d_swc = 1.8
			self.m_att = 1

		if p.ncompFactor == 4:
			### efunc opt
			self.d_swp = 0.63
			self.d_cut = 1.78
			self.e_swp = -9.57
			self.m_min = 5.29
			self.m_swp = 4.16
			self.m_cut = 1.87

			### pos opt
			self.d_swc = 2.2
			self.m_att = 1

		if p.ncompFactor not in [1,2,3,4]:
			print("Flag: Number of complementary factors not standard, using energy function for 2 complementary factors.")


################################################################################
### DNAfold

### translate caDNAno design to DNAfold model
def buildDNAfoldModel(cadFile, p):

	### parameters
	nnt_bead = 8

	### parse the caDNAno file
	scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap = parseCaDNAno(cadFile)
	
	### initial calculations
	print("Building DNAfold model...")
	p.n_scaf = nnt_scaf // nnt_bead
	p.n_stap = nnt_stap // nnt_bead
	p.n_ori = p.n_scaf + p.n_stap
	p.nbead = p.n_scaf + p.n_stap*p.stap_copies
	print("Using " + str(p.n_scaf) + " scaffold beads and " + str(p.n_stap) + " staple beads.")

	### initialze interaction and geometry arrays
	strands = [0 for i in range(p.n_ori)]
	backbone_neighbors = [[-1,-1] for i in range(p.n_ori)]
	complements = [-1 for i in range(p.n_ori)]
	vstrands = [0 for i in range(p.n_ori)]
	is_crossover = [False for i in range(p.n_scaf)]

	### initialize nucleotide and bead indices
	ni_current = 0
	bi_current = 0

	### kick off nucleotide and bead indexing with 5' scaffold end
	ni_scaffoldArr = find(fiveP_end_scaf[0], fiveP_end_scaf[1], scaffold)
	scaffold[ni_scaffoldArr].extend([ni_current, bi_current])
	vstrands[bi_current] = scaffold[ni_scaffoldArr][0]
	vstrand = scaffold[ni_scaffoldArr][0]
	vstrand_prev = vstrand

	### error message
	if scaffold[ni_scaffoldArr][0] % 2 == 0:
		if scaffold[ni_scaffoldArr][1] % nnt_bead != 0:
			print(f"Error: Scaffold 5' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
			sys.exit()
	elif scaffold[ni_scaffoldArr][1] % nnt_bead != 7:
		print(f"Error: Scaffold 5' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
		sys.exit()

	### track along scaffold until 3' end eached
	while scaffold[ni_scaffoldArr][4] != -1:
		ni_scaffoldArr = find(scaffold[ni_scaffoldArr][4], scaffold[ni_scaffoldArr][5], scaffold)

		### update nucleotide and bead indices
		ni_current += 1
		bi_current = ni_current // nnt_bead
		scaffold[ni_scaffoldArr].extend([ni_current, bi_current])
		vstrand = scaffold[ni_scaffoldArr][0]

		### store vstrand and backbone bonds for new beads
		if bi_current > (ni_current-1) // nnt_bead:
			backbone_neighbors[bi_current][0] = bi_current-1
			backbone_neighbors[bi_current-1][1] = bi_current
			vstrands[bi_current] = scaffold[ni_scaffoldArr][0]

		### error message
		elif vstrand != vstrand_prev:
			print("Error: Scaffold crossover not located at nultiple-of-8 position.\n")
			sys.exit()
		vstrand_prev = vstrand

	### error message
	if scaffold[ni_scaffoldArr][0] % 2 == 0:
		if scaffold[ni_scaffoldArr][1] % nnt_bead != 7:
			print(f"Error: Scaffold 3' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
			sys.exit()
	elif scaffold[ni_scaffoldArr][1] % nnt_bead != 0:
		print(f"Error: Scaffold 3' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
		sys.exit()

	### loop over staples
	nstap = len(fiveP_ends_stap)
	for sti in range(nstap):

		### new nucleotide and bead incides
		ni_current += 1
		bi_current = ni_current // nnt_bead

		### pick up nucleotide and bead indexing with 5' staple end
		ni_staplesArr = find(fiveP_ends_stap[sti][0],fiveP_ends_stap[sti][1], staples)
		staples[ni_staplesArr].extend([ni_current, bi_current])
		strands[bi_current] = sti+1
		vstrands[bi_current] = staples[ni_staplesArr][0]
		vstrand = staples[ni_staplesArr][0]
		vstrand_prev = vstrand

		### identify paired beads
		if scaffold[ni_staplesArr][2] != -1 or scaffold[ni_staplesArr][4] != -1:
			complements[scaffold[ni_staplesArr][7]] = bi_current
			complements[bi_current] = scaffold[ni_staplesArr][7]

		### error message
		if staples[ni_staplesArr][0] % 2 == 0:
			if staples[ni_staplesArr][1] % nnt_bead != 7:
				print(f"Error: Staple 5' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
				sys.exit()
		elif staples[ni_staplesArr][1] % nnt_bead != 0:
			print(f"Error: Staple 5' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
			sys.exit()

		### track along staple until 3' end eached
		while staples[ni_staplesArr][4] != -1:
			ni_staplesArr = find(staples[ni_staplesArr][4], staples[ni_staplesArr][5], staples)

			### update nucleotide and bead indices
			ni_current += 1
			bi_current = ni_current // nnt_bead
			staples[ni_staplesArr].extend([ni_current, bi_current])
			vstrand = staples[ni_staplesArr][0]

			### store vstrand, strand, and backbone bonds for new beads
			if bi_current > (ni_current-1) // nnt_bead:
				strands[bi_current] = sti+1
				backbone_neighbors[bi_current][0] = bi_current-1
				backbone_neighbors[bi_current-1][1] = bi_current
				vstrands[bi_current] = scaffold[ni_staplesArr][0]

				### identify paired beads
				if scaffold[ni_staplesArr][2] != -1 or scaffold[ni_staplesArr][4] != -1:
					complements[scaffold[ni_staplesArr][7]] = bi_current
					complements[bi_current] = scaffold[ni_staplesArr][7]

				### error message
				elif vstrand != vstrand_prev:
					print(f"Error: Staple crossover not located at nultiple-of-8 position (vstrand {vstrand}).\n")
					sys.exit()
				else:
					vstrand_prev = vstrand

		### error message
		if staples[ni_staplesArr][0] % 2 == 0:
			if staples[ni_staplesArr][1] % nnt_bead != 0:
				print(f"Error: Staple 3' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
				sys.exit()
		elif staples[ni_staplesArr][1] % nnt_bead != 7:
			print(f"Error: Staple 3' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
			sys.exit()

	### identify crossovers
	for bi in range(1, p.n_scaf):
		if vstrands[bi] != vstrands[bi-1]:
			if strands[bi] == strands[bi-1]:
				is_crossover[bi] = True
				is_crossover[bi-1] = True

	### end crossover
	if vstrands[0] != vstrands[p.n_scaf-1]:
		is_crossover[0] = True
		is_crossover[p.n_scaf-1] = True

	### adjustments for circular scaffold
	if p.circularScaf:
		backbone_neighbors[0][0] = p.n_scaf-1
		backbone_neighbors[p.n_scaf-1][1] = 0

	### strand count
	p.nstrand = max(strands)+1

	### return results
	return strands, backbone_neighbors, complements, is_crossover, p


### extract necessary info from caDNAno file
def parseCaDNAno(cadFile):
	print("Parsing caDNAno file...")
	
	### load caDNAno file
	ars.checkFileExist(cadFile,"caDNAno")
	with open(cadFile, 'r') as f:
		json_string = f.read()
	j = json.loads(json_string)

	### initialize
	scaffold = []
	staples = []
	fiveP_end_scaf = []
	fiveP_ends_stap = []

	
	### loop over virtual strands
	for el1 in j["vstrands"]:
		
		### loop over the elements of the virtual strand
		for el2_key, el2 in el1.items():
			
			### read virtual strand index
			if el2_key == "num":
				vi = el2
			
			### read scaffold side of virtual strand
			elif el2_key == "scaf":
				
				### loop over nucleotides
				for ni_vstrand, neighbors in enumerate(el2):
					
					### store virtual strand index and nucleotide index for current nucleotide and its neighbors
					nt = [vi, int(ni_vstrand)]
					for s in neighbors:
						nt.append(int(s))
					scaffold.append(nt)
					
					### identify 5' end
					if nt[2] == -1 and nt[4] != -1:
						fiveP_end_scaf = nt
			
			### read staple side of helix
			elif el2_key == "stap":
				
				### loop over nucleotides
				for ni_vstrand, neighbors in enumerate(el2):
					
					### store virtual strand index and nucleotide index for current nucleotide and its neighbors
					nt = [vi, int(ni_vstrand)]
					for s in neighbors:
						nt.append(int(s))
					staples.append(nt)
					
					### identify 5' end
					if nt[2] == -1 and nt[4] != -1:
						fiveP_ends_stap.append(nt)
			
	### tally up the nucleotides
	nnt_scaf = sum(1 for s in scaffold if s[2] != -1 or s[4] != -1)
	nnt_stap = sum(1 for s in staples if s[2] != -1 or s[4] != -1)

	### error message
	if fiveP_end_scaf is None:
		print("Error: Scaffold 5' end not found.\n")
		sys.exit()
	
	### result
	print(f"Found {nnt_scaf} scaffold nucleotides and {nnt_stap} staple nucleotides.")
	return scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap


### search for entry in strand/index list that matches given strand/index
def find(strand, index, list):
	for i,item in enumerate(list):
		if item[0] == strand and item[1] == index:
			if item[2] == -1 and item[3] == -1 and item[4] == -1 and item[5] == -1:
				return -1
			return i
	print("Error: Index not found in strand/index list.\n")
	sys.exit()


### adjust for scaffold cut location
def shiftScaffold(complements, is_crossover, p):

	### initialize
	complements_shifted = copy.deepcopy(complements)
	is_crossover_shifted = copy.deepcopy(is_crossover)

	### only for linear scaffolds with nonzero shift
	if not p.circularScaf and p.scaf_shift != 0:

		### adjust complements
		for i in range(p.n_scaf):
			complements_shifted[i] = complements[ np.mod(i+p.scaf_shift, p.n_scaf) ]
		for i in range(p.n_scaf,p.n_ori):
			complements_shifted[i] = np.mod(complements[i]-p.scaf_shift, p.n_scaf)

		### adjust crossover
		for i in range(p.n_scaf):
			is_crossover_shifted[i] = is_crossover[ np.mod(i+p.scaf_shift, p.n_scaf) ]

	### result
	return complements_shifted, is_crossover_shifted


### run the script
if __name__ == "__main__":
	main()
	print()

