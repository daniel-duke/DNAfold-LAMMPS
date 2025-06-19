import armament as ars
import utils
import utilsLocal
import argparse
import parameters
import numpy as np
import random
import itertools
import copy
import json
import sys
import os

## Description
# this script takes a caDNAno json file, creates the interaction and geometry
  # arrays necessary for the dnafold model, and writes the geometry and input
  # files necessary to simulate the system in lammps.
# all indexing starts at 0, then is increased by 1 when written to lammps files.
# reserved staples files list the strand indices of the staples (starting from 2)
  # to reserve, whereas reserved scaffold files list the scaffold bead indices
  # (starting from 1) paired to the staples to reserve; the code uses reserved
  # staple files by default, but it can use reserved scaffold if neccessary.
# if the angle dehyb frequency is too high, there is a risk of missing some
  # angle dehybridizations (permenantly); this is because the angle dehyb
  # templates assume all hyb bonds still exist, so if the hyb bond is broken 
  # before the angle dehyb can break the angle, the dehyb template will never
  # apply; this also means the risk also rises if the hyb cutoff is lowered;
  # the main

# To Do
# find optimal tradeoff between commuication cutoff and bond break, parameterize
  # 90 degree angular potential, reactions that shortens crossover bond length; 
  # make sure bridge is accurate for multiple staple copies.


################################################################################
### Parameters

def main():

	### where to get files
	useMyFiles = True

	### extract files from my mac
	if useMyFiles:

		### chose design
		desID = "2HBx4"				# design identification
		simTag = ""					# added to desID to get name of simulation folder
		simType = "experiment"		# where create simulation folder
		rstapTag = None				# if reserving staples, tag for reserved staples file (None for all staples)
		rseed = 2					# random seed (also used for naming precise simulation folder)

		### choose parameters
		nstep			= 1E6		# steps		- number of simulation steps
		nstep_relax		= 1E5		# steps		- number of steps for relaxation
		dump_every		= 1E4		# steps		- number of steps between positions dumps
		dbox			= 40		# nm		- periodic boundary diameter
		forceBind		= False		# bool		- whether to force hybridization
		startBound		= False		# bool		- whether to start at caDNAno positions
		circularScaf	= False		# bool		- whether the scaffold is circular
		stap_copies		= 1 		# int		- number of copies for each staples

		### get input files
		cadFile = utilsLocal.getCadFile(desID)
		rstapFile = utilsLocal.getRstapFile(desID, rstapTag) if rstapTag is not None else None

		### set parameters
		cadFile, rstapFile, p = readInput(None, rseed, cadFile, rstapFile, nstep, nstep_relax, dump_every, dbox, forceBind, startBound, circularScaf, stap_copies)
		
		### set output folder
		outFold = utilsLocal.getSimHomeFold(desID, simTag, simType)

		### copy input files to output folder
		ars.createSafeFold(outFold)
		outCadFile = outFold + desID + ".json"
		os.system(f"cp \"{cadFile}\" \"{outCadFile}\"")
		if p.reserveStap:
			outRstapFile = outFold + "rstap_" + desID + rstapTag + ".txt"
			os.system(f"cp \"{rstapFile}\" \"{outRstapFile}\"")

	### use files in current folder
	if not useMyFiles:

		### get arguments
		parser = argparse.ArgumentParser()
		parser.add_argument('--inFile',	type=str,	required=True,	help='name of input file')
		parser.add_argument('--rseed',	type=int,	default=1,		help='random seed (also used to name output folder)')
		
		### set arguments
		args = parser.parse_args()
		inFile = args.inFile
		rseed = args.rseed

		### set parameters
		cadFile, rstapFile, p = readInput(inFile, rseed)

		### set output folder
		outFold = "./"

	### record parameters
	paramsFile = outFold + "parameters.txt"
	p.record(paramsFile)


################################################################################
### Heart

	### set random seed
	random.seed(p.rseed)

	### read caDNAno file
	strands, backbone_neighbors, complements, is_crossover, p = buildDNAfoldModel(cadFile, p)

	### read and write reserved staples file
	is_reserved_strand = readRstap(rstapFile, p)

	### create simulation folders
	outSimFold = outFold + f"sim{p.rseed:02.0f}/"
	outReactFold = outSimFold + "react/"
	ars.createEmptyFold(outReactFold)

	### write geometry files
	r, nhyb, nangle = composeGeo(outSimFold, strands, backbone_neighbors, complements, is_crossover, is_reserved_strand, cadFile, p)
	composeGeoVis(outSimFold, strands, backbone_neighbors, r, p)

	### write react files
	writeReactHybBond(outReactFold, backbone_neighbors, complements, p)
	nreact_angleHyb = writeReactAngleHyb(outReactFold, backbone_neighbors, complements, is_crossover, p)
	nreact_chargeDehyb, bridgeEnds = writeReactChargeDehyb(outReactFold, backbone_neighbors, complements, is_crossover, p)
	if bridgeEnds: writeReactBridgeBond(outReactFold, backbone_neighbors, complements, is_crossover, p)

	### write lammps input file
	writeInput(outSimFold, is_crossover, nhyb, nangle, nreact_angleHyb, nreact_chargeDehyb, bridgeEnds, p)


################################################################################
### File Handlers

### read input file
def readInput(inFile=None, rseed=1, cadFile=None, rstapFile=None, nstep=None, nstep_relax=1E5, dump_every=1E4, dbox=100, forceBind=False, startBound=False, circularScaf=True, stap_copies=1):

	### list keys that can have 'None' as their final value
	allow_none_default = {'rstapFile'}

	### define parameters with their default values
	param_defaults = {
		'cadFile':		cadFile,		# str			- name of caDNAno file (required)
		'rstapFile':	rstapFile,		# str			- name of reserved staples file
		'nstep': 		nstep,			# steps			- number of simulation steps (required)
		'nstep_relax': 	nstep_relax,	# steps			- number of steps for relaxation
		'dump_every': 	dump_every,		# steps			- number of steps between positions dumps
		'dt': 			0.01,			# ns			- integration time step
		'dbox': 		dbox,			# nm			- periodic boundary diameter
		'debug': 		False,			# bool			- whether to include debugging output
		'dehyb': 		True,			# bool			- whether to include dehybridization reactions (unnecessary for 1 staple copy)
		'forceBind': 	forceBind,		# bool			- whether to force hybridization (not applied if >1 staple copies)
		'startBound': 	startBound,		# bool			- whether to start at caDNAno positions
		'circularScaf':	circularScaf,	# bool			- whether the scaffold is circular
		'stap_copies': 	stap_copies,	# int			- number of copies for each staples
		'T':			300,			# K				- temperature
		'T_relax':		600,			# K				- temperature for relaxation
		'r_h_bead':		1.28,			# nm			- hydrodynamic radius of single bead
		'visc':			0.8472,			# mPa/s			- viscosity (units equivalent to pN*ns/mn^2)
		'sigma':		2.14,			# nm			- bead Van der Waals radius
		'epsilon':		4.0,			# kcal/mol		- WCA energy parameter
		'r12_eq':		2.72,			# nm			- equilibrium bead separation
		'k_x': 			120.0,			# kcal/mol/nm2	- backbone spring constant (standard definition)
		'r12_cut_hyb':	2.0,			# nm			- hybridization potential cutoff radius
		'U_hyb':		8.0,			# kcal/mol		- depth of hybridization potential
		'dsLp': 		50.0			# nm			- persistence length of dsDNA
	}

	### define types for each parameter
	param_types = {
		'cadFile':		str,
		'rstapFile':	str,
		'nstep':		lambda x: int(float(x)),
		'nstep_relax':	lambda x: int(float(x)),
		'dump_every':	lambda x: int(float(x)),
		'dt':			float,
		'dbox':			float,
		'debug':		lambda x: x.lower() == 'true',
		'dehyb':		lambda x: x.lower() == 'true',
		'forceBind':	lambda x: x.lower() == 'true',
		'startBound':	lambda x: x.lower() == 'true',
		'circularScaf':	lambda x: x.lower() == 'true',
		'stap_copies':	int,
		'T':			float,
		'T_relax':		float,
		'r_h_bead':		float,
		'visc':			float,
		'sigma':		float,
		'epsilon':		float,
		'r12_eq':		float,
		'k_x':			float,
		'r12_cut_hyb':	float,
		'U_hyb':		float,
		'dsLp':			float
	}

	### store parsed parameters
	params = {}

	### read parameters from file
	if inFile is not None:
		ars.testFileExist(inFile,'input')
		with open(inFile, 'r') as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith('#'):
					continue
				if '=' not in line:
					print(f"Error: Invalid line in config file: {line}")
					sys.exit()
				key, value = map(str.strip, line.split('=', 1))
				if key not in param_defaults:
					print(f"Unknown parameter: {key}")
					sys.exit()
				try:
					params[key] = param_types[key](value)
				except:
					print(f"Error parsing value for '{key}': {value}")
					sys.exit()

	### apply defaults and check for required values
	for key, default in param_defaults.items():
		if key not in params:
			if default is None:
				if key not in allow_none_default:
					print(f"Missing required parameter: {key}")
					sys.exit()
			params[key] = default

	### get caDNAno file
	cadFile = params['cadFile']
	del params['cadFile']

	### get reserved staples file
	rstapFile = params['rstapFile']
	del params['rstapFile']
	params['reserveStap'] = False if rstapFile is None else True

	### add parameters not set though input file
	params['rseed'] = rseed
	params['nnt_per_bead'] = 8

	### convert into parameters class and return
	p = parameters.parameters(params)
	return cadFile, rstapFile, p


### read reserved staples file
def readRstap(rstapFile, p):
	is_reserved_strand = [ False for i in range(p.nstrand) ]

	### skip if not reserving staples
	if not p.reserveStap:
		return is_reserved_strand

	### read staples
	ars.testFileExist(rstapFile,"reserved staples")
	with open(rstapFile, 'r') as f:
		reserved_strands = [ int(line.strip())-1 for line in f ]
	for si in range(len(reserved_strands)):
		is_reserved_strand[reserved_strands[si]] = True

	### return strand reservations status
	return is_reserved_strand


### write lammps geometry file, for simulation
def composeGeo(outSimFold, strands, backbone_neighbors, complements, is_crossover, is_reserved_strand, cadFile, p):
	print("Writing simulation geometry file...")

	### currently, oxDNA positions only available through local file structure
	oxdnaPositions = False
	desID = "triSS_edit"
	confTag = "_ideal"

	### initailize positions
	if p.startBound:
		stap_offset = 0.01
		if not oxdnaPositions:
			r = utils.initPositionsCaDNAno(cadFile)[0]
		else:
			topFile, confFile = utilsLocal.getOxFiles(desID, confTag)
			r = utils.initPositionsOxDNA(cadFile, topFile, confFile)[0]
		r[p.n_scaf:] += stap_offset
	else:
		r = initPositions(strands, p)
	r = np.append(r,np.zeros((1,3)),axis=0)

	### initialize
	molecules = np.ones(p.nbead+1,dtype=int)
	types = np.ones(p.nbead+1,dtype=int)
	charges = np.zeros(p.nbead+1)
	bonds = np.zeros((0,3),dtype=int)
	angles = np.zeros((0,4),dtype=int)

	### scaffold atoms
	for bi in range(p.n_scaf):
		molecules[bi] = bi + 1
		charges[bi] = is_crossover[bi] + 1

	### staple atoms
	nhyb = 0
	for ci in range(p.stap_copies):
		for obi in range(p.n_scaf,p.n_ori):
			rbi = obi + ci*p.n_stap
			molecules[rbi] = complements[obi] + 1
			types[rbi] = 2
			charges[rbi] = strands[obi] + ci*(p.nstrand-1) + 1
			if complements[obi] != -1:
				if ci == 0:
					nhyb += 1
			if is_reserved_strand[strands[obi]]:
				types[rbi] = 3
				r[rbi] = [0,0,0]

	### dummy atom
	molecules[p.nbead] = 0
	types[p.nbead] = 3

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
				if is_reserved_strand[strands[obi]]:
					type = 3
				else:
					type = 1
				atom1 = obi + ci*p.n_stap + 1
				atom2 = backbone_neighbors[obi][1] + ci*p.n_stap + 1
				bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### hybridization bonds
	if p.forceBind or p.startBound:
		for bi in range(p.n_scaf):
			if complements[bi] != -1:
				if not is_reserved_strand[strands[complements[bi]]]:
					type = 2
					atom1 = bi + 1
					atom2 = complements[bi] + 1
					bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### count angles
	nangle = 0
	for bi in range(p.n_scaf):
		bi_5p,bi_3p = getAssembledNeighbors(bi,backbone_neighbors,complements)
		if bi_5p != -1 and bi_3p != -1:
			if complements[bi_5p] != -1 and complements[bi] != -1 and complements[bi_3p] != -1:
				nangle += 1

	### write file
	outGeoFile = outSimFold + "geometry.in"
	ars.writeGeo(outGeoFile, p.dbox, r, molecules, types, bonds, nbondType=3, nangleType=2, charges=charges)

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
	for rbi in range(p.n_scaf,p.nbead):
		obi = rbi2obi(rbi, p)
		molecules[rbi] = strands[obi] + 1
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


### write lammps input file for lammps
def writeInput(outSimFold, is_crossover, nhyb, nangle, nreact_angleHyb, nreact_chargeDehyb, bridgeEnds, p):
	print("Writing input file...")

	### computational parameters
	verlet_skin				= 4		# nm			- width of neighbor list skin (= r12_cut - sigma)
	neigh_every				= 10	# steps			- how often to consider updating neighbor list
	bond_res 				= 0.1	# nm			- distance between tabular bond interpolation points
	F_forceBind				= 1		# kcal/mol/nm	- force to apply for forced binding
	r12_cut_react_hybBond	= 4		# nm			- cutoff radius for potential hybridization bonds
	react_every_bondHyb		= 1E2	# steps			- how often to check for new hybridization bonds
	react_every_angleHyb	= 1E4	# steps			- how often to check for new angles
	react_every_chargeDehyb	= 1E2	# steps			- how often to check for deactivating charges (makes eligible for angle and bond removal)
	react_every_angleDehyb	= 1E2	# steps			- how often to check for removing angles
	react_every_bondDehyb	= 1E2	# steps			- how often to check for removing hybridization bonds
	comm_cutoff				= 12	# nm			- communication cutoff (relevant for parallelization)
	U_barrier_comm			= 10	# kcal/mol		- energy barrier to exceeding communication cutoff

	### adjust comm_cutoff for forced binding
	if p.forceBind:
		comm_cutoff = int(np.sqrt(3)*p.dbox+1)

	### count digits
	len_nreact_angleHyb = len(str(nreact_angleHyb))
	len_nreact_chargeDehyb = len(str(nreact_chargeDehyb))

	### write table for hybridization bond
	npoint_bond = writeBondHyb(outSimFold, bond_res, F_forceBind, comm_cutoff, U_barrier_comm, p)

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
			"read_data       geometry.in &\n"
			"                extra/bond/per/atom 10 &\n"
			"                extra/angle/per/atom 10 &\n"
			"                extra/special/per/atom 100\n\n")

		### neighbor list
		f.write(
			"## Parameters\n"
		   f"neighbor        {verlet_skin:0.2f} bin\n"
		   f"neigh_modify    every {int(neigh_every)}\n"
		   f"pair_style      hybrid zero {p.r12_cut_WCA:0.2f} lj/cut {p.r12_cut_WCA:0.2f}\n"
			"pair_modify     pair lj/cut shift yes\n"
		   f"pair_coeff      * * lj/cut {p.epsilon:0.2f} {p.sigma:0.2f} {p.r12_cut_WCA:0.2f}\n"
		   f"pair_coeff      * 3 zero\n"
			"special_bonds   lj 0.0 1.0 1.0\n"
		   f"comm_modify     cutoff {comm_cutoff}\n")

		### bonded interactions
		f.write(
		   f"bond_style      hybrid zero harmonic table linear {npoint_bond}\n"
		   f"bond_coeff      1 harmonic {p.k_x/2:0.2f} {p.r12_eq:0.2f}\n"
		   f"bond_coeff      2 table bond_hyb.txt hyb\n"
			"bond_coeff      3 zero\n")

		### angled interactions
		if nangle: f.write(
		   f"angle_style     harmonic\n"
		   f"angle_coeff     1 {p.k_theta/2:0.2f} 180\n"
		   f"angle_coeff     2 {p.k_theta/2:0.2f} 90\n")

		### group atoms
		f.write(
			"variable        varQ atom q\n"
			"variable        varMolID atom mol\n"
		   f"variable        varStrand atom \"(id <= {p.n_scaf})*1.0 + (id > {p.n_scaf})*q\"\n"
			"compute         nhyb all nbond/atom bond/type 2\n"
			"variable        nhyb atom c_nhyb\n"
		   f"group           scaf id <= {p.n_scaf}\n"
		   f"group           real id <= {p.nbead}\n"
			"group           mobile type 1 2\n\n")

		### relax everything
		if p.nstep_relax > 0: f.write(
			"## Relaxation\n"
		   f"fix             tstat1 mobile langevin {p.T_relax} {p.T_relax} {1/p.gamma_t:0.4f} {p.rseed}\n"
			"fix             tstat2 mobile nve/limit 0.1\n"
		   f"timestep        {p.dt}\n"
		   f"run             {int(p.nstep_relax)}\n"
			"unfix           tstat1\n"
			"unfix           tstat2\n"
			"reset_timestep  0\n\n")

		#-------- molecule templates --------#

		### molecule template header
		f.write(
			"## Molecules\n")

		### bond templates
		if not p.forceBind: f.write(
		   f"molecule        hybBondMid_mol_bondNo react/hybBondMid_mol_bondNo.txt\n"
		   f"molecule        hybBondMid_mol_bondYa react/hybBondMid_mol_bondYa.txt\n")
		if not p.forceBind and not p.circularScaf: f.write(
		   f"molecule        hybBondEnd_mol_bondNo react/hybBondEnd_mol_bondNo.txt\n"
		   f"molecule        hybBondEnd_mol_bondYa react/hybBondEnd_mol_bondYa.txt\n")

		### bridge creating templates
		if bridgeEnds: f.write(
		   f"molecule        bridgeEnds_mol_bondNo react/bridgeEnds_mol_bondNo.txt\n"
		   f"molecule        bridgeEnds_mol_bondYa react/bridgeEnds_mol_bondYa.txt\n")

		### bridge breaking templates
		if bridgeEnds and p.dehyb: f.write(
		   f"molecule        unbridge1_mol_bondYa react/unbridge1_mol_bondYa.txt\n"
		   f"molecule        unbridge1_mol_bondNo react/unbridge1_mol_bondNo.txt\n"
		   f"molecule        unbridge2_mol_bondYa react/unbridge2_mol_bondYa.txt\n"
		   f"molecule        unbridge2_mol_bondNo react/unbridge2_mol_bondNo.txt\n")
		
		### angle hybridization templates
		for ri in range(nreact_angleHyb): f.write(
		   f"molecule        angleHyb{ri+1:0>{len_nreact_angleHyb}}_mol react/angleHyb{ri+1:0>{len_nreact_angleHyb}}_mol.txt\n")

		### angle dehybridization templates
		for ri in range(nreact_chargeDehyb): f.write(
		   f"molecule        chargeDehyb{ri+1:0>{len_nreact_chargeDehyb}}_mol react/chargeDehyb{ri+1:0>{len_nreact_chargeDehyb}}_mol.txt\n")
		f.write("\n")

		#-------- reactions --------#

		### reaction header
		f.write(
			"## Reactions\n"
			"fix             reactions all bond/react reset_mol_ids no")

		### bond hybridization reactions
		if not p.forceBind: f.write(
		   f" &\n                react bondHybMid all {int(react_every_bondHyb)} 0.0 {r12_cut_react_hybBond:.1f} hybBondMid_mol_bondNo hybBondMid_mol_bondYa react/hybBondMid_map.txt")
		if not p.forceBind and not p.circularScaf: f.write(
		   f" &\n                react bondHybEnd all {int(react_every_bondHyb)} 0.0 {r12_cut_react_hybBond:.1f} hybBondEnd_mol_bondNo hybBondEnd_mol_bondYa react/hybBondEnd_map.txt")
		
		### bridge making reaction
		if bridgeEnds: f.write(
		   f" &\n                react bridgeEnds all {int(react_every_bondHyb)} 0.0 {p.r12_eq:.2f} bridgeEnds_mol_bondNo bridgeEnds_mol_bondYa react/bridgeEnds_map.txt custom_charges 3")

		### bridge breaking reaction
		if bridgeEnds and p.dehyb: f.write(
		   f" &\n                react unbridgeEnds1 all {int(react_every_bondHyb)} {p.r12_cut_hyb:.1f} {comm_cutoff} unbridge1_mol_bondYa unbridge1_mol_bondNo react/unbridge1_map.txt custom_charges 4"
		   f" &\n                react unbridgeEnds2 all {int(react_every_bondHyb)} {p.r12_cut_hyb:.1f} {comm_cutoff} unbridge2_mol_bondYa unbridge2_mol_bondNo react/unbridge1_map.txt custom_charges 4")

		### angle hybridization reactions
		for ri in range(nreact_angleHyb): f.write(
		   f" &\n                react angleHyb{ri+1:0>{len_nreact_angleHyb}} all {int(react_every_angleHyb)} 0.0 {p.r12_cut_hyb:.1f} angleHyb{ri+1:0>{len_nreact_angleHyb}}_mol angleHyb{ri+1:0>{len_nreact_angleHyb}}_mol react/angleHyb{ri+1:0>{len_nreact_angleHyb}}_map.txt custom_charges 4")

		### charge dehybridization reactions
		for ri in range(nreact_chargeDehyb): f.write(
	  	   f" &\n                react chargeDehyb{ri+1:0>{len_nreact_chargeDehyb}} all {int(react_every_angleDehyb)} {p.r12_cut_hyb:.1f} {comm_cutoff} chargeDehyb{ri+1:0>{len_nreact_chargeDehyb}}_mol chargeDehyb{ri+1:0>{len_nreact_chargeDehyb}}_mol react/chargeDehyb{ri+1:0>{len_nreact_chargeDehyb}}_map.txt custom_charges 4")

		### bond dehybridization reactions
		if not p.forceBind and p.dehyb: f.write(
		   f" &\n                react bondDehybMid all {int(react_every_bondDehyb)} {r12_cut_react_hybBond:.1f} {comm_cutoff} hybBondMid_mol_bondYa hybBondMid_mol_bondNo react/hybBondMid_map.txt")
		if not p.forceBind and p.dehyb and not p.circularScaf: f.write(
		   f" &\n                react bondDehybEnd all {int(react_every_bondDehyb)} {r12_cut_react_hybBond:.1f} {comm_cutoff} hybBondEnd_mol_bondYa hybBondEnd_mol_bondNo react/hybBondEnd_map.txt")
		if not p.forceBind and p.dehyb: f.write(
		   f" &\n                react angleDehybMid all {int(react_every_angleDehyb)} {p.r12_cut_hyb:.1f} {r12_cut_react_hybBond:.1f} hybBondMid_mol_bondYa hybBondMid_mol_bondYa react/hybBondMid_map.txt")
		if not p.forceBind and p.dehyb and not p.circularScaf: f.write(
		   f" &\n                react angleDehybEnd all {int(react_every_angleDehyb)} {p.r12_cut_hyb:.1f} {r12_cut_react_hybBond:.1f} hybBondEnd_mol_bondYa hybBondEnd_mol_bondYa react/hybBondEnd_map.txt")
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

		### debugging output
		if p.debug:
			f.write(
			"## Debugging\n"
			"compute         compD1a all bond/local dist engpot\n"
			"compute         compD1b all property/local btype batom1 batom2\n"
		   f"dump            dumpD1 all local {int(p.dump_every)} dump_bonds.dat index c_compD1a[1] c_compD1a[2] c_compD1b[1] c_compD1b[2] c_compD1b[3] \n"
			"dump_modify     dumpD1 append yes\n"
			"compute         compD2a all angle/local theta eng\n"
			"compute         compD2b all property/local atype aatom1 aatom2 aatom3\n"
		   f"dump            dumpD2 all local {int(p.dump_every)} dump_angles.dat index c_compD2a[1] c_compD2a[2] c_compD2b[1] c_compD2b[2] c_compD2b[3] c_compD2b[4]\n"
			"dump_modify     dumpD2 append yes\n"
		   f"dump            dumpD3 scaf custom {int(p.dump_every)} dump_charges.dat id q\n"
			"dump_modify     dumpD3 sort id append yes\n\n")

		### production
		f.write(
			"## Production\n"
		   f"fix             tstat1 mobile langevin {p.T} {p.T} {1/p.gamma_t:0.4f} {p.rseed}\n"
			"fix             tstat2 mobile nve\n"
		   f"timestep        {p.dt}\n"
		   f"dump            dump1 real custom {int(p.dump_every)} trajectory.dat id v_varStrand xs ys zs\n"
			"dump_modify     dump1 sort id append yes\n"
		   f"restart         {int(p.dump_every/2)} restart_binary1.out restart_binary2.out\n\n")

		### run the simulation
		f.write(
			"## Go Time\n"
		   f"run             {int(p.nstep)}\n"
			"write_data      restart_geometry.out\n\n")


### write reaction files for hybridization bonds
def writeReactHybBond(outReactFold, backbone_neighbors, complements, p):
	print("Writing bond react files...")

	### no reactions necessary for forced binding
	if p.forceBind:
		return

	### initialize
	atoms_all = []
	bonds_all = []
	edges_all = []

	### two flanking scaffold beads
	atoms = [ [0,0], [1,1], [0,2], [0,3] ]
	bonds = [ [0,2,0], [0,0,3] ]
	edges = [ 1, 2, 3 ]
	atoms_all.append(atoms)
	bonds_all.append(bonds)
	edges_all.append(edges)

	### one flanking scaffold bead
	if not p.circularScaf:
		atoms = [ [0,0], [1,1], [0,2] ]
		bonds = [ [0,2,0] ]
		edges = [ 1, 2 ]
		atoms_all.append(atoms)
		bonds_all.append(bonds)
		edges_all.append(edges)

	### loop over reactions
	nreact = len(atoms_all)
	for ri in range(nreact):

		### reaction tag
		if ri == 0: tag = "Mid"
		if ri == 1: tag = "End"

		### scaffold
		if ri == 0: scaf = [0,2,3]
		if ri == 1: scaf = [0,2]

		atoms = atoms_all[ri]
		bonds = bonds_all[ri]
		edges = edges_all[ri]
		natom = len(atoms)
		nbond = len(bonds)
		nedge = len(edges)

		molFile = f"{outReactFold}hybBond{tag}_mol_bondNo.txt"
		with open(molFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")			
			f.write(f"{natom} fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

			f.write("\nFragments\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

		molFile = f"{outReactFold}hybBond{tag}_mol_bondYa.txt"
		with open(molFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond+1} bonds\n")			
			f.write(f"{natom} fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")
			f.write(f"{bondi+2}\t2\t1\t2\n")

			f.write("\nFragments\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

		mapFile = f"{outReactFold}hybBond{tag}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"{natom} constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write("1\n")
			f.write("2\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges[edgei]+1}\n")

			f.write("\nConstraints\n\n")
			f.write(f"custom \"rxnsum(v_varMolID,1) == rxnsum(v_varMolID,2)\"\n")
			for atomi in scaf:
				f.write(f"custom \"rxnsum(v_varQ,{atomi+1}) == 1 || rxnsum(v_varQ,{atomi+1}) == 2\"\n")

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")


### write reaction files for hybridization angles
def writeReactAngleHyb(outReactFold, backbone_neighbors, complements, is_crossover, p):
	print("Writing angle hybridization react files...")

	### initialize template list
	atoms_all = []
	bonds_all = []
	angls_all = []
	edges_all = []

	#-------- template loop --------#

	### loop over all beads
	for bi in range(p.n_scaf):

		### get neighbors to central bead
		bi_5p,bi_3p = getAssembledNeighbors(bi,backbone_neighbors,complements)

		### skip if core scaffold is not present
		if bi_5p == -1 or bi_3p == -1:
			continue

		### skip if core scaffold is not fully complimentary
		if complements[bi_5p] == -1 or complements[bi] == -1 or complements[bi_3p] == -1:
			continue

		### loop over configurational options (bridged ends, 5p staple copied, 3p staple copied)
		for options in itertools.product([True, False], repeat=3):

			#-------- working 5' to 3' --------#

			### intialize
			a = bi_5p
			b = bi
			c = bi_3p
			aC = complements[a]
			bC = complements[b]
			cC = complements[c]

			### core topology
			atoms_5to3 = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [1,-1,aC] ]
			bonds_5to3 = [ [0,a,b], [0,b,c], [1,a,aC], [1,b,bC], [1,c,cC] ]
			angls_5to3 = [ [int(is_crossover[b]),a,b,c,1] ]
			edges_5to3 = [ cC, aC ]

			### add scaffold 5' side topology
			a_5p = getAssembledNeighbors(a, backbone_neighbors, complements)[0]
			if a_5p != -1:
				if backbone_neighbors[a][0] != -1 or options[0]:
					atoms_5to3.append([0,-1,a_5p])
					bonds_5to3.append([0,a_5p,a])
					angls_5to3.append([int(is_crossover[a]),a_5p,a,b,0])
					edges_5to3.append(a_5p)

			### add scaffold 3' side topology
			c_3p = getAssembledNeighbors(c, backbone_neighbors, complements)[1]
			if c_3p != -1:
				if backbone_neighbors[c][1] != -1 or options[0]:
					atoms_5to3.append([0,-1,c_3p])
					bonds_5to3.append([0,c,c_3p])
					angls_5to3.append([int(is_crossover[c]),b,c,c_3p,2])
					edges_5to3.append(c_3p)

			### add central staple 5' side topology
			bC_5p = backbone_neighbors[bC][0]
			if bC_5p != -1:
				if bC_5p == cC:
					if p.stap_copies > 1 and options[1]:
						atoms_5to3.append([1,-1,bC_5p+p.n_stap])
						edges_5to3.append(bC_5p+p.n_stap)
						bonds_5to3.append([0,bC_5p+p.n_stap,bC])
					else:
						bonds_5to3.append([0,cC,bC])
				else:
					if [1,-1,bC_5p] not in atoms_5to3:
						atoms_5to3.append([1,-1,bC_5p])
						edges_5to3.append(bC_5p)
					bonds_5to3.append([0,bC_5p,bC])

			### add central staple 3' side topology
			bC_3p = backbone_neighbors[bC][1]
			if bC_3p != -1:
				if bC_3p == aC:
					if p.stap_copies > 1 and options[2]:
						atoms_5to3s.append([1,-1,bC_3p+p.n_stap])
						edges_5to3s.append(bC_3p+p.n_stap)
						bonds_5to3s.append([0,bC,bC_3p+p.n_stap])
					else:
						bonds_5to3.append([0,bC,aC])
				else:
					if [1,-1,bC_3p] not in atoms_5to3:
						atoms_5to3.append([1,-1,bC_3p])
						edges_5to3.append(bC_3p)
					bonds_5to3.append([0,bC,bC_3p])

			#-------- working 3' to 5' --------#

			### intialize
			a = bi_3p
			b = bi
			c = bi_5p
			aC = complements[a]
			bC = complements[b]
			cC = complements[c]

			### core topology
			atoms_3to5 = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [1,-1,aC] ]
			bonds_3to5 = [ [0,a,b], [0,b,c], [1,a,aC], [1,b,bC], [1,c,cC] ]
			angls_3to5 = [ [int(is_crossover[b]),a,b,c,1] ]
			edges_3to5 = [ cC, aC ]

			### add scaffold 3' side topology
			a_3p = getAssembledNeighbors(a, backbone_neighbors, complements)[1]
			if a_3p != -1:
				if backbone_neighbors[a][1] != -1 or options[0]:
					atoms_3to5.append([0,-1,a_3p])
					bonds_3to5.append([0,a_3p,a])
					angls_3to5.append([int(is_crossover[a]),a_3p,a,b,0])
					edges_3to5.append(a_3p)

			### add scaffold 5' side topology
			c_5p = getAssembledNeighbors(c, backbone_neighbors, complements)[0]
			if c_5p != -1:
				if backbone_neighbors[c][0] != -1 or options[0]:
					atoms_3to5.append([0,-1,c_5p])
					bonds_3to5.append([0,c,c_5p])
					angls_3to5.append([int(is_crossover[c]),b,c,c_5p,2])
					edges_3to5.append(c_5p)

			### add central staple 3' side topology
			bC_3p = backbone_neighbors[bC][1]
			if bC_3p != -1:
				if bC_3p == cC:
					if p.stap_copies > 1 and options[2]:
						atoms_3to5.append([1,-1,bC_3p+p.n_stap])
						edges_3to5.append(bC_3p+p.n_stap)
						bonds_3to5.append([0,bC_3p+p.n_stap,bC])
					else:
						bonds_3to5.append([0,cC,bC])
				else:
					if [1,-1,bC_3p] not in atoms_3to5:
						atoms_3to5.append([1,-1,bC_3p])
						edges_3to5.append(bC_3p)
					bonds_3to5.append([0,bC_3p,bC])

			### add central staple 5' side topology
			bC_5p = backbone_neighbors[bC][0]
			if bC_5p != -1:
				if bC_5p == aC:
					if p.stap_copies > 1 and options[1]:
						atoms_3to5.append([1,-1,bC_5p+p.n_stap])
						edges_3to5.append(bC_5p+p.n_stap)
						bonds_3to5.append([0,bC,bC_5p+p.n_stap])
					else:
						bonds_3to5.append([0,bC,aC])
				else:
					if [1,-1,bC_5p] not in atoms_3to5:
						atoms_3to5.append([1,-1,bC_5p])
						edges_3to5.append(bC_5p)
					bonds_3to5.append([0,bC,bC_5p])

			#-------- add template to list, if not duplicate --------#

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

			atoms_all.append(atoms_5to3)
			bonds_all.append(bonds_5to3)
			angls_all.append(angls_5to3)
			edges_all.append(edges_5to3)
			atoms_all.append(atoms_3to5)
			bonds_all.append(bonds_3to5)
			angls_all.append(angls_3to5)
			edges_all.append(edges_3to5)
			templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angls_all,edges_all)]
			templates = removeDuplicateElements(templates)
			if not symmetric:
				templates.pop()
			atoms_all,bonds_all,angls_all,edges_all = unzip4(templates)

			if len(angls_5to3) >= 2:
				angls_5to3_copy = copy.deepcopy(angls_5to3)
				angls_5to3_copy[1][0] += 2
				angls_3to5_copy = copy.deepcopy(angls_3to5)
				angls_3to5_copy[1][0] += 2
				atoms_5to3_copy = copy.deepcopy(atoms_5to3)
				atoms_5to3_copy[angls_5to3[1][4]][1] += 2
				atoms_3to5_copy = copy.deepcopy(atoms_3to5)
				atoms_3to5_copy[angls_3to5[1][4]][1] += 2
				atoms_all.append(atoms_5to3_copy)
				bonds_all.append(bonds_5to3)
				angls_all.append(angls_5to3_copy)
				edges_all.append(edges_5to3)
				atoms_all.append(atoms_3to5_copy)
				bonds_all.append(bonds_3to5)
				angls_all.append(angls_3to5_copy)
				edges_all.append(edges_3to5)
				templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angls_all,edges_all)]
				templates = removeDuplicateElements(templates)
				if not symmetric:
					templates.pop()
				atoms_all,bonds_all,angls_all,edges_all = unzip4(templates)

			if len(angls_5to3) >= 3:
				if not symmetric:
					angls_5to3_copy = copy.deepcopy(angls_5to3)
					angls_5to3_copy[2][0] += 2
					angls_3to5_copy = copy.deepcopy(angls_3to5)
					angls_3to5_copy[2][0] += 2
					atoms_5to3_copy = copy.deepcopy(atoms_5to3)
					atoms_5to3_copy[angls_5to3[2][4]][1] += 2
					atoms_3to5_copy = copy.deepcopy(atoms_3to5)
					atoms_3to5_copy[angls_3to5[2][4]][1] += 2
					atoms_all.append(atoms_5to3_copy)
					bonds_all.append(bonds_5to3)
					angls_all.append(angls_5to3_copy)
					edges_all.append(edges_5to3)
					atoms_all.append(atoms_3to5_copy)
					bonds_all.append(bonds_3to5)
					angls_all.append(angls_3to5_copy)
					edges_all.append(edges_3to5)
					templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angls_all,edges_all)]
					templates = removeDuplicateElements(templates)
					templates.pop()
					atoms_all,bonds_all,angls_all,edges_all = unzip4(templates)

				angls_5to3_copy = copy.deepcopy(angls_5to3)
				angls_5to3_copy[1][0] += 2
				angls_5to3_copy[2][0] += 2
				angls_3to5_copy = copy.deepcopy(angls_3to5)
				angls_3to5_copy[1][0] += 2
				angls_3to5_copy[2][0] += 2
				atoms_5to3_copy = copy.deepcopy(atoms_5to3)
				atoms_5to3_copy[angls_5to3[1][4]][1] += 2
				atoms_5to3_copy[angls_5to3[2][4]][1] += 2
				atoms_3to5_copy = copy.deepcopy(atoms_3to5)
				atoms_3to5_copy[angls_3to5[1][4]][1] += 2
				atoms_3to5_copy[angls_3to5[2][4]][1] += 2
				atoms_all.append(atoms_5to3_copy)
				bonds_all.append(bonds_5to3)
				angls_all.append(angls_5to3_copy)
				edges_all.append(edges_5to3)
				atoms_all.append(atoms_3to5_copy)
				bonds_all.append(bonds_3to5)
				angls_all.append(angls_3to5_copy)
				edges_all.append(edges_3to5)
				templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angls_all,edges_all)]
				templates = removeDuplicateElements(templates)
				if not symmetric:
					templates.pop()
				atoms_all,bonds_all,angls_all,edges_all = unzip4(templates)

	#-------- end template loop --------#

	### for nice debug output
	if p.debug: print()

	### loop over templates to write files
	nreact = len(atoms_all)
	len_nreact = len(str(nreact))
	for ri in range(nreact):

		atoms = atoms_all[ri]
		bonds = bonds_all[ri]
		angls = angls_all[ri]
		edges = edges_all[ri]
		natom = len(atoms)
		nbond = len(bonds)
		nangl = len(angls)
		nedge = len(edges)

		if p.debug:
			print(f"Angle template {ri+1}:")
			print(atoms)
			print(bonds)
			print(angls)
			print(edges)
			print()

		molFile = f"{outReactFold}angleHyb{ri+1:0>{len_nreact}}_mol.txt"
		with open(molFile, 'w') as f:

			atoms_copy = copy.deepcopy(atoms)
			angls_copy = copy.deepcopy(angls)
			atoms_copy[1][1] += 2
			angls_copy[0][0] += 2

			nangl_on = 0
			for anglei in range(nangl):
				if angls_copy[anglei][0] > 1:
					nangl_on += 1

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")
			f.write(f"{nangl_on} angles\n")
			f.write(f"4 fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms_copy[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

			f.write("\nAngles\n\n")
			angle_count = 1
			for anglei in range(nangl):
				if angls_copy[anglei][0] >= 2:
					angle_count += 1
					f.write(f"{angle_count}\t{angls_copy[anglei][0]-1}\t{angls_copy[anglei][1]+1}\t{angls_copy[anglei][2]+1}\t{angls_copy[anglei][3]+1}\n")
			
			f.write("\nCharges\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms_copy[atomi][1]+1}\n")

			f.write("\nFragments\n\n")
			f.write("1\t1\n")
			f.write("2\t2\n")
			f.write("3\t3\n")
			f.write("4\t1 2 3\n")

		mapFile = f"{outReactFold}angleHyb{ri+1:0>{len_nreact}}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"{nangl+2} constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write("2\n")
			f.write("5\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges[edgei]+1}\n")
			
			f.write("\nConstraints\n\n")
			f.write(f"custom \"round(rxnsum(v_varQ,{angls[0][4]+1})) == {angls[0][0]+1}\"\n")
			if nangl >= 2:
				f.write(f"custom \"round(rxnsum(v_varQ,{angls[1][4]+1})) == {angls[1][0]+1}\"\n")
			if nangl >= 3:
				f.write(f"custom \"round(rxnsum(v_varQ,{angls[2][4]+1})) == {angls[2][0]+1}\"\n")
			f.write("distance 1 6 0.0 2.0\n")
			f.write("distance 3 4 0.0 2.0\n")

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

	### return reaction count
	return nreact


### write reaction files that update charges after dehybridizations
def writeReactChargeDehyb(outReactFold, backbone_neighbors, complements, is_crossover, p):
	print("Writing angle dehybridization react files...")

	### determine if scaffold ends (if they exist) are bridged
	bridgeEnds = False
	if not p.circularScaf and getAssembledNeighbors(0,backbone_neighbors,complements)[0] != -1:
		bridgeEnds = True
	nconstraint = 5 if bridgeEnds else 3

	### only necessary if including dehybridization
	if not p.dehyb:

		### warning for end bridging
		if bridgeEnds:
			print("Flag: Scaffold ends are bridged, but dehybridization is is off, so end bridging is permenant.")

		### no reactions
		return 0, bridgeEnds

	### initialize templates
	atoms_all = []
	bonds_all = []
	edges_all = []
	extra_all = []

	### extra array description
	# [ initiator scaffold, initiator staple, whether off state is allowed (for each angle), whether on state is allowed (for each angle) ]

	### core topology - three staples, two staple backbone bonds
	atoms = [ [0,0,0], [0,0,1], [0,0,2], [1,-1,3], [1,-1,4], [1,-1,5], [0,-1,6], [0,-1,7] ]
	bonds = [ [0,0,1], [0,1,2], [1,0,5], [1,1,4], [1,2,3], [0,6,0], [0,2,7], [0,3,4], [0,4,5] ]
	edges = [ 3, 4, 5, 6, 7 ]
	extra = [ 1, 4, [True,False,True], [True,True,True] ]

	### all 180
	if checkTemplateMatch(True, True, True, True, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 1")
		atoms_all.append(copy.deepcopy(atoms))
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### side 90
	if checkTemplateMatch(True, True, True, False, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 2")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### side, middle 90
	if checkTemplateMatch(True, True, True, False, False, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 3")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[1][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### side, side 90
	if checkTemplateMatch(True, True, True, False, True, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 4")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### all 90
	if checkTemplateMatch(True, True, True, False, False, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 5")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[1][1] = 1
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### core topology - three staples, one staple backbone bond
	atoms = [ [0,0,0], [0,0,1], [0,0,2], [1,-1,3], [1,-1,4], [1,-1,5], [0,-1,6], [0,-1,7] ]
	bonds = [ [0,0,1], [0,1,2], [1,0,5], [1,1,4], [1,2,3], [0,6,0], [0,2,7], [0,3,4] ]
	edges = [ 3, 4, 5, 6, 7 ]
	extra = [ 1, 4, [True,False,True], [True,True,True] ]

	### all 180
	if checkTemplateMatch(True, True, False, True, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 6")
		atoms_all.append(copy.deepcopy(atoms))
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### left 90
	if checkTemplateMatch(True, True, False, False, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 7")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### right 90
	if checkTemplateMatch(True, True, False, True, True, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 8")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### left, middle 90
	if checkTemplateMatch(True, True, False, False, False, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 9")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[1][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### middle, right 90
	if checkTemplateMatch(True, True, False, True, False, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 10")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[1][1] = 1
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### left, right 90
	if checkTemplateMatch(True, True, False, False, True, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 11")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### all 90
	if checkTemplateMatch(True, True, False, False, False, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 12")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[1][1] = 1
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### core topology - three staples, no staple backbone bonds
	atoms = [ [0,0,0], [0,0,1], [0,0,2], [1,-1,3], [1,-1,4], [1,-1,5], [0,-1,6], [0,-1,7] ]
	bonds = [ [0,0,1], [0,1,2], [1,0,5], [1,1,4], [1,2,3], [0,6,0], [0,2,7] ]
	edges = [ 3, 4, 5, 6, 7 ]
	extra = [ 1, 4, [True,False,True], [True,True,True] ]

	### all 180
	if checkTemplateMatch(True, False, False, True, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 13")
		atoms_all.append(copy.deepcopy(atoms))
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### side 90
	if checkTemplateMatch(True, False, False, False, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 14")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### side, middle 90
	if checkTemplateMatch(True, False, False, False, False, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 15")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[1][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### side, side 90
	if checkTemplateMatch(True, False, False, False, True, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 16")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### all 90
	if checkTemplateMatch(True, False, False, False, False, False, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 17")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_copy[1][1] = 1
		atoms_copy[2][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### core topology - two staples, one staple backbone bond
	atoms = [ [0,0,0], [0,0,1], [0,0,2], [1,-1,3], [1,-1,4], [0,-1,5], [0,-1,6] ]
	bonds = [ [0,0,1], [0,1,2], [1,1,4], [1,2,3], [0,5,0], [0,2,6], [0,3,4] ]
	edges = [ 3, 4, 5, 6 ]
	extra = [ 1, 4, [True,True,False], [False,False,True] ]

	### left 180
	if checkTemplateMatch(False, False, True, True, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 18")
		atoms_all.append(copy.deepcopy(atoms))
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### left 90
	if checkTemplateMatch(False, False, True, False, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 19")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### core topology - two staples, no staple backbone bonds
	atoms = [ [0,0,0], [0,0,1], [0,0,2], [1,-1,3], [1,-1,4], [0,-1,5], [0,-1,6] ]
	bonds = [ [0,0,1], [0,1,2], [1,0,4], [1,1,3], [0,5,0], [0,2,6] ]
	edges = [ 3, 4, 5, 6 ]
	extra = [ 1, 4, [True,True,False], [False,False,True] ]

	### left 180
	if checkTemplateMatch(False, False, False, True, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 20")
		atoms_all.append(copy.deepcopy(atoms))
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### left 90
	if checkTemplateMatch(False, False, False, False, True, True, backbone_neighbors, complements, is_crossover, p):
		if p.debug: print("Used charge dehybridization template 21")
		atoms_copy = copy.deepcopy(atoms)
		atoms_copy[0][1] = 1
		atoms_all.append(atoms_copy)
		bonds_all.append(copy.deepcopy(bonds))
		edges_all.append(copy.deepcopy(edges))
		extra_all.append(copy.deepcopy(extra))

	### count circular scaffold reactions
	nreact_circular = len(atoms_all)

	### for nice debug output
	if p.debug: print()

	### linear scaffold additions
	if not p.circularScaf:

		#-------- 5' end, working 5' to 3' --------#

		### core beads
		a = 0
		b = 1
		c = 2
		aC = complements[a]
		bC = complements[b]
		cC = complements[c]
		c_3p = 3

		### fully complimentary
		if aC != -1 and bC != -1 and cC != -1:

			### dehybridization of bead next to end
			atoms = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [1,-1,aC], [0,-1,c_3p] ]
			bonds = [ [0,a,b], [0,b,c], [1,a,aC], [1,b,bC], [1,c,cC], [0,c,c_3p] ]
			edges = [ cC, bC, aC, c_3p ]
			extra = [ b, bC, [True,False,True], [False,True,True] ]

			### this does not account for multiple staple copies
			if backbone_neighbors[bC][0] == cC:
				bonds.append([0,cC,bC])
			if backbone_neighbors[bC][1] == aC:
				bonds.append([0,bC,aC])

			atoms,bonds,edges,extra = renumberAtoms_chargeTemplate(atoms,bonds,edges,extra)
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))
			extra_all.append(copy.deepcopy(extra))

			### dehybridization of end bead
			atoms = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [1,-1,aC], [0,-1,c_3p] ]
			bonds = [ [0,a,b], [0,b,c], [1,a,aC], [1,b,bC], [1,c,cC], [0,c,c_3p] ]
			edges = [ cC, bC, aC, c_3p ]
			extra = [ a, aC, [True,False,False], [False,True,False] ]

			### this does not account for multiple staple copies
			if backbone_neighbors[bC][0] == cC:
				bonds.append([0,cC,bC])
			if backbone_neighbors[bC][1] == aC:
				bonds.append([0,bC,aC])

			atoms,bonds,edges,extra = renumberAtoms_chargeTemplate(atoms,bonds,edges,extra)
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))
			extra_all.append(copy.deepcopy(extra))

		### non-complimentary end bead
		if  aC == -1 and bC != -1 and cC != -1:

			### dehybridization of bead next to end
			atoms = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [0,-1,c_3p] ]
			bonds = [ [0,a,b], [0,b,c], [1,b,bC], [1,c,cC], [0,c,c_3p] ]
			edges = [ cC, bC, c_3p ]
			extra = [ b, bC, [True,True,False], [False,False,True] ]

			### this does not account for multiple staple copies
			if backbone_neighbors[bC][0] == cC:
				bonds.append([0,cC,bC])

			atoms,bonds,edges,extra = renumberAtoms_chargeTemplate(atoms,bonds,edges,extra)
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))
			extra_all.append(copy.deepcopy(extra))

		#-------- 3' scaffold end --------#

		### core beads
		a = p.n_scaf-1
		b = p.n_scaf-2
		c = p.n_scaf-3
		aC = complements[a]
		bC = complements[b]
		cC = complements[c]
		c_5p = p.n_scaf-4

		### fully complimentary
		if aC != -1 and bC != -1 and cC != -1:

			### dehybridization of bead next to end
			atoms = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [1,-1,aC], [0,-1,c_5p] ]
			bonds = [ [0,a,b], [0,b,c], [1,a,aC], [1,b,bC], [1,c,cC], [0,c,c_5p] ]
			edges = [ cC, bC, aC, c_5p ]
			extra = [ b, bC, [True,False,True], [False,True,True] ]

			### this does not account for multiple staple copies
			if backbone_neighbors[bC][0] == cC:
				bonds.append([0,cC,bC])
			if backbone_neighbors[bC][1] == aC:
				bonds.append([0,bC,aC])

			atoms,bonds,edges,extra = renumberAtoms_chargeTemplate(atoms,bonds,edges,extra)
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))
			extra_all.append(copy.deepcopy(extra))

			### dehybridization of end bead
			atoms = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [1,-1,aC], [0,-1,c_5p] ]
			bonds = [ [0,a,b], [0,b,c], [1,a,aC], [1,b,bC], [1,c,cC], [0,c,c_5p] ]
			edges = [ cC, bC, aC, c_5p ]
			extra = [ a, aC, [True,False,False], [False,True,False] ]

			if backbone_neighbors[bC][0] == cC:
				bonds.append([0,cC,bC])
			if backbone_neighbors[bC][1] == aC:
				bonds.append([0,bC,aC])

			atoms,bonds,edges,extra = renumberAtoms_chargeTemplate(atoms,bonds,edges,extra)
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))
			extra_all.append(copy.deepcopy(extra))

		### non-complimentary end bead
		if  aC == -1 and bC != -1 and cC != -1:

			### dehybridization of bead next to end
			atoms = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [0,-1,c_5p] ]
			bonds = [ [0,a,b], [0,b,c], [1,b,bC], [1,c,cC], [0,c,c_5p] ]
			edges = [ cC, bC, c_5p ]
			extra = [ b, bC, [True,True,False], [False,False,True] ]

			### this does not account for multiple staple copies
			if backbone_neighbors[bC][0] == cC:
				bonds.append([0,cC,bC])

			atoms,bonds,edges,extra = renumberAtoms_chargeTemplate(atoms,bonds,edges,extra)
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))
			extra_all.append(copy.deepcopy(extra))

		#-------- remove duplicates --------#

		### remove possible duplicates from linear scaffold analysis
		templates = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,edges_all,extra_all)]
		templates = removeDuplicateElements(templates)
		atoms_all,bonds_all,edges_all,extra_all = unzip4(templates)

		#-------- end linear scaffold templates --------#

	### loop over reactions
	nreact = len(atoms_all)
	len_nreact = len(str(nreact))
	for ri in range(nreact):

		atoms = atoms_all[ri]
		bonds = bonds_all[ri]
		edges = edges_all[ri]
		extra = extra_all[ri]
		natom = len(atoms)
		nbond = len(bonds)
		nedge = len(edges)

		nconstraint = 0
		for i in range(3):
			if extra[2][i] or extra[3][i]:
				nconstraint += 1
		if bridgeEnds and ri < nreact_circular:
			nconstraint += 2

		molFile = f"{outReactFold}chargeDehyb{ri+1:0>{len_nreact}}_mol.txt"
		with open(molFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")			
			f.write(f"1 angles\n")			
			f.write(f"4 fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

			f.write("\nAngles\n\n")
			f.write(f"1\t1\t3\t4\t5\n")

			f.write("\nCharges\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

			f.write("\nFragments\n\n")
			f.write("1\t1\n")
			f.write("2\t2\n")
			f.write("3\t3\n")
			f.write("4\t1 2 3\n")

		mapFile = f"{outReactFold}chargeDehyb{ri+1:0>{len_nreact}}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"{nconstraint} constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write(f"{extra[0]+1}\n")
			f.write(f"{extra[1]+1}\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges[edgei]+1}\n")

			### constraints
			f.write("\nConstraints\n\n")

			### constraint on atom 1
			if extra[2][0] or extra[3][0]:
				f.write(f"custom ")
				if extra[2][0]:
					f.write(f"\"round(rxnsum(v_varQ,1)) == {atoms[0][1]+1}\"")
					if extra[3][0]:
						f.write(" || ")
				if extra[3][0]:
					f.write(f"\"round(rxnsum(v_varQ,1)) == {atoms[0][1]+3}\"")
				f.write(f"\n")

			### constraint on atom 2
			if extra[2][1] or extra[3][1]:
				f.write(f"custom ")
				if extra[2][1]:
					f.write(f"\"round(rxnsum(v_varQ,2)) == {atoms[1][1]+1}\"")
					if extra[3][1]:
						f.write(" || ")
				if extra[3][1]:
					f.write(f"\"round(rxnsum(v_varQ,2)) == {atoms[1][1]+3}\"")
				f.write(f"\n")

			### constraint on atom 3
			if extra[2][2] or extra[3][2]:
				f.write(f"custom ")
				if extra[2][2]:
					f.write(f"\"round(rxnsum(v_varQ,3)) == {atoms[2][1]+1}\"")
					if extra[3][2]:
						f.write(" || ")
				if extra[3][2]:
					f.write(f"\"round(rxnsum(v_varQ,3)) == {atoms[2][1]+3}\"")
				f.write(f"\n")

			### avoid ends when appropriate
			if bridgeEnds and ri < nreact_circular:
				f.write(f"custom \"round(rxnsum(v_varMolID,{extra[0]+1})) != {p.n_scaf}\"\n")
				f.write(f"custom \"round(rxnsum(v_varMolID,{extra[0]+1})) != 1\"\n")
		
			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

	### if bridging ends
	if bridgeEnds:

		### initialize templates
		atoms_all = []
		bonds_all = []
		edges_all = []
		extra_all = []

		#-------- center as 5' end, working 3' to 5'  --------#

		### core beads
		a = 1
		b = 0
		c = p.n_scaf-1
		aC = complements[a]
		bC = complements[b]
		cC = complements[c]
		a_3p = 2
		c_5p = p.n_scaf-2

		### core topology
		atoms = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [0,-1,a_3p], [0,-1,c_5p] ]
		bonds = [ [0,a,b], [0,b,c], [1,b,bC], [1,c,cC], [0,cC,bC], [0,a_3p,a], [0,c,c_5p] ]
		edges = [ cC, a_3p, c_5p ]
		extra = [ b, bC, [True, True, True], [True, True, True] ]

		### scaffold 3' side compliment
		if aC != 0:
			atoms.append([1,-1,aC])
			bonds.append([1,a,aC])
			edges.append(aC)

		### central staple 5' side bond
		bC_5p = backbone_neighbors[bC][0]
		if bC_5p == -1:
			atoms,bonds,edges,extra = renumberAtoms_chargeTemplate(atoms,bonds,edges,extra)
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))
			extra_all.append(copy.deepcopy(extra))

		else:
			if bC_5p == aC:
				bonds_copy = copy.deepcopy(bonds)
				bonds_copy.append([0,bC,aC])

				atoms_copy,bonds_copy,edges_copy,extra = renumberAtoms_chargeTemplate(atoms_copy,bonds_copy,edges_copy,extra)
				atoms_all.append(copy.deepcopy(atoms))
				bonds_all.append(bonds_copy)
				edges_all.append(copy.deepcopy(edges))
				extra_all.append(copy.deepcopy(extra))

			if bC_5p != aC or p.stap_copies > 1:
				atoms_copy = copy.deepcopy(atoms)
				bonds_copy = copy.deepcopy(bonds)
				edges_copy = copy.deepcopy(edges)
				atoms_copy.append([1,-1,bC_5p])
				bonds_copy.append([0,bC,bC_5p])
				edges_copy.append(bC_5p)

				atoms_copy,bonds_copy,edges_copy,extra = renumberAtoms_chargeTemplate(atoms_copy,bonds_copy,edges_copy,extra)
				atoms_all.append(atoms_copy)
				bonds_all.append(bonds_copy)
				edges_all.append(edges_copy)
				extra_all.append(copy.deepcopy(extra))


		#-------- center as 3' end, working 5' to 3' --------#

		### core beads
		a = p.n_scaf-2
		b = p.n_scaf-1
		c = 0
		aC = complements[a]
		bC = complements[b]
		cC = complements[c]
		a_5p = p.n_scaf-3
		c_3p = 1

		### core topology
		atoms = [ [0,int(is_crossover[a]),a], [0,int(is_crossover[b]),b], [0,int(is_crossover[c]),c], [1,-1,cC], [1,-1,bC], [0,-1,a_5p], [0,-1,c_3p] ]
		bonds = [ [0,a,b], [0,b,c], [1,b,bC], [1,c,cC], [0,cC,bC], [0,a_5p,a], [0,c,c_3p] ]
		edges = [ bC, cC, a_5p, c_3p ]
		extra = [ b, bC, [True, True, True], [True, True, True] ]

		### scaffold 5' side compliment
		if aC != 0:
			atoms.append([1,-1,aC])
			bonds.append([1,a,aC])
			edges.append(aC)

		### central staple 3' side bond
		bC_3p = backbone_neighbors[bC][0]
		if bC_3p == -1:
			atoms,bonds,edges,extra = renumberAtoms_chargeTemplate(atoms,bonds,edges,extra)
			atoms_all.append(copy.deepcopy(atoms))
			bonds_all.append(copy.deepcopy(bonds))
			edges_all.append(copy.deepcopy(edges))
			extra_all.append(copy.deepcopy(extra))

		else:
			if bC_3p == aC:
				bonds_copy = copy.deepcopy(bonds)
				bonds_copy.append([0,bC,aC])

				atoms_copy,bonds_copy,edges_copy,extra = renumberAtoms_chargeTemplate(atoms_copy,bonds_copy,edges_copy,extra)
				atoms_all.append(copy.deepcopy(atoms))
				bonds_all.append(bonds_copy)
				edges_all.append(copy.deepcopy(edges))
				extra_all.append(copy.deepcopy(extra))

			if bC_3p != aC or p.stap_copies > 1:
				atoms_copy = copy.deepcopy(atoms)
				bonds_copy = copy.deepcopy(bonds)
				edges_copy = copy.deepcopy(edges)
				atoms_copy.append([1,-1,bC_3p])
				bonds_copy.append([0,bC,bC_3p])
				edges_copy.append(bC_3p)

				atoms_copy,bonds_copy,edges_copy,extra = renumberAtoms_chargeTemplate(atoms_copy,bonds_copy,edges_copy,extra)
				atoms_all.append(atoms_copy)
				bonds_all.append(bonds_copy)
				edges_all.append(edges_copy)
				extra_all.append(copy.deepcopy(extra))

		#-------- end bridge templates --------#

		### count reactions
		nreact = len(atoms_all)

		### loop over both reactions
		for ri in range(nreact):

			atoms = atoms_all[ri]
			bonds = bonds_all[ri]
			edges = edges_all[ri]
			extra = extra_all[ri]
			natom = len(atoms)
			nbond = len(bonds)
			nedge = len(edges)

			molFile = f"{outReactFold}unbridge{ri+1}_mol_bondYa.txt"
			with open(molFile, 'w') as f:

				f.write("## Hybridization\n")
				f.write(f"{natom} atoms\n")
				f.write(f"{nbond} bonds\n")			
				f.write(f"1 angles\n")		
				f.write(f"4 fragments\n")

				f.write("\nTypes\n\n")
				for atomi in range(natom):
					f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

				f.write("\nBonds\n\n")
				for bondi in range(nbond):
					f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

				f.write("\nAngles\n\n")
				f.write(f"1\t1\t3\t4\t5\n")

				f.write("\nCharges\n\n")
				for atomi in range(natom):
					f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

				f.write("\nFragments\n\n")
				f.write("1\t1\n")
				f.write("2\t2\n")
				f.write("3\t3\n")
				f.write("4\t1 2 3\n")

			molFile = f"{outReactFold}unbridge{ri+1}_mol_bondNo.txt"
			with open(molFile, 'w') as f:

				f.write("## Hybridization\n")
				f.write(f"{natom} atoms\n")
				f.write(f"{nbond-1} bonds\n")			
				f.write(f"1 angles\n")		
				f.write(f"4 fragments\n")

				f.write("\nTypes\n\n")
				for atomi in range(natom):
					f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

				f.write("\nBonds\n\n")
				for bondi in range(nbond):
					if bondi != 1:
						f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

				f.write("\nAngles\n\n")
				f.write(f"1\t1\t3\t4\t5\n")

				f.write("\nCharges\n\n")
				for atomi in range(natom):
					f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

				f.write("\nFragments\n\n")
				f.write("1\t1\n")
				f.write("2\t2\n")
				f.write("3\t3\n")
				f.write("4\t1 2 3\n")

			mapFile = f"{outReactFold}unbridge{ri+1}_map.txt"
			with open(mapFile, 'w') as f:

				f.write("## Hybridization\n")
				f.write(f"{natom} equivalences\n")
				f.write(f"{nedge} edgeIDs\n")
				f.write(f"5 constraints\n")

				f.write(f"\nInitiatorIDs\n\n")
				f.write(f"{extra[0]+1}\n")
				f.write(f"{extra[1]+1}\n")

				f.write(f"\nEdgeIDs\n\n")
				for edgei in range(nedge):
					f.write(f"{edges[edgei]+1}\n")

				### constraints
				f.write("\nConstraints\n\n")

				### constraint on atom 1
				f.write(f"custom ")
				if extra[2][0]:
					f.write(f"\"round(rxnsum(v_varQ,1)) == {atoms[0][1]+1}\"")
					if extra[3][0]:
						f.write(" || ")
				if extra[3][0]:
					f.write(f"\"round(rxnsum(v_varQ,1)) == {atoms[0][1]+3}\"")
				f.write(f"\n")

				### constraint on atom 2
				f.write(f"custom ")
				if extra[2][1]:
					f.write(f"\"round(rxnsum(v_varQ,2)) == {atoms[1][1]+1}\"")
					if extra[3][1]:
						f.write(" || ")
				if extra[3][1]:
					f.write(f"\"round(rxnsum(v_varQ,2)) == {atoms[1][1]+3}\"")
				f.write(f"\n")

				### constraint on atom 3
				f.write(f"custom ")
				if extra[2][2]:
					f.write(f"\"round(rxnsum(v_varQ,3)) == {atoms[2][1]+1}\"")
					if extra[3][2]:
						f.write(" || ")
				if extra[3][2]:
					f.write(f"\"round(rxnsum(v_varQ,3)) == {atoms[2][1]+3}\"")
				f.write(f"\n")

				### ensure correct location
				f.write(f"custom \"round(rxnsum(v_varMolID,2)) == 1 || "
								 f"round(rxnsum(v_varMolID,2)) == {p.n_scaf}\"\n")
				f.write(f"custom \"round(rxnsum(v_varMolID,3)) == 1 || "
								 f"round(rxnsum(v_varMolID,3)) == {p.n_scaf}\"\n")
			
				f.write("\nEquivalences\n\n")
				for atomi in range(natom):
					f.write(f"{atomi+1}\t{atomi+1}\n")

	### return reaction count, end bridging boolean
	return nreact, bridgeEnds


### write reaction files that connect bridged scaffold ends
def writeReactBridgeBond(outReactFold, backbone_neighbors, complements, is_crossover, p):

	### core beads
	a = p.n_scaf-1
	b = 0
	aC = complements[a]
	a_5p = p.n_scaf-2

	### core topology
	atoms = [ [0,-1,a], [0,-1,b], [1,-1,aC], [0,int(is_crossover[a_5p]),a_5p] ]
	bonds = [ [1,a,aC], [0,a_5p,a] ]
	edges = [ b, aC, a_5p ]

	### renumber and count
	atoms,bonds,edges = renumberAtoms_bridgeTemplate(atoms,bonds,edges)
	natom = len(atoms)
	nbond = len(bonds)
	nedge = len(edges)

	### write files
	molFile = f"{outReactFold}bridgeEnds_mol_bondNo.txt"
	with open(molFile, 'w') as f:

		f.write("## Hybridization\n")
		f.write(f"{natom} atoms\n")
		f.write(f"{nbond} bonds\n")			
		f.write(f"3 fragments\n")

		f.write("\nTypes\n\n")
		for atomi in range(natom):
			f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

		f.write("\nBonds\n\n")
		for bondi in range(nbond):
			f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

		f.write("\nCharges\n\n")
		for atomi in range(natom):
			f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

		f.write("\nFragments\n\n")
		f.write("1\t1\n")
		f.write("2\t2\n")
		f.write("3\t4\n")

	molFile = f"{outReactFold}bridgeEnds_mol_bondYa.txt"
	with open(molFile, 'w') as f:

		f.write("## Hybridization\n")
		f.write(f"{natom} atoms\n")
		f.write(f"{nbond+1} bonds\n")			
		f.write(f"3 fragments\n")

		f.write("\nTypes\n\n")
		for atomi in range(natom):
			f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

		f.write("\nBonds\n\n")
		for bondi in range(nbond):
			f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")
		f.write(f"{bondi+2}\t1\t1\t2\n")

		f.write("\nCharges\n\n")
		for atomi in range(natom):
			f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

		f.write("\nFragments\n\n")
		f.write("1\t1\n")
		f.write("2\t2\n")
		f.write("3\t4\n")

	mapFile = f"{outReactFold}bridgeEnds_map.txt"
	with open(mapFile, 'w') as f:

		f.write("## Hybridization\n")
		f.write(f"{natom} equivalences\n")
		f.write(f"{nedge} edgeIDs\n")
		f.write(f"3 constraints\n")

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
		f.write(f"custom \"round(rxnsum(v_varMolID,1)) == {p.n_scaf}\"\n")
		f.write(f"custom \"round(rxnsum(v_varMolID,2)) == 1\"\n")
		f.write(f"custom \"round(rxnsum(v_nhyb,2)) == 1\"\n")


### write table for hybridization bond
def writeBondHyb(outSimFold, bond_res, F_forceBind, comm_cutoff, U_barrier_comm, p):
	bondFile = outSimFold + "bond_hyb.txt"
	npoint = int(comm_cutoff/bond_res+1)

	### write file
	with open(bondFile, 'w') as f:
		f.write(f"hyb\n")
		f.write(f"N {npoint}\n\n")
		f.write("# r E(r) F(r)\n")
		
		### loop over points
		for i in range(npoint):
			r12 = i * bond_res
			if r12 < p.r12_cut_hyb:
				U = p.U_hyb*(r12/p.r12_cut_hyb-1)
				F = -p.U_hyb/p.r12_cut_hyb
			elif p.forceBind:
				U = 6.96*F_forceBind*(r12-p.r12_cut_hyb)
				F = -6.96*F_forceBind
			elif r12 > comm_cutoff - 2:
				U = 6.96*U_barrier_comm*((r12-(comm_cutoff-2))/2)**2
				F = -6.96*U_barrier_comm*(r12-(comm_cutoff-2))
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
				print("Error: Too dense to initiate swirl.")
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
				r_propose = ars.applyPBC(r[rbi-1] + p.r12_eq*ars.unitVector(ars.boxMuller()), p.dbox)

			### random position for new strand
			else:
				r_propose = ars.randPos(p.dbox)

			### evaluate position, break loop if no overlap and (for scaffold) within scaffold box
			if not ars.checkOverlap(r_propose,r[:rbi],p.sigma,p.dbox):
				if strands[obi] > 0:
					break
				elif all(abs(r_propose)<dbox_scaf/2):
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
				print("Error: could not place beads, try again with larger box.")
				sys.exit()

	### return positions
	return r


################################################################################
### Utilify Functions

### check if template exists anywhere in origami (not including bridging bonds)
def checkTemplateMatch(leftStap, leftStapBond, rightStapBond, left180, mid180, right180, backbone_neighbors, complements, is_crossover, p):
	for bi in range(p.n_scaf):

		### check input:
		if leftStapBond and not leftStap: 
			print("Error: Cannot have left staple bond without left staple.")
			sys.exit()

		### check for core scaffold
		bi_5p = backbone_neighbors[bi][0]
		bi_3p = backbone_neighbors[bi][1]
		if bi_5p == -1 or bi_3p == -1:
			continue

		### check for flanking scaffold
		bi_5p5p = backbone_neighbors[bi_5p][0]
		bi_3p3p = backbone_neighbors[bi_3p][1]
		if bi_5p5p == -1 or bi_3p3p == -1:
			continue

		#-------- working 5' to 3' --------#
		pass_5p = True

		### get core scaffold
		a = bi_5p
		b = bi
		c = bi_3p

		### check for core staples
		if complements[b] == -1 or complements[c] == -1:
			pass_5p = False

		### check for left stap
		if leftStap == (complements[a] == -1):
			pass_5p = False

		### get core staples
		aC = complements[a]
		bC = complements[b]
		cC = complements[c]

		### check left staple backbone bond
		if leftStapBond:
			if backbone_neighbors[bC][1] != aC:
				pass_5p = False
		elif leftStap:
			if p.stap_copies == 1 and backbone_neighbors[bC][1] == aC:
				pass_5p = False

		### check right staple backbone bond
		if rightStapBond:
			if backbone_neighbors[bC][0] != cC:
				pass_5p = False
		else:
			if p.stap_copies == 1 and backbone_neighbors[bC][0] == cC:
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
		if complements[b] == -1 or complements[c] == -1:
			pass_3p = False

		### check for left stap
		if leftStap == (complements[a] == -1):
			pass_3p = False

		### get core staples
		aC = complements[a]
		bC = complements[b]
		cC = complements[c]

		### check left staple backbone bond
		if leftStapBond:
			if backbone_neighbors[bC][0] != aC:
				pass_3p = False
		elif leftStap:
			if p.stap_copies == 1 and backbone_neighbors[bC][0] == aC:
				pass_3p = False

		### check right staple backbone bond
		if rightStapBond:
			if backbone_neighbors[bC][1] != cC:
				pass_3p = False
		else:
			if p.stap_copies == 1 and backbone_neighbors[bC][1] == cC:
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


### make duplicate of template and add it to template array
def branchTemplate(atoms,bonds,angls,edges):
	for ti in range(len(atoms)):
		atoms.append(copy.deepcopy(atoms[ti]))
		bonds.append(copy.deepcopy(bonds[ti]))
		angls.append(copy.deepcopy(angls[ti]))
		edges.append(copy.deepcopy(edges[ti]))
	return atoms,bonds,angls,edges


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


### renumber atoms (starting from 0) in charge template starting
def renumberAtoms_chargeTemplate(atoms, bonds, edges, extra):
	atoms_bi = [row[2] for row in atoms]
	for atomi in range(len(atoms)):
		atoms[atomi][2] = atomi
	for bondi in range(len(bonds)):
		for i in range(1,3):
			bonds[bondi][i] = atoms_bi.index(bonds[bondi][i])
	for edgei in range(len(edges)):
		edges[edgei] = atoms_bi.index(edges[edgei])
	for extrai in range(2):
		extra[extrai] = atoms_bi.index(extra[extrai])
	return atoms, bonds, edges, extra


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
def getAssembledNeighbors(bi, backbone_neighbors, complements):

	### for vast majority of cases, this is the result
	bi_5p = backbone_neighbors[bi][0]
	bi_3p = backbone_neighbors[bi][1]

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
		print("Error: origami bead index not defined for the dummy atom.")


################################################################################
### DNAfold

### translate caDNAno design to DNAfold model
def buildDNAfoldModel(cadFile, p):

	### parse the caDNAno file
	scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap = parseCaDNAno(cadFile)
	
	### initial calculations
	print("Building DNAfold model...")
	p.n_scaf = nnt_scaf // p.nnt_per_bead
	p.n_stap = nnt_stap // p.nnt_per_bead
	p.n_ori = p.n_scaf + p.n_stap
	p.nbead = p.n_scaf + p.n_stap*p.stap_copies
	print("Using " + str(p.n_scaf) + " scaffold beads and " + str(p.n_stap) + " staple beads.")

	### initialze interaction and geometry arrays
	strands = [0 for i in range(p.n_ori)]
	backbone_neighbors = [[-1,-1] for i in range(p.n_ori)]
	complements = [-1 for i in range(p.n_ori)]
	vstrands = [0 for i in range(p.n_ori)]
	is_crossover = [False for i in range(p.n_ori)]

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
		if scaffold[ni_scaffoldArr][1] % p.nnt_per_bead != 0:
			print(f"Error: Scaffold 5' end not located at multiple-of-8 position (vstrand {vstrand}).")
			sys.exit()
	elif scaffold[ni_scaffoldArr][1] % p.nnt_per_bead != 7:
		print(f"Error: Scaffold 5' end not located at multiple-of-8 position (vstrand {vstrand}).")
		sys.exit()

	### track along scaffold until 3' end eached
	while scaffold[ni_scaffoldArr][4] != -1:
		ni_scaffoldArr = find(scaffold[ni_scaffoldArr][4], scaffold[ni_scaffoldArr][5], scaffold)

		### update nucleotide and bead indices
		ni_current += 1
		bi_current = ni_current // p.nnt_per_bead
		scaffold[ni_scaffoldArr].extend([ni_current, bi_current])
		vstrand = scaffold[ni_scaffoldArr][0]

		### store vstrand and backbone bonds for new beads
		if bi_current > (ni_current-1) // p.nnt_per_bead:
			backbone_neighbors[bi_current][0] = bi_current-1
			backbone_neighbors[bi_current-1][1] = bi_current
			vstrands[bi_current] = scaffold[ni_scaffoldArr][0]

		### error message
		elif vstrand != vstrand_prev:
			print("Error: Scaffold crossover not located at nultiple-of-8 position.")
			sys.exit()
		vstrand_prev = vstrand

	### error message
	if scaffold[ni_scaffoldArr][0] % 2 == 0:
		if scaffold[ni_scaffoldArr][1] % p.nnt_per_bead != 7:
			print(f"Error: Scaffold 3' end not located at multiple-of-8 position (vstrand {vstrand}).")
			sys.exit()
	elif scaffold[ni_scaffoldArr][1] % p.nnt_per_bead != 0:
		print(f"Error: Scaffold 3' end not located at multiple-of-8 position (vstrand {vstrand}).")
		sys.exit()

	### loop over staples
	nstap = len(fiveP_ends_stap)
	for sti in range(nstap):

		### new nucleotide and bead incides
		ni_current += 1
		bi_current = ni_current // p.nnt_per_bead

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
			if staples[ni_staplesArr][1] % p.nnt_per_bead != 7:
				print(f"Error: Staple 5' end not located at multiple-of-8 position (vstrand {vstrand}).")
				sys.exit()
		elif staples[ni_staplesArr][1] % p.nnt_per_bead != 0:
			print(f"Error: Staple 5' end not located at multiple-of-8 position (vstrand {vstrand}).")
			sys.exit()

		### track along staple until 3' end eached
		while staples[ni_staplesArr][4] != -1:
			ni_staplesArr = find(staples[ni_staplesArr][4], staples[ni_staplesArr][5], staples)

			### update nucleotide and bead indices
			ni_current += 1
			bi_current = ni_current // p.nnt_per_bead
			staples[ni_staplesArr].extend([ni_current, bi_current])
			vstrand = staples[ni_staplesArr][0]

			### store vstrand, strand, and backbone bonds for new beads
			if bi_current > (ni_current-1) // p.nnt_per_bead:
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
					print(f"Error: Staple crossover not located at nultiple-of-8 position (vstrand {vstrand}).")
					sys.exit()
				else:
					vstrand_prev = vstrand

		### error message
		if staples[ni_staplesArr][0] % 2 == 0:
			if staples[ni_staplesArr][1] % p.nnt_per_bead != 0:
				print(f"Error: Staple 3' end not located at multiple-of-8 position (vstrand {vstrand}).")
				sys.exit()
		elif staples[ni_staplesArr][1] % p.nnt_per_bead != 7:
			print(f"Error: Staple 3' end not located at multiple-of-8 position (vstrand {vstrand}).")
			sys.exit()

	### identify crossovers
	for bi in range(1, p.n_ori):
		if vstrands[bi] != vstrands[bi-1]:
			if strands[bi] == strands[bi-1]:
				is_crossover[bi] = True
				is_crossover[bi-1] = True

	### adjustments for circular scaffold
	if p.circularScaf:
		backbone_neighbors[0][0] = p.n_scaf-1
		backbone_neighbors[p.n_scaf-1][1] = 0
		if vstrands[0] != vstrands[p.n_scaf]:
			is_crossover[0] = True
			is_crossover[p.n_scaf] = True

	### strand count
	p.nstrand = max(strands)+1

	### return results			
	return strands, backbone_neighbors, complements, is_crossover, p


### extract necessary info from caDNAno file
def parseCaDNAno(cadFile):
	print("Parsing caDNAno file...")
	
	### load caDNAno file
	ars.testFileExist(cadFile,"caDNAno")
	with open(cadFile, 'r') as f:
		json_string = f.read()
	j = json.loads(json_string)

	### initialize
	scaffold = []
	staples = []
	fiveP_ends_stap = []
	
	### loop over virtual strands
	for el1 in j["vstrands"]:
		
		### loop over the elements of the virtual strand
		for el2_key, el2 in el1.items():
			
			### read virtual strand index
			if el2_key == "num":
				vstrand_current = el2
			
			### read scaffold side of virtual strand
			elif el2_key == "scaf":
				
				### loop over nucleotides
				for ni_vstrand, neighbors in enumerate(el2):
					
					### store virtual strand index and nucleotide index for current nucleotide and its neighbors
					scaffold_current = [vstrand_current, int(ni_vstrand)]
					for s in neighbors:
						scaffold_current.append(int(s))
					scaffold.append(scaffold_current)
					
					### identify 5' end
					if scaffold_current[2] == -1 and scaffold_current[4] != -1:
						fiveP_end_scaf = scaffold_current
			
			### read staple side of helix
			elif el2_key == "stap":
				
				### loop over nucleotides
				for ni_vstrand, neighbors in enumerate(el2):
					
					### store virtual strand index and nucleotide index for current nucleotide and its neighbors
					staple_current = [vstrand_current, int(ni_vstrand)]
					for s in neighbors:
						staple_current.append(int(s))
					staples.append(staple_current)
					
					### identify 5' end
					if staple_current[2] == -1 and staple_current[4] != -1:
						fiveP_ends_stap.append(staple_current)
			
	### tally up the nucleotides
	nnt_scaf = sum(1 for s in scaffold if s[2] != -1 or s[4] != -1)
	nnt_stap = sum(1 for s in staples if s[2] != -1 or s[4] != -1)

	### error message
	if 'fiveP_end_scaf' not in locals():
		print("Error: Scaffold 5' end not found.")
		sys.exit()
	
	### report
	print(f"Found {nnt_scaf} scaffold nucleotides and {nnt_stap} staple nucleotides.")
	return scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap


### search for entry in strand/index list that matches given strand/index
def find(strand, index, list):
	for i,item in enumerate(list):
		if item[0] == strand and item[1] == index:
			if item[2] == -1 and item[3] == -1 and item[4] == -1 and item[5] == -1:
				return -1
			return i
	print("Error: index not found in strand/index list.")
	sys.exit()


### run the script
if __name__ == "__main__":
	main()
	print()

