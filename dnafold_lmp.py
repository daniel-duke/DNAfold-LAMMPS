import armament as ars
import utils
import utilsLocal
import argparse
import parameters
import numpy as np
import random
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

# To Do
# add blocking cases for angle template that arise from using multiple staple
  # copies, parameterize 90 degree angular potential, reactions that shorten
  # crossover bond length


################################################################################
### Parameters

def main():

	### where to get files
	useMyFiles = True

	### extract files from my mac
	if useMyFiles:

		### chose design
		desID = "2HBx4"
		simTag = ""
		simType = "experiment"
		rstapTag = ""
		rseed = 1

		### choose parameters
		nstep			= 1E7		# steps		- number of simulation steps
		nstep_relax		= 1E5		# steps		- number of steps for relaxation
		dump_every		= 1E4		# steps		- number of steps between positions dumps
		dbox			= 40		# nm		- periodic boundary diameter
		forceBind		= False		# bool		- whether to force hybridization
		startBound		= False		# bool		- whether to start at caDNAno positions
		circularScaf	= True		# bool		- whether the scaffold is circular
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

	### create simulation folder
	outSimFold = outFold + f"sim{p.rseed:02.0f}/"
	ars.createSafeFold(outSimFold)

	### write geometry files
	r, nhyb, nangle = composeGeo(outSimFold, strands, backbone_neighbors, complements, is_crossover, is_reserved_strand, cadFile, p)
	composeGeoVis(outSimFold, strands, backbone_neighbors, r, p)

	### write bond react files
	nreact_bond = writeReactBond(outSimFold, backbone_neighbors, complements, is_crossover, p)

	### write angle react files
	nreact_angle_hyb, nreact_angle_dehyb = writeReactAngle(outSimFold, strands, backbone_neighbors, complements, is_crossover, p)

	### write lammps input file
	writeInput(outSimFold, is_crossover, nhyb, nangle, nreact_bond, nreact_angle_hyb, nreact_angle_dehyb, p)


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
		'r_h_bead':		1.28,			# nm			- hydrodynamic radius of single bead
		'visc':			0.8472,			# mPa/s			- viscosity (units equivalent to pN*ns/mn^2)
		'sigma':		2.14,			# nm			- bead Van der Waals radius
		'epsilon':		4.0,			# kcal/mol		- WCA energy parameter
		'r12_eq':		2.72,			# nm			- equilibrium bead separation
		'k_x': 			120.0,			# kcal/mol/nm2	- backbone spring constant (standard definition)
		'r12_cut_hyb':	2.0,			# nm			- hybridization potential cutoff radius
		'U_hyb':		10.0,			# kcal/mol		- depth of hybridization potential
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

	### initailize positions
	if p.startBound:
		stap_offset = 0.01
		r = utils.initPositionsCaDNAno(cadFile)[0]
		r[p.n_scaf:] += stap_offset
	else:
		r = initPositions(strands, p)
	r = np.append(r,np.zeros((p.n_scaf,3)),axis=0)

	### initialize
	molecules = np.ones(p.nbead+p.n_scaf,dtype=int)
	types = np.ones(p.nbead+p.n_scaf,dtype=int)
	charges = np.zeros(p.nbead+p.n_scaf)
	bonds = np.zeros((0,3),dtype=int)
	angles = np.zeros((0,4),dtype=int)

	### scaffold atoms
	charge_step = 1/(10**len(str(p.n_scaf)))
	for bi in range(p.n_scaf):
		charges[bi] = charge_step*(bi+1)

	### staple atoms
	nhyb = 0
	for ci in range(p.stap_copies):
		for obi in range(p.n_scaf,p.n_ori):
			rbi = obi + ci*p.n_stap
			molecules[rbi] = strands[obi] + ci*(p.nstrand-1) + 1
			types[rbi] = 2
			if complements[obi] != -1:
				if ci == 0:
					nhyb += 1
				charges[rbi] = charge_step*(complements[obi]+1)
			if is_reserved_strand[strands[obi]]:
				types[rbi] = 3
				r[rbi] = [0,0,0]

	### dummy atoms
	for sbi in range(p.n_scaf):
		dbi = sbi + p.nbead
		molecules[dbi] = 0
		types[dbi] = 3
		charges[dbi] = is_crossover[sbi] + 1

	### scaffold backbone bonds
	for bi in range(p.n_scaf-1):
		type = 1
		atom1 = bi + 1
		atom2 = bi + 2
		bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### scaffold end-to-end bond
	if p.circularScaf:
		bonds = np.append(bonds,[[1,p.n_scaf,1]],axis=0)
	else:
		if getScafNeighbors(0,backbone_neighbors,complements)[0] == p.n_scaf-1:
			bonds = np.append(bonds,[[3,1,p.n_scaf]],axis=0)
			bonds = np.append(bonds,[[3,1,p.n_scaf+p.nbead]],axis=0)
			bonds = np.append(bonds,[[3,p.n_scaf,1+p.nbead]],axis=0)

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

	### dummy bonds
	for bi in range(p.n_scaf):
		type = 3
		atom1 = bi + 1
		atom2 = bi + p.nbead + 1
		bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### hybridization bonds
	if p.forceBind or p.startBound:
		for bi in range(p.n_scaf):
			if complements[bi] != -1:
				type = 2
				atom1 = bi + 1
				atom2 = complements[bi] + 1
				bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### count angles
	nangle = 0
	for bi in range(p.n_scaf):
		bi_5p,bi_3p = getScafNeighbors(bi,backbone_neighbors,complements)
		if bi_5p != -1 and bi_3p != -1:
			if complements[bi_5p] != -1 and complements[bi] != -1 and complements[bi_3p] != -1:
				nangle += 1

	### write file
	outGeoFile = outSimFold + "geometry.in"
	ars.writeGeo(outGeoFile, p.dbox, r, molecules, types, bonds, nangleType=2, charges=charges)

	### return positions (without dummy atoms)
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
def writeInput(outSimFold, is_crossover, nhyb, nangle, nreact_bond, nreact_angle_hyb, nreact_angle_dehyb,  p):
	print("Writing input file...")

	### computational parameters
	verlet_skin				= 4		# nm		- width of neighbor list skin (= r12_cut - sigma)
	neigh_every				= 10	# steps		- how often to consider updating neighbor list
	bond_res 				= 0.1	# nm		- distance between tabular bond interpolation points
	r12_cut_react_bond		= 4		# nm		- cutoff radius for potential hybridization bonds
	react_every_bond		= 1E2	# steps		- how often to check for new hybridization bonds
	react_every_angle_hyb	= 1E4	# steps		- how often to check for new hybridization angles
	react_every_angle_dehyb	= 5E3	# steps		- how often to check for removed hybridization angles

	### bond file calculations
	r12_max = np.sqrt(3)*p.dbox
	r12_max = r12_max - r12_max%bond_res + bond_res
	npoint_bond = int(r12_max/bond_res+1)

	### count digits
	len_nreact_bond = len(str(nreact_bond))
	len_nreact_angle_hyb = len(str(nreact_angle_hyb))
	len_nreact_angle_dehyb = len(str(nreact_angle_dehyb))

	### write table for hybridization bond
	writeBondHyb(outSimFold, bond_res, r12_max, p)

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
			"pair_coeff      * 3 zero\n"
			"special_bonds   lj 0.0 1.0 1.0\n")

		### bonded interactions
		f.write(
		   f"bond_style      hybrid zero harmonic table linear {npoint_bond}\n"
		   f"bond_coeff      1 harmonic {p.k_x/2:0.2f} {p.r12_eq:0.2f}\n"
		   f"bond_coeff      2 table bond_hyb.txt hyb\n"
			"bond_coeff      3 zero\n")

		### angled interactions
		if nangle:
			f.write(
			   f"angle_style     harmonic\n"
			   f"angle_coeff     1 {p.k_theta/2:0.2f} 180\n"
			   f"angle_coeff     2 {p.k_theta/2:0.2f} 90\n")

		### group atoms
		f.write(
			"variable        varQ atom q\n"
		   f"group           real id <= {p.nbead}\n"
			"group           mobile type 1 2\n\n")

		### relax everything
		f.write(
			"## Relaxation\n"
		   f"fix             tstat1 mobile langevin {p.T} {p.T} {1/p.gamma_t:0.4f} {p.rseed}\n"
			"fix             tstat2 mobile nve/limit 0.1\n"
		   f"timestep        {p.dt}\n"
		   f"run             {int(p.nstep_relax)}\n"
			"unfix           tstat1\n"
			"unfix           tstat2\n"
			"reset_timestep  0\n\n")

		### reactions
		if nreact_angle_hyb or nreact_bond:

			### molecule template header
			f.write(
				"## Molecule Templates\n")

			### bond templates
			for ri in range(nreact_bond):
				f.write(
			   f"molecule        bondHyb{ri+1:0>{len_nreact_bond}}_mol_pre react/bondHyb{ri+1:0>{len_nreact_bond}}_mol_pre.txt\n"
			   f"molecule        bondHyb{ri+1:0>{len_nreact_bond}}_mol_pst react/bondHyb{ri+1:0>{len_nreact_bond}}_mol_pst.txt\n")

			### angle hybridization templates
			for ri in range(nreact_angle_hyb):
				f.write(
			   f"molecule        angleHyb{ri+1:0>{len_nreact_angle_hyb}}_mol_pre react/angleHyb{ri+1:0>{len_nreact_angle_hyb}}_mol_pre.txt\n"
			   f"molecule        angleHyb{ri+1:0>{len_nreact_angle_hyb}}_mol_pst react/angleHyb{ri+1:0>{len_nreact_angle_hyb}}_mol_pst.txt\n")

			### angle dehybridization templates
			for ri in range(nreact_angle_dehyb):
				f.write(
			   f"molecule        angleDehyb{ri+1:0>{len_nreact_angle_dehyb}}_mol react/angleDehyb{ri+1:0>{len_nreact_angle_dehyb}}_mol.txt\n")
			f.write("\n")

			### reaction header
			f.write(
				"## Reactions\n")

			### bond dehybridization reactions
			if p.dehyb and not p.forceBind:
				f.write(
			   f"fix             bondDehyb all bond/break {int(react_every_bond)} 2 {r12_cut_react_bond:.1f}\n")

			### bond hybridization reactions
			f.write(
				"fix             reactions all bond/react reset_mol_ids no")
			for ri in range(nreact_bond):
				f.write(
			   f" &\n                react bondHyb{ri+1} all {int(react_every_bond)} 0.0 {r12_cut_react_bond:.1f} bondHyb{ri+1:0>{len_nreact_bond}}_mol_pre bondHyb{ri+1:0>{len_nreact_bond}}_mol_pst react/bondHyb{ri+1:0>{len_nreact_bond}}_map.txt")
			
			### angle hybridization reactions
			react_copies = 1
			for ri in range(nreact_angle_hyb):
				for i in range(react_copies):
					f.write(
			   f" &\n                react angleHyb{ri+1:0>{len_nreact_angle_hyb}} all {int(react_every_angle_hyb)} 0.0 {p.r12_cut_hyb:.1f} angleHyb{ri+1:0>{len_nreact_angle_hyb}}_mol_pre angleHyb{ri+1:0>{len_nreact_angle_hyb}}_mol_pst react/angleHyb{ri+1:0>{len_nreact_angle_hyb}}_map.txt custom_charges 4")

			### angle dehybridization reactions
			for ri in range(nreact_angle_dehyb):
				f.write(
			   f" &\n                react angleDehyb{ri+1:0>{len_nreact_angle_dehyb}} all {int(react_every_angle_dehyb)} {p.r12_cut_hyb:.1f} {r12_max:.1f} angleDehyb{ri+1:0>{len_nreact_angle_dehyb}}_mol angleDehyb{ri+1:0>{len_nreact_angle_dehyb}}_mol react/angleDehyb{ri+1:0>{len_nreact_angle_dehyb}}_map.txt custom_charges 4")
			f.write("\n\n")

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
			   f"dump            dumpD3 all custom {int(p.dump_every)} dump_charges.dat id q\n"
				"dump_modify     dumpD3 sort id append yes\n")

		### 
		f.write(
			"## Production\n"
		   f"fix             tstat1 mobile langevin {p.T} {p.T} {1/p.gamma_t:0.4f} {p.rseed}\n"
			"fix             tstat2 mobile nve\n"
		   f"timestep        {p.dt}\n"
		   f"dump            dump1 real custom {int(p.dump_every)} trajectory.dat id mol xs ys zs\n"
			"dump_modify     dump1 sort id append yes\n"
		   f"restart         {int(p.dump_every/2)} restart_binary1.out restart_binary2.out\n\n")

		f.write(
			"## Go Time\n"
		   f"run             {int(p.nstep)}\n"
			"write_data      restart_geometry.out\n\n")


### write reaction files for hybridization angles
def writeReactBond(outSimFold, backbone_neighbors, complements, is_crossover, p):
	print("Writing bond react files...")

	### no reactions necessary for forced binding
	if p.forceBind:
		return 0

	### initialize
	atoms_all = []
	bonds_all = []
	edges_all = []

	### loop over scaffold beads
	for bi in range(p.n_scaf):

		### initialize
		comp_5p = False
		comp_3p = False

		### get neighbors to central scaffold bead
		b_5p,b_3p = getScafNeighbors(bi,backbone_neighbors,complements)

		### skip if central scaffold bead is not complimentary
		if complements[bi] == -1:
			continue

		#-------- working from 5' scaffold end --------#

		### intialize
		b = bi
		bC = complements[b]
		bD = b + p.nbead

		### core topology
		atoms_5to3 =  [ [0,-1,b], [1,-1,bC], [2,int(is_crossover[b]),bD] ]
		bonds_5to3 =  [ [2,b,bD] ]
		edges_5to3 =  [ ]

		### add central scaffold 5' side topology
		if b_5p != -1:
			atoms_5to3.append([0,-1,b_5p])
			edges_5to3.append(b_5p)
			if b_5p == backbone_neighbors[b][0]:
				bonds_5to3.append([0,b_5p,b])
			else:
				b_5p_D = b_5p + p.nbead
				atoms_5to3.append([2,int(is_crossover[b_5p]),b_5p_D])
				edges_5to3.append(b_5p_D)
				bonds_5to3.append([2,b_5p,b])
				bonds_5to3.append([2,b_5p,b_5p_D])
				bonds_5to3.append([2,b,b_5p_D])
				bonds_5to3.append([2,b_5p,bD])

		### add central scaffold 3' side topology
		if b_3p != -1:
			atoms_5to3.append([0,-1,b_3p])
			edges_5to3.append(b_3p)
			if b_3p == backbone_neighbors[b][1]:
				bonds_5to3.append([0,b,b_3p])
			else:
				b_3p_D = b_3p + p.nbead
				atoms_5to3.append([2,int(is_crossover[b_3p]),b_3p_D])
				edges_5to3.append(b_3p_D)
				bonds_5to3.append([2,b,b_3p])
				bonds_5to3.append([2,b_3p,b_3p_D])
				bonds_5to3.append([2,b,b_3p_D])
				bonds_5to3.append([2,b_3p,bD])

		### add central staple 5' end topology
		bC_5p = backbone_neighbors[bC][0]
		if bC_5p != -1:
			atoms_5to3.append([1,-1,bC_5p])
			bonds_5to3.append([0,bC_5p,bC])
			edges_5to3.append(bC_5p)

		### add central staple 3' end topology
		bC_3p = backbone_neighbors[bC][1]
		if bC_3p != -1:
			atoms_5to3.append([1,-1,bC_3p])
			bonds_5to3.append([0,bC,bC_3p])
			edges_5to3.append(bC_3p)

		### add central scaffold 5' end hybridization bond
		if b_5p != -1 and bC_3p != -1 and bC_3p == complements[b_5p]:
			comp_5p = True
			bonds_5to3.append([1,b_5p,bC_3p])

		### add central scaffold 3' end hybridization bond
		if b_3p != -1 and bC_5p != -1 and bC_5p == complements[b_3p]:
			comp_3p = True
			bonds_5to3.append([1,b_3p,bC_5p])

		#-------- working from 3' scaffold end --------#

		### intialize
		b = bi
		bC = complements[b]
		bD = b + p.nbead

		### core topology
		atoms_3to5 =  [ [0,-1,b], [1,-1,bC], [2,int(is_crossover[b]),bD] ]
		bonds_3to5 =  [ [2,b,bD] ]
		edges_3to5 =  [ ]

		### add central scaffold 3' side topology
		if b_3p != -1:
			atoms_3to5.append([0,-1,b_3p])
			edges_3to5.append(b_3p)
			if b_3p == backbone_neighbors[b][1]:
				bonds_3to5.append([0,b_3p,b])
			else:
				b_3p_D = b_3p + p.nbead
				atoms_3to5.append([2,int(is_crossover[b_3p]),b_3p_D])
				edges_3to5.append(b_3p_D)
				bonds_3to5.append([2,b_3p,b])
				bonds_3to5.append([2,b_3p,b_3p_D])
				bonds_3to5.append([2,b,b_3p_D])
				bonds_3to5.append([2,b_3p,bD])

		### add central scaffold 5' side topology
		if b_5p != -1:
			atoms_3to5.append([0,-1,b_5p])
			edges_3to5.append(b_5p)
			if b_5p == backbone_neighbors[b][0]:
				bonds_3to5.append([0,b,b_5p])
			else:
				b_5p_D = b_5p + p.nbead
				atoms_3to5.append([2,int(is_crossover[b_5p]),b_5p_D])
				edges_3to5.append(b_5p_D)
				bonds_3to5.append([2,b,b_5p])
				bonds_3to5.append([2,b_5p,b_5p_D])
				bonds_3to5.append([2,b,b_5p_D])
				bonds_3to5.append([2,b_5p,bD])

		### add central staple 3' end topology
		bC_3p = backbone_neighbors[bC][1]
		if bC_3p != -1:
			atoms_3to5.append([1,-1,bC_3p])
			bonds_3to5.append([0,bC_3p,bC])
			edges_3to5.append(bC_3p)

		### add central staple 5' end topology
		bC_5p = backbone_neighbors[bC][0]
		if bC_5p != -1:
			atoms_3to5.append([1,-1,bC_5p])
			bonds_3to5.append([0,bC,bC_5p])
			edges_3to5.append(bC_5p)

		### add central scaffold 3' end hybridization bond
		if comp_3p:
			bonds_3to5.append([1,b_3p,bC_5p])

		### add central scaffold 5' end hybridization bond
		if comp_5p:
			bonds_3to5.append([1,b_5p,bC_3p])

		#-------- prepare template for comparison --------#

		### renumber
		atoms_5to3,bonds_5to3,edges_5to3 = renumberBond(atoms_5to3,bonds_5to3,edges_5to3)
		atoms_3to5,bonds_3to5,edges_3to5 = renumberBond(atoms_3to5,bonds_3to5,edges_3to5)

		### test for symmetry
		abe_zipped = [[a,b,d] for a,b,d in zip([atoms_5to3,atoms_3to5],[bonds_5to3,bonds_3to5],[edges_5to3,edges_3to5])]
		abe_zipped = removeDuplicateEntries(abe_zipped)
		if len(abe_zipped) == 1:
			symmetric = True
		else:
			symmetric = False

		#-------- add to templates --------#

		atoms_all.append(atoms_5to3)
		bonds_all.append(bonds_5to3)
		edges_all.append(edges_5to3)
		atoms_all.append(atoms_3to5)
		bonds_all.append(bonds_3to5)
		edges_all.append(edges_3to5)
		abe_zipped = [[a,b,d] for a,b,d in zip(atoms_all,bonds_all,edges_all)]
		abe_zipped = removeDuplicateEntries(abe_zipped)
		if not symmetric:
			abe_zipped.pop()
		atoms_all,bonds_all,edges_all = unzip3(abe_zipped)

		if comp_5p or comp_3p:
			bonds_5to3_copy = bonds_5to3[:-1]
			bonds_3to5_copy = bonds_3to5[:-1]
			atoms_all.append(atoms_5to3)
			bonds_all.append(bonds_5to3_copy)
			edges_all.append(edges_5to3)
			atoms_all.append(atoms_3to5)
			bonds_all.append(bonds_3to5_copy)
			edges_all.append(edges_3to5)
			abe_zipped = [[a,b,d] for a,b,d in zip(atoms_all,bonds_all,edges_all)]
			abe_zipped = removeDuplicateEntries(abe_zipped)
			if not symmetric:
				abe_zipped.pop()
			atoms_all,bonds_all,edges_all = unzip3(abe_zipped)

		if comp_5p and comp_3p:
			bonds_5to3_copy = bonds_5to3[:-2] + [bonds_5to3[-1]]
			bonds_3to5_copy = bonds_3to5[:-2] + [bonds_3to5[-1]]
			atoms_all.append(atoms_5to3)
			bonds_all.append(bonds_5to3_copy)
			edges_all.append(edges_5to3)
			atoms_all.append(atoms_3to5)
			bonds_all.append(bonds_3to5_copy)
			edges_all.append(edges_3to5)
			abe_zipped = [[a,b,d] for a,b,d in zip(atoms_all,bonds_all,edges_all)]
			abe_zipped = removeDuplicateEntries(abe_zipped)
			if not symmetric:
				abe_zipped.pop()
			atoms_all,bonds_all,edges_all = unzip3(abe_zipped)

			bonds_5to3_copy = bonds_5to3[:-2]
			bonds_3to5_copy = bonds_3to5[:-2]
			atoms_all.append(atoms_5to3)
			bonds_all.append(bonds_5to3_copy)
			edges_all.append(edges_5to3)
			atoms_all.append(atoms_3to5)
			bonds_all.append(bonds_3to5_copy)
			edges_all.append(edges_3to5)
			abe_zipped = [[a,b,d] for a,b,d in zip(atoms_all,bonds_all,edges_all)]
			abe_zipped = removeDuplicateEntries(abe_zipped)
			if not symmetric:
				abe_zipped.pop()
			atoms_all,bonds_all,edges_all = unzip3(abe_zipped)

	#-------- write hybridization files --------#

	nreact = len(atoms_all)
	len_nreact = len(str(nreact))
	outReactFold = outSimFold + "react/"
	ars.createEmptyFold(outReactFold)

	for ri in range(nreact):

		natom = len(atoms_all[ri])
		nbond = len(bonds_all[ri])
		nedge = len(edges_all[ri])

		if p.debug:
			if ri == 0:
				print()
			print(f"Bond template {ri+1} (hybridization):")
			print(atoms_all[ri])
			print(bonds_all[ri])
			print(edges_all[ri])
			print()

		molPreFile = f"{outReactFold}bondHyb{ri+1:0>{len_nreact}}_mol_pre.txt"
		with open(molPreFile, 'w') as f:

			atoms = copy.deepcopy(atoms_all[ri])
			bonds = copy.deepcopy(bonds_all[ri])

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")
			f.write(f"2 fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

			f.write("\nFragments\n\n")
			f.write("1\t1\n")
			f.write("2\t2\n")

		molPstFile = f"{outReactFold}bondHyb{ri+1:0>{len_nreact}}_mol_pst.txt"
		with open(molPstFile, 'w') as f:

			atoms = copy.deepcopy(atoms_all[ri])
			bonds = copy.deepcopy(bonds_all[ri])

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond+1} bonds\n")
			f.write(f"2 fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")
			f.write(f"{bondi+2}\t2\t1\t2\n")

			f.write("\nFragments\n\n")
			f.write("1\t1\n")
			f.write("2\t2\n")

		mapFile = f"{outReactFold}bondHyb{ri+1:0>{len_nreact}}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"1 constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write("1\n")
			f.write("2\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges_all[ri][edgei]+1}\n")
			
			f.write("\nConstraints\n\n")
			f.write("custom \"rxnsum(v_varQ,1) == rxnsum(v_varQ,2)\"\n")

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

	### return reaction counts
	return nreact


### write reaction files for hybridization angles
def writeReactAngle(outSimFold, strands, backbone_neighbors, complements, is_crossover, p):
	print("Writing angle react files...")

	### initialize hyb
	atoms_all_hyb = []
	bonds_all_hyb = []
	angles_all_hyb = []
	edges_all_hyb = []

	### initialize dehyb
	atoms_all_dehyb = []
	bonds_all_dehyb = []
	angles_all_dehyb = []
	edges_all_dehyb = []

	### loop over candidate beads
	for bi in range(p.n_scaf):

		### get neighbors to candidate bead
		bi_5p,bi_3p = getScafNeighbors(bi,backbone_neighbors,complements)

		### skip if core scaffold is not present
		if bi_5p == -1 or bi_3p == -1:
			continue

		### skip if core scaffold is not fully complimentary
		if complements[bi_5p] == -1 or complements[bi] == -1 or complements[bi_3p] == -1:
			continue

		#-------- working from 5' scaffold end --------#

		### intialize
		a = bi_5p
		b = bi
		c = bi_3p
		aC = complements[a]
		bC = complements[b]
		cC = complements[c]
		aD = a + p.nbead
		bD = b + p.nbead
		cD = c + p.nbead

		### core topology
		atoms_5to3 =  [ [0,-1,a], [0,-1,b], [0,-1,c], [1,-1,cC], [1,-1,bC], [1,-1,aC], [2,int(is_crossover[a]),aD], [2,int(is_crossover[b]),bD], [2,int(is_crossover[c]),cD] ]
		bonds_5to3 =  [ [1,a,aC], [1,b,bC], [1,c,cC], [2,a,aD], [2,b,bD], [2,c,cD] ]
		angles_5to3 = [ [int(is_crossover[b]),a,b,c,1] ]
		edges_5to3 =  [ cC, aC ]

		### add core 5' side topology
		if backbone_neighbors[b][0] == a:
			bonds_5to3.append([0,a,b])
		else:
			bonds_5to3.append([2,a,b])
			bonds_5to3.append([2,b,aD])
			bonds_5to3.append([2,a,bD])

		### add core 3' side topology
		if backbone_neighbors[b][1] == c:
			bonds_5to3.append([0,b,c])
		else:
			bonds_5to3.append([2,b,c])
			bonds_5to3.append([2,b,cD])
			bonds_5to3.append([2,c,bD])

		### add scaffold 5' end topology
		a_5p = getScafNeighbors(a, backbone_neighbors, complements)[0]
		if a_5p != -1:
			atoms_5to3.append([0,-1,a_5p])
			angles_5to3.append([int(is_crossover[a]),a_5p,a,b,0])
			edges_5to3.append(a_5p)
			if backbone_neighbors[a][0] == a_5p:
				bonds_5to3.append([0,a_5p,a])
			else:
				bonds_5to3.append([2,a_5p,a])
				a_5p_D = a_5p + p.nbead
				atoms_5to3.append([2,int(is_crossover[a_5p]),a_5p_D])
				bonds_5to3.append([2,a_5p,a_5p_D])
				bonds_5to3.append([2,a,a_5p_D])
				bonds_5to3.append([2,a_5p,aD])

		### add scaffold 3' end topology
		c_3p = getScafNeighbors(c, backbone_neighbors, complements)[1]
		if c_3p != -1:
			atoms_5to3.append([0,-1,c_3p])
			angles_5to3.append([int(is_crossover[c]),b,c,c_3p,2])
			edges_5to3.append(c_3p)
			if backbone_neighbors[c][1] == c_3p:
				bonds_5to3.append([0,c,c_3p])
			else:
				bonds_5to3.append([2,c,c_3p])
				c_3p_D = c_3p + p.nbead
				atoms_5to3.append([2,int(is_crossover[c_3p]),c_3p_D])
				bonds_5to3.append([2,c_3p,c_3p_D])
				bonds_5to3.append([2,c,c_3p_D])
				bonds_5to3.append([2,c_3p,cD])

		### add central staple 5' end topology
		bC_5p = backbone_neighbors[bC][0]
		if bC_5p == cC:
			bonds_5to3.append([0,cC,bC])
		elif bC_5p != -1:
			if [1,-1,bC_5p] not in atoms_5to3:
				atoms_5to3.append([1,-1,bC_5p])
				edges_5to3.append(bC_5p)
			bonds_5to3.append([0,bC_5p,bC])

		### add central staple 3' end topology
		bC_3p = backbone_neighbors[bC][1]
		if bC_3p == aC:
			bonds_5to3.append([0,bC,aC])
		elif bC_3p != -1:
			if [1,-1,bC_3p] not in atoms_5to3:
				atoms_5to3.append([1,-1,bC_3p])
				edges_5to3.append(bC_3p)
			bonds_5to3.append([0,bC,bC_3p])

		#-------- working from 3' scaffold end --------#

		### intialize
		a = bi_3p
		b = bi
		c = bi_5p
		aC = complements[a]
		bC = complements[b]
		cC = complements[c]
		aD = a + p.nbead
		bD = b + p.nbead
		cD = c + p.nbead

		### core topology
		atoms_3to5 =  [ [0,-1,a], [0,-1,b], [0,-1,c], [1,-1,cC], [1,-1,bC], [1,-1,aC], [2,int(is_crossover[a]),aD], [2,int(is_crossover[b]),bD], [2,int(is_crossover[c]),cD] ]
		bonds_3to5 =  [ [1,a,aC], [1,b,bC], [1,c,cC], [2,a,aD], [2,b,bD], [2,c,cD] ]
		angles_3to5 = [ [int(is_crossover[b]),a,b,c,1] ]
		edges_3to5 =  [ cC, aC ]

		### add core 3' side topology
		if backbone_neighbors[b][1] == a:
			bonds_3to5.append([0,a,b])
		else:
			bonds_3to5.append([2,a,b])
			bonds_3to5.append([2,b,aD])
			bonds_3to5.append([2,a,bD])

		### add core 5' side topology
		if backbone_neighbors[b][0] == c:
			bonds_3to5.append([0,b,c])
		else:
			bonds_3to5.append([2,b,c])
			bonds_3to5.append([2,b,cD])
			bonds_3to5.append([2,c,bD])

		### add scaffold 3' end topology
		a_3p = getScafNeighbors(a, backbone_neighbors, complements)[1]
		if a_3p != -1:
			atoms_3to5.append([0,-1,a_3p])
			angles_3to5.append([int(is_crossover[a]),a_3p,a,b,0])
			edges_3to5.append(a_3p)
			if backbone_neighbors[a][1] == a_3p:
				bonds_3to5.append([0,a_3p,a])
			else:
				bonds_3to5.append([2,a_3p,a])
				a_3p_D = a_3p + p.nbead
				atoms_3to5.append([2,int(is_crossover[a_3p]),a_3p_D])
				bonds_3to5.append([2,a_3p,a_3p_D])
				bonds_3to5.append([2,a,a_3p_D])
				bonds_3to5.append([2,a_3p,aD])

		### add scaffold 5' end topology
		c_5p = getScafNeighbors(c, backbone_neighbors, complements)[0]
		if c_5p != -1:
			atoms_3to5.append([0,-1,c_5p])
			angles_3to5.append([int(is_crossover[c]),b,c,c_5p,2])
			edges_3to5.append(c_5p)
			if backbone_neighbors[c][0] == c_5p:
				bonds_3to5.append([0,c,c_5p])
			else:
				bonds_3to5.append([2,c,c_5p])
				c_5p_D = c_5p + p.nbead
				atoms_3to5.append([2,int(is_crossover[c_5p]),c_5p_D])
				bonds_3to5.append([2,c_5p,c_5p_D])
				bonds_3to5.append([2,c,c_5p_D])
				bonds_3to5.append([2,c_5p,cD])

		### add central staple 3' end topology
		bC_3p = backbone_neighbors[bC][1]
		if bC_3p == cC:
			bonds_3to5.append([0,cC,bC])
		elif bC_3p != -1:
			if [1,-1,bC_3p] not in atoms_3to5:
				atoms_3to5.append([1,-1,bC_3p])
				edges_3to5.append(bC_3p)
			bonds_3to5.append([0,bC_3p,bC])

		### add central staple 5' end topology
		bC_5p = backbone_neighbors[bC][0]
		if bC_5p == aC:
			bonds_3to5.append([0,bC,aC])
		elif bC_5p != -1:
			if [1,-1,bC_5p] not in atoms_3to5:
				atoms_3to5.append([1,-1,bC_5p])
				edges_3to5.append(bC_5p)
			bonds_3to5.append([0,bC,bC_5p])

		#-------- prepare template for comparison --------#

		### renumber
		atoms_5to3,bonds_5to3,angles_5to3,edges_5to3 = renumberAngle(atoms_5to3,bonds_5to3,angles_5to3,edges_5to3)
		atoms_3to5,bonds_3to5,angles_3to5,edges_3to5 = renumberAngle(atoms_3to5,bonds_3to5,angles_3to5,edges_3to5)

		### test for symmetry
		abae_zipped = [[a,b,c,d] for a,b,c,d in zip([atoms_5to3,atoms_3to5],[bonds_5to3,bonds_3to5],[angles_5to3,angles_3to5],[edges_5to3,edges_3to5])]
		abae_zipped = removeDuplicateEntries(abae_zipped)
		if len(abae_zipped) == 1:
			symmetric = True
		else:
			symmetric = False

		#-------- add to hybridization templates --------#

		atoms_all_hyb.append(atoms_5to3)
		bonds_all_hyb.append(bonds_5to3)
		angles_all_hyb.append(angles_5to3)
		edges_all_hyb.append(edges_5to3)
		atoms_all_hyb.append(atoms_3to5)
		bonds_all_hyb.append(bonds_3to5)
		angles_all_hyb.append(angles_3to5)
		edges_all_hyb.append(edges_3to5)
		abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb)]
		abae_zipped = removeDuplicateEntries(abae_zipped)
		if not symmetric:
			abae_zipped.pop()
		atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb = unzip4(abae_zipped)

		if len(angles_5to3) >= 2:
			angles_5to3_copy = copy.deepcopy(angles_5to3)
			angles_5to3_copy[1][0] += 2
			angles_3to5_copy = copy.deepcopy(angles_3to5)
			angles_3to5_copy[1][0] += 2
			atoms_5to3_copy = copy.deepcopy(atoms_5to3)
			atoms_5to3_copy[angles_5to3[1][4]+6][1] += 2
			atoms_3to5_copy = copy.deepcopy(atoms_3to5)
			atoms_3to5_copy[angles_3to5[1][4]+6][1] += 2
			atoms_all_hyb.append(atoms_5to3_copy)
			bonds_all_hyb.append(bonds_5to3)
			angles_all_hyb.append(angles_5to3_copy)
			edges_all_hyb.append(edges_5to3)
			atoms_all_hyb.append(atoms_3to5_copy)
			bonds_all_hyb.append(bonds_3to5)
			angles_all_hyb.append(angles_3to5_copy)
			edges_all_hyb.append(edges_3to5)
			abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb)]
			abae_zipped = removeDuplicateEntries(abae_zipped)
			if not symmetric:
				abae_zipped.pop()
			atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb = unzip4(abae_zipped)

		if len(angles_5to3) >= 3:
			if not symmetric:
				angles_5to3_copy = copy.deepcopy(angles_5to3)
				angles_5to3_copy[2][0] += 2
				angles_3to5_copy = copy.deepcopy(angles_3to5)
				angles_3to5_copy[2][0] += 2
				atoms_5to3_copy = copy.deepcopy(atoms_5to3)
				atoms_5to3_copy[angles_5to3[2][4]+6][1] += 2
				atoms_3to5_copy = copy.deepcopy(atoms_3to5)
				atoms_3to5_copy[angles_3to5[2][4]+6][1] += 2
				atoms_all_hyb.append(atoms_5to3_copy)
				bonds_all_hyb.append(bonds_5to3)
				angles_all_hyb.append(angles_5to3_copy)
				edges_all_hyb.append(edges_5to3)
				atoms_all_hyb.append(atoms_3to5_copy)
				bonds_all_hyb.append(bonds_3to5)
				angles_all_hyb.append(angles_3to5_copy)
				edges_all_hyb.append(edges_3to5)
				abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb)]
				abae_zipped = removeDuplicateEntries(abae_zipped)
				abae_zipped.pop()
				atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb = unzip4(abae_zipped)

			angles_5to3_copy = copy.deepcopy(angles_5to3)
			angles_5to3_copy[1][0] += 2
			angles_5to3_copy[2][0] += 2
			angles_3to5_copy = copy.deepcopy(angles_3to5)
			angles_3to5_copy[1][0] += 2
			angles_3to5_copy[2][0] += 2
			atoms_5to3_copy = copy.deepcopy(atoms_5to3)
			atoms_5to3_copy[angles_5to3[1][4]+6][1] += 2
			atoms_5to3_copy[angles_5to3[2][4]+6][1] += 2
			atoms_3to5_copy = copy.deepcopy(atoms_3to5)
			atoms_3to5_copy[angles_3to5[1][4]+6][1] += 2
			atoms_3to5_copy[angles_3to5[2][4]+6][1] += 2
			atoms_all_hyb.append(atoms_5to3_copy)
			bonds_all_hyb.append(bonds_5to3)
			angles_all_hyb.append(angles_5to3_copy)
			edges_all_hyb.append(edges_5to3)
			atoms_all_hyb.append(atoms_3to5_copy)
			bonds_all_hyb.append(bonds_3to5)
			angles_all_hyb.append(angles_3to5_copy)
			edges_all_hyb.append(edges_3to5)
			abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb)]
			abae_zipped = removeDuplicateEntries(abae_zipped)
			if not symmetric:
				abae_zipped.pop()
			atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb = unzip4(abae_zipped)

		#-------- add to dehybridization templates --------#

		if p.dehyb:
			atoms_all_dehyb.append(atoms_5to3)
			bonds_all_dehyb.append(bonds_5to3)
			angles_all_dehyb.append(angles_5to3)
			edges_all_dehyb.append(edges_5to3)
			atoms_all_dehyb.append(atoms_3to5)
			bonds_all_dehyb.append(bonds_3to5)
			angles_all_dehyb.append(angles_3to5)
			edges_all_dehyb.append(edges_3to5)
			abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all_dehyb,bonds_all_dehyb,angles_all_dehyb,edges_all_dehyb)]
			abae_zipped = removeDuplicateEntries(abae_zipped)
			if not symmetric:
				abae_zipped.pop()
			atoms_all_dehyb,bonds_all_dehyb,angles_all_dehyb,edges_all_dehyb = unzip4(abae_zipped)

	#-------- write hybridization files --------#

	nreact_hyb = len(atoms_all_hyb)
	nreact_dehyb = len(atoms_all_dehyb)
	len_nreact_hyb = len(str(nreact_hyb))
	len_nreact_dehyb = len(str(nreact_dehyb))
	outReactFold = outSimFold + "react/"
	if p.forceBind:
		ars.createEmptyFold(outReactFold)		

	for ri in range(nreact_hyb):

		natom = len(atoms_all_hyb[ri])
		nbond = len(bonds_all_hyb[ri])
		nangle = len(angles_all_hyb[ri])
		nedge = len(edges_all_hyb[ri])

		if p.debug:
			print(f"Angle template {ri+1} (hybridization):")
			print(atoms_all_hyb[ri])
			print(bonds_all_hyb[ri])
			print(angles_all_hyb[ri])
			print(edges_all_hyb[ri])
			print()

		molFile = f"{outReactFold}angleHyb{ri+1:0>{len_nreact_hyb}}_mol_pre.txt"
		with open(molFile, 'w') as f:

			atoms = copy.deepcopy(atoms_all_hyb[ri])
			bonds = copy.deepcopy(bonds_all_hyb[ri])
			angles = copy.deepcopy(angles_all_hyb[ri])

			nangle_on = 0
			for anglei in range(nangle):
				if angles[anglei][0] > 1:
					nangle_on += 1

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")
			if nangle_on:
				f.write(f"{nangle_on} angles\n")
			f.write(f"4 fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

			if nangle_on:
				f.write("\nAngles\n\n")
				angle_count = 0
				for anglei in range(nangle):
					if angles[anglei][0] > 1:
						angle_count += 1
						f.write(f"{angle_count}\t{angles[anglei][0]-1}\t{angles[anglei][1]+1}\t{angles[anglei][2]+1}\t{angles[anglei][3]+1}\n")
			
			f.write("\nCharges\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

			f.write("\nFragments\n\n")
			f.write("1\t7\n")
			f.write("2\t8\n")
			f.write("3\t9\n")
			f.write("4\t7 8 9\n")

		molFile = f"{outReactFold}angleHyb{ri+1:0>{len_nreact_hyb}}_mol_pst.txt"
		with open(molFile, 'w') as f:

			atoms = copy.deepcopy(atoms_all_hyb[ri])
			bonds = copy.deepcopy(bonds_all_hyb[ri])
			angles = copy.deepcopy(angles_all_hyb[ri])

			angles[0][0] += 2
			atoms[7][1] += 2

			nangle_on = 0
			for anglei in range(nangle):
				if angles[anglei][0] > 1:
					nangle_on += 1

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")
			if nangle_on:
				f.write(f"{nangle_on} angles\n")
			f.write(f"4 fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

			if nangle_on:
				f.write("\nAngles\n\n")
				angle_count = 1
				for anglei in range(nangle):
					if angles[anglei][0] > 1:
						angle_count += 1
						f.write(f"{angle_count}\t{angles[anglei][0]-1}\t{angles[anglei][1]+1}\t{angles[anglei][2]+1}\t{angles[anglei][3]+1}\n")
			
			f.write("\nCharges\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

			f.write("\nFragments\n\n")
			f.write("1\t7\n")
			f.write("2\t8\n")
			f.write("3\t9\n")
			f.write("4\t7 8 9\n")

		mapFile = f"{outReactFold}angleHyb{ri+1:0>{len_nreact_hyb}}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Hybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"{nangle+2} constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write("2\n")
			f.write("5\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges_all_hyb[ri][edgei]+1}\n")
			
			f.write("\nConstraints\n\n")
			f.write(f"custom \"round(rxnsum(v_varQ,{angles_all_hyb[ri][0][4]+1})) == {angles_all_hyb[ri][0][0]+1} || "
							 f"round(rxnsum(v_varQ,{angles_all_hyb[ri][0][4]+1})) == {angles_all_hyb[ri][0][0]+3}\"\n")
			if nangle >= 2:
				f.write(f"custom \"round(rxnsum(v_varQ,{angles_all_hyb[ri][1][4]+1})) == {angles_all_hyb[ri][1][0]+1}\"\n")
			if nangle >= 3:
				f.write(f"custom \"round(rxnsum(v_varQ,{angles_all_hyb[ri][2][4]+1})) == {angles_all_hyb[ri][2][0]+1}\"\n")
			f.write("distance 1 6 0.0 2.0\n")
			f.write("distance 3 4 0.0 2.0\n")

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

	#-------- write dehybridization files --------#

	for ri in range(nreact_dehyb):

		natom = len(atoms_all_dehyb[ri])
		nbond = len(bonds_all_dehyb[ri])
		nangle = len(angles_all_dehyb[ri])
		nedge = len(edges_all_dehyb[ri])

		if p.debug:
			print(f"Angle template {ri+1} (dehybridization):")
			print(atoms_all_dehyb[ri])
			print(bonds_all_dehyb[ri])
			print(angles_all_dehyb[ri])
			print(edges_all_dehyb[ri])
			print()

		molFile = f"{outReactFold}angleDehyb{ri+1:0>{len_nreact_dehyb}}_mol.txt"
		with open(molFile, 'w') as f:

			atoms = atoms_all_dehyb[ri]
			bonds = bonds_all_dehyb[ri]
			angles = angles_all_dehyb[ri]

			f.write("## Dehybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")
			f.write(f"4 fragments\n")

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
			f.write("1\t7\n")
			f.write("2\t8\n")
			f.write("3\t9\n")
			f.write("4\t7 8 9\n")

		mapFile = f"{outReactFold}angleDehyb{ri+1:0>{len_nreact_dehyb}}_map.txt"
		with open(mapFile, 'w') as f:

			f.write("## Dehybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"{nangle} constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write("2\n")
			f.write("5\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges_all_dehyb[ri][edgei]+1}\n")

			f.write("\nConstraints\n\n")
			f.write(f"custom \"round(rxnsum(v_varQ,{angles_all_hyb[ri][0][4]+1})) == {angles_all_dehyb[ri][0][0]+1} || "
							 f"round(rxnsum(v_varQ,{angles_all_hyb[ri][0][4]+1})) == {angles_all_dehyb[ri][0][0]+3}\"\n")
			if nangle >= 2:
				f.write(f"custom \"round(rxnsum(v_varQ,{angles_all_dehyb[ri][1][4]+1})) == {angles_all_dehyb[ri][1][0]+1} || "
								 f"round(rxnsum(v_varQ,{angles_all_dehyb[ri][1][4]+1})) == {angles_all_dehyb[ri][1][0]+3}\"\n")
			if nangle >= 3:
				f.write(f"custom \"round(rxnsum(v_varQ,{angles_all_dehyb[ri][2][4]+1})) == {angles_all_dehyb[ri][2][0]+1} || "
								 f"round(rxnsum(v_varQ,{angles_all_dehyb[ri][2][4]+1})) == {angles_all_dehyb[ri][2][0]+3}\"\n")

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

	### return reaction counts
	return nreact_hyb, nreact_dehyb


### write table for hybridization bond
def writeBondHyb(bondFold, bond_res, r12_max, p):
	bondFile = bondFold + "bond_hyb.txt"
	npoint = int(r12_max/bond_res+1)

	### forced binding force (kcal/mol/nm)
	F_force = 1

	with open(bondFile, 'w') as f:
		f.write(f"hyb\n")
		f.write(f"N {npoint}\n\n")
		f.write("# r E(r) F(r)\n")
		
		for i in range(npoint):
			r12 = i * r12_max / (npoint - 1)
			if r12 < p.r12_cut_hyb:
				U = p.U_hyb*(i*bond_res/p.r12_cut_hyb-1)
				F = -p.U_hyb/p.r12_cut_hyb
			elif p.forceBind:
				U = 6.96*F_force*(i*bond_res-p.r12_cut_hyb)
				F = -6.96*F_force
			else:
				U = 0
				F = 0
			f.write(f"{i + 1} {r12:.4f} {U:.4f} {F:.4f}\n")


################################################################################
### Calculation Managers

### initialize positions, keeping strands together
def initPositions(strands, p):
	print("Initializing positions...")

	### parameters
	max_nfail_strand = 20
	max_nfail_bead = 20
	dbox_scaf = p.dbox*0.8

	### initializations
	nbead_placed = 1
	nbead_locked = 0
	nstrand_locked = 0
	r = np.zeros((p.nbead,3))

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
					print(f"Placed strand {nstrand_locked} after {nfail_strand} failures.")

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

### renumber atoms starting from 0 (tailored for hyb bonds)
def renumberBond(atoms, bonds, edges):
	atoms_bi = [row[2] for row in atoms]
	for bondi in range(len(bonds)):
		for i in range(1,len(bonds[bondi])):
			bonds[bondi][i] = atoms_bi.index(bonds[bondi][i])
	for edgei in range(len(edges)):
		edges[edgei] = atoms_bi.index(edges[edgei])
	for atomi in range(len(atoms)):
		atoms[atomi][2] = atomi
	return atoms, bonds, edges


### renumber atoms starting from 0 (tailored for hyb angles)
def renumberAngle(atoms, bonds, angles, edges):
	atoms_bi = [row[2] for row in atoms]
	for bondi in range(len(bonds)):
		for i in range(1,len(bonds[bondi])):
			bonds[bondi][i] = atoms_bi.index(bonds[bondi][i])
	for anglei in range(len(angles)):
		for i in range(1,len(angles[anglei])-1):
			angles[anglei][i] = atoms_bi.index(angles[anglei][i])
	for edgei in range(len(edges)):
		edges[edgei] = atoms_bi.index(edges[edgei])
	for atomi in range(len(atoms)):
		atoms[atomi][2] = atomi
	return atoms, bonds, angles, edges


### remove duplicate elements from array along first dimension
def removeDuplicateEntries(array):
	seen = []
	n_unique = 0
	for i in range(len(array)):
		element_test = array[i]
		new = True
		for j in range(n_unique):
			element_seen = seen[j]
			if element_test == element_seen:
				new = False
				break
		if new:
			n_unique += 1
			seen.append(element_test)
	return seen


### unzip three zipped arrays
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


### return 5p and 3p neighbors for scaffold bead
def getScafNeighbors(bi, backbone_neighbors, complements):

	### for vast majority of cases, this is the result
	bi_5p = backbone_neighbors[bi][0]
	bi_3p = backbone_neighbors[bi][1]

	### check for 5' side break in scaffold, adjust accordingly
	if bi_5p == -1:
		if complements[bi] != -1:
			if backbone_neighbors[complements[bi]][1] != -1:
				if complements[backbone_neighbors[complements[bi]][1]] != -1:
					bi_5p = complements[backbone_neighbors[complements[bi]][1]]

	### check for 3' side break in scaffold, adjust accordingly
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
		print("Error: origami bead index not defined for dummy atoms.")


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

