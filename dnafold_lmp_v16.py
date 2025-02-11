import arsenal_deployed as ars
import parameters
import numpy as np
import copy
import json
import sys
import os

## Description
# this script takes a cadnano json file, creates the interaction and geometry
  # arrays necessary for the dnafold model, and writes the geometry and input
  # files necessary to simulate the system in lammps

## Version Note
# added capabilities: parameters output file, and the return of forced binding!

# To Do
# add blocking cases for angle template that arise from using multiple staple copies,
  # add dummy bonds between scaffold ends and opposing dummy atoms and remove dummy
  # bond between scaffold ends (this dummy bond is being replaced by real backbone bond
  # after hybridization), check dehybridization (angles are being deleted, but that 
  # may not be an issue), input file


################################################################################
### Parameters

def main():

	### simulation ID
	simID = "2HB"
	simTag = "_v16"

	### source folders
	inSrcFold = getSrcFold(simID)
	outSrcFold = "/Users/dduke/Files/dnafold_lmp/"

	### file parameters
	inCadFile = inSrcFold + simID + ".json"
	outSimFold = outSrcFold + simID + simTag + "/"
	outCadFile = outSimFold + simID + ".json"
	outGeoFile = outSimFold + "geometry.in"
	outGeoVisFile = outSimFold + "geometry_vis.in"
	outParamsFile = outSimFold + "parameters.txt"
	outLammpsFile = outSimFold + "lammps.in"
	outReactFold = outSimFold + "react/"
	bondName_hyb = "hyb"
	debug = True

	### prepare folder
	ars.createSafeFold(outSimFold)
	ars.createEmptyFold(outReactFold)

	### computational parameters
	nstep			= 4E6		#steps		- number of simulation steps
	dump_every		= 1E4		#steps		- number of steps between positions dumps
	react_every		= 1E3		#steps		- number of steps between reactions
	dt				= 0.01		#ns			- time step
	dbox			= 40		#nm			- periodic boundary diameter
	verlet_skin		= 4			#nm			- width of neighbor list skin (= r12_cut - sigma)
	neigh_every		= 10		#steps		- how often to consider updating neighbor list
	bond_res 		= 0.1		#nm			- distance between tabular bond interpolation points
	nstep_scaf		= 1E4		#steps		- number of steps for scaffold relaxation
	force_bind		= False		#bool		- whether to force hybridization (not applied if >1 staple copies)
	dehyb_bond		= False		#bool		- whether to include dehybridization bond reactions
	dehyb_angle		= False		#bool		- whether to include dehybridization angle reactions

	### design parameters
	nnt_per_bead	= 8			#nt			- nucleotides per bead (only 8)
	circular_scaf	= False		#bool		- whether the scaffold is circular
	staple_copies	= 1			#int		- number of copies for each staple

	### physical parameters
	kB				= 0.0138	#pN*nm/K	- Boltzmann constant
	T				= 300		#K			- temperature
	r_h_bead		= 1.28		#nm			- hydrodynamic radius of single bead
	visc			= 0.8472	#mPa/s		- viscosity (pN*ns/mn^2)

	### interaction parameters
	sigma			= 2.14		#nm			- bead van der Walls radius
	epsilon			= 6.96		#pN*nm		- WCA energy parameter
	r12_eq			= 2.72		#nm			- equilibrium bead separation
	k_x				= 152		#pN/nm		- backbone spring constant
	r12_cut_hyb		= 2.0		#nm			- hybridization potential cutoff radius
	U_hyb			= 10 		#kcal/mol	- depth of hybridization potential
	dsLp			= 50		#nm			- persistence length of dsDNA

	### create parameters class
	p = parameters.parameters( 	debug, nstep, dump_every, react_every, dt, dbox, verlet_skin, neigh_every, bond_res,
								nstep_scaf, force_bind, dehyb_bond, dehyb_angle, nnt_per_bead, circular_scaf, staple_copies,
								kB, T, r_h_bead, visc, sigma, epsilon, r12_eq, k_x, r12_cut_hyb, U_hyb, dsLp)
	
	### record metadata
	p.record(outParamsFile)
	os.system(f"cp \"{inCadFile}\" \"{outCadFile}\"")

	### parse cadnano
	strand_index, backbone_bonds, comp_hyb_bonds, is_crossover, p = build_dnafold_model(inCadFile, p)

	### write simulation and visualization geometry files
	r, nhyb, nangle = compose_geo(outGeoFile, strand_index, backbone_bonds, comp_hyb_bonds, is_crossover, p)
	compose_geo_vis(outGeoVisFile, strand_index, backbone_bonds, r, p)

	### write bond react files
	nreact_bond = write_react_bond(outReactFold, backbone_bonds, comp_hyb_bonds, is_crossover, p)

	### write angle react files
	nreact_angle_hyb, nreact_angle_dehyb = write_react_angle(outReactFold, strand_index, backbone_bonds, comp_hyb_bonds, is_crossover, p)

	### write table for hybridization bond
	write_bond_hyb(outSimFold, bondName_hyb, p)

	### write lammps input file
	write_input(outLammpsFile, bondName_hyb, is_crossover, nhyb, nangle, nreact_bond, nreact_angle_hyb, nreact_angle_dehyb, p)


################################################################################
### File Managers

### list of possible source folders given 
def getSrcFold(simID):

	### the root of all designs
	projects = "/Users/dduke/OneDrive - Duke University/DukeU/Research/Projects/"

	### folders with useful designs
	if simID.startswith("4HB"):
		inSrcFold = projects + "elementary/cadnano/4HB/"
	elif simID.startswith("ds_"):
		inSrcFold = projects + "elementary/cadnano/strands/"
	elif simID.startswith("cube_4HB"):
		inSrcFold = projects + "gang_cube/full_cube/designs/4HB/"
	elif simID.startswith("triS_edit"):
		inSrcFold = projects + "baigl_isotherms/sharp_triangle_small/"

	### default folder
	else:
		inSrcFold = projects + "elementary/cadnano/"
	return inSrcFold


### extract necessary info from json file
def parse_json(inFile):
	print("Parsing JSON file...")
	
	### load JSON file
	ars.testFileExist(inFile,"json")
	with open(inFile, 'r') as f:
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
	
	### report
	print(f"Found {nnt_scaf} scaffold nucleotides and {nnt_stap} staple nucleotides.")
	return scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap


### write lammps geometry file, for simulation
def compose_geo(outGeoFile, strand_index, backbone_bonds, comp_hyb_bonds, is_crossover, p):
	print("Writing simulation geometry file...")

	### initailize positions
	r = init_positions(strand_index, p)
	natom_real = r.shape[0]
	r = np.append(r,np.zeros((p.n_scaf,3)),axis=0)
	natom = r.shape[0]

	### initialize
	molecules = np.zeros(natom,dtype=int)
	types = np.zeros(natom,dtype=int)
	charges = np.zeros(natom)
	bonds = np.zeros((0,3))
	angles = np.zeros((0,4))
	nhyb = 0
	nangle = 0

	### prepare charges
	len_n_scaf = len(str(p.n_scaf))
	charge_step = 1/(10**len_n_scaf)

	### scaffold atoms
	for bi in range(p.n_scaf):
		molecules[bi] = strand_index[bi] + 1
		types[bi] = min([1,strand_index[bi]]) + 1
		charges[bi] = charge_step*(bi+1)

	### staple atoms
	for ci in range(p.staple_copies):
		for bi in range(p.n_scaf,p.nbead):
			ai = bi + ci*p.n_stap
			nstap = max(strand_index)
			types[ai] = min([1,strand_index[bi]]) + 1
			molecules[ai] = strand_index[bi] + ci*nstap + 1
			if comp_hyb_bonds[bi] != -1:
				if ci == 0:
					nhyb += 1
				charges[ai] = charge_step*(comp_hyb_bonds[bi]+1)

	### dummy atoms
	for bi in range(p.n_scaf):
		ai = bi + natom_real
		molecules[ai] = 0
		types[ai] = 3
		if bi < p.n_scaf:
			charges[ai] = is_crossover[bi] + 1

	### scaffold backbone bonds
	for bi in range(p.n_scaf-1):
		type = 1
		atom1 = bi + 1
		atom2 = bi + 2
		bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### scaffold end-to-end bond
	if p.circular_scaf:
		bonds = np.append(bonds,[[1,p.n_scaf,1]],axis=0)
	else:
		if get_scaf_neighbors(p.n_scaf-1,backbone_bonds,comp_hyb_bonds)[1] == 0:
			# bonds = np.append(bonds,[[3,1,p.n_scaf]],axis=0)
			bonds = np.append(bonds,[[3,1,p.n_scaf+natom_real]],axis=0)
			bonds = np.append(bonds,[[3,p.n_scaf,1+natom_real]],axis=0)

	### staple backbone bonds
	for ci in range(p.staple_copies):
		for bi in range(p.n_scaf,p.nbead):
			if backbone_bonds[bi][1] != -1:
				type = 1
				atom1 = bi + ci*p.n_stap + 1
				atom2 = backbone_bonds[bi][1] + ci*p.n_stap + 1
				bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### dummy bonds
	for bi in range(p.n_scaf):
		type = 3
		atom1 = bi + 1
		atom2 = bi + natom_real + 1
		bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### hybridization bonds
	if p.force_bind:
		for bi in range(p.n_scaf):
			if comp_hyb_bonds[bi] != -1:
				type = 2
				atom1 = bi + 1
				atom2 = comp_hyb_bonds[bi] + 1
				bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### scaffold angles
	for bi in range(p.n_scaf):
		bi_5p,bi_3p = get_scaf_neighbors(bi,backbone_bonds,comp_hyb_bonds)
		if bi_5p != -1 and bi_3p != -1:
			if comp_hyb_bonds[bi_5p] != -1 and comp_hyb_bonds[bi] != -1 and comp_hyb_bonds[bi_3p] != -1:
				nangle += 1

	### write file
	ars.write_geo(outGeoFile, p.dbox, r, molecules, types, bonds, "none", nangleType=4, charges=charges)

	### return positions (without dummy atoms)
	r = r[0:natom_real]
	return r, nhyb, nangle


### write lammps geometry file, for visualization
def compose_geo_vis(outGeoFile, strand_index, backbone_bonds, r, p):

	### count atoms
	natom = r.shape[0]

	### initialize
	molecules = np.zeros(natom,dtype=int)
	types = np.zeros(natom,dtype=int)
	bonds = np.zeros((0,3))

	### compile atom information
	for ai in range(natom):
		bi = ai2bi(ai, p)
		types[ai] = strand_index[bi] + 1

	### compile bond information
	for bi in range(p.n_scaf):
		if backbone_bonds[bi][1] != -1:
			type = 1
			atom1 = bi + 1
			atom2 = backbone_bonds[bi][1] + 1
			bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)
	for ci in range(p.staple_copies):
		for bi in range(p.n_scaf,p.nbead):
			if backbone_bonds[bi][1] != -1:
				type = 1
				atom1 = bi + ci*p.n_stap + 1
				atom2 = backbone_bonds[bi][1] + ci*p.n_stap + 1
				bonds = np.append(bonds,[[type,atom1,atom2]],axis=0)

	### write file
	ars.write_geo(outGeoFile, p.dbox, r, molecules, types, bonds, "none")


### write lammps input file for lammps
def write_input(outLammpsFile, bondName_hyb, is_crossover, nhyb, nangle, nreact_bond, nreact_angle_hyb, nreact_angle_dehyb,  p):
	print("Writing input file...")

	### bond file calculations
	r12_cut_bond = np.sqrt(3)*p.dbox
	r12_cut_bond = r12_cut_bond - r12_cut_bond%p.bond_res + p.bond_res
	npoint_bond = int(r12_cut_bond/p.bond_res+1)

	### count digits
	len_nreact_angle_hyb = len(str(nreact_angle_hyb))
	len_nreact_angle_dehyb = len(str(nreact_angle_dehyb))

	### hybridization bond reaction parameters
	react_every_bond = 10
	r12_cut_react_bond = 4

	### open file
	with open(outLammpsFile, 'w') as f:

		### header
		f.write(
			"\n#------ Begin Input ------#\n"
			"# Written by dnafold_lmp_v16.py\n\n")

		### basic setup
		f.write(
			"## Initialization\n"
			"units           nano\n"
			"dimension       3\n"
			"boundary        p p p\n"
			"atom_style      full\n"
			"read_data       geometry.in &\n"
			"                extra/bond/per/atom 100 &\n"
			"                extra/angle/per/atom 100 &\n"
			"                extra/special/per/atom 100\n\n")

		### neighbor list
		f.write(
			"## System Definition\n"
		   f"neighbor        {p.verlet_skin} bin\n"
		   f"neigh_modify    every {int(p.neigh_every)}\n")

		### pairwise interactions
		f.write(
		   f"pair_style      hybrid zero {round(p.r12_cut_WCA,2)} lj/cut {round(p.r12_cut_WCA,2)}\n"
			"pair_modify     pair lj/cut shift yes\n"
		   f"pair_coeff      * * lj/cut {p.epsilon} {p.sigma} {round(p.r12_cut_WCA,2)}\n"
		    "pair_coeff      * 3 zero\n"
		    "special_bonds   lj 0.0 1.0 1.0\n")

		### bonded interactions
		f.write(
		   f"bond_style      hybrid zero harmonic table linear {npoint_bond}\n"
		   f"bond_coeff      1 harmonic {p.k_x} {p.r12_eq}\n"
		   f"bond_coeff      2 table bond_{bondName_hyb}.txt {bondName_hyb}\n"
		    "bond_coeff      3 zero\n")

		### angled interactions
		if nangle:
			f.write(
			   f"angle_style     hybrid harmonic zero\n"
			   f"angle_coeff     1 zero\n"
			   f"angle_coeff     2 zero\n"
			   f"angle_coeff     3 harmonic {round(p.k_theta/2,2)} 180\n"
			   f"angle_coeff     4 harmonic {round(p.k_theta/2,2)} 90\n")

		### group atoms
		f.write(
			"group           scaffold type 1\n"
			"group           real type 1 2\n"
			"variable        var1 atom q\n\n")

		### relax scaffold
		f.write(
			"## Scaffold Relaxation\n"
		   f"fix             tstat1 scaffold langevin {p.T} {p.T} {round(1/p.gamma_t,4)} 37\n"
			"fix             tstat2 scaffold nve\n"
		   f"timestep        {p.dt/10}\n"
		   f"run             {int(p.nstep_scaf)}\n"
			"reset_timestep  0\n"
			"unfix           tstat1\n"
			"unfix           tstat2\n\n")

		### reactions
		if nreact_angle_hyb or nreact_bond:

			### molecule template header
			f.write(
				"## Reaction Templates\n")

			### bond templates
			for ri in range(nreact_bond):
				if not p.force_bind:
					f.write(
			   f"molecule        bondHyb{ri+1}_mol_pre react/bondHyb{ri+1}_mol_pre.txt\n"
			   f"molecule        bondHyb{ri+1}_mol_pst react/bondHyb{ri+1}_mol_pst.txt\n")

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
			if p.dehyb_bond:
				f.write(
			   f"fix             bondDehyb all bond/break {int(react_every_bond)} 2 {r12_cut_react_bond:.1f}\n")

			### bond hybridization reactions
			f.write(
				"fix             reactions all bond/react reset_mol_ids no")
			if not p.force_bind:
				for ri in range(nreact_bond):
					f.write(
			   f" &\n                react bondHyb{ri+1} all {int(react_every_bond)} 0.0 {r12_cut_react_bond:.1f} bondHyb{ri+1}_mol_pre bondHyb{ri+1}_mol_pst react/bondHyb{ri+1}_map.txt")
			
			### angle hybridization reactions
			for ri in range(nreact_angle_hyb):
				f.write(
			   f" &\n                react angleHyb{ri+1:0>{len_nreact_angle_hyb}} all {int(p.react_every)} 0.0 {p.r12_cut_hyb:.1f} angleHyb{ri+1:0>{len_nreact_angle_hyb}}_mol_pre angleHyb{ri+1:0>{len_nreact_angle_hyb}}_mol_pst react/angleHyb{ri+1:0>{len_nreact_angle_hyb}}_map.txt custom_charges 4")
			
			### angle dehybridization reactions
			if p.dehyb_angle:
				for ri in range(nreact_angle_dehyb):
					f.write(
			   f" &\n                react angleDehyb{ri+1:0>{len_nreact_angle_dehyb}} all {int(p.react_every)} {p.r12_cut_hyb:.1f} {r12_cut_bond:.1f} angleDehyb{ri+1:0>{len_nreact_angle_dehyb}}_mol angleDehyb{ri+1:0>{len_nreact_angle_dehyb}}_mol react/angleDehyb{ri+1:0>{len_nreact_angle_dehyb}}_map.txt custom_charges 2")
			f.write("\n\n")

		### setup simulation
		f.write(
			"## Simulation Setup\n"
		   f"fix             tstat1 real langevin {p.T} {p.T} {round(1/p.gamma_t,4)} 37\n"
			"fix             tstat2 real nve\n"
		   f"dump            dump1 real custom {int(p.dump_every)} trajectory.dat id mol xs ys zs\n"
			"dump_modify     dump1 sort id\n"
		   f"timestep        {p.dt}\n\n")

		### binding updates
		f.write(
			"## Binding Updates\n"
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

		### debugging
		if p.debug:
			f.write(
				"## Debugging Output\n"
				"compute         compD1a all bond/local dist engpot\n"
				"compute         compD1b all property/local btype batom1 batom2\n"
			   f"dump            dumpD1 all local {int(p.dump_every)} dump_bonds.dat index c_compD1a[1] c_compD1a[2] c_compD1b[1] c_compD1b[2] c_compD1b[3] \n"
				"compute         compD2a all angle/local theta eng\n"
				"compute         compD2b all property/local atype aatom1 aatom2 aatom3\n"
			   f"dump            dumpD2 all local {int(p.dump_every)} dump_angles.dat index c_compD2a[1] c_compD2a[2] c_compD2b[1] c_compD2b[2] c_compD2b[3] c_compD2b[4]\n"
			   f"dump            dumpD3 all custom {int(p.dump_every)} dump_charges.dat id q\n"
				"dump_modify     dumpD3 sort id\n")
			f.write("\n")

		### run
		f.write(
			"## Go Time\n"
		   f"run             {int(p.nstep)}\n\n")


### write reaction files for hybridization angles
def write_react_bond(outReactFold, backbone_bonds, comp_hyb_bonds, is_crossover, p):
	print("Writing bond react files...")

	### initialize
	atoms_all = []
	bonds_all = []
	angles_all = []
	edges_all = []

	### loop over scaffold beads
	for bi in range(p.n_scaf):

		### initialize
		comp_5p = False
		comp_3p = False

		### get neighbors to central scaffold bead
		b_5p,b_3p = get_scaf_neighbors(bi,backbone_bonds,comp_hyb_bonds)

		### skip if central scaffold bead is not complimentary
		if comp_hyb_bonds[bi] == -1:
			continue

		#-------- working from 5' scaffold end --------#

		### intialize
		b = bi
		bC = comp_hyb_bonds[b]
		bD = b + p.n_scaf + p.n_stap*p.staple_copies

		### core topology
		atoms_5to3 =  [ [0,-1,b], [1,-1,bC], [2,int(is_crossover[b]),bD] ]
		bonds_5to3 =  [ [2,b,bD] ]
		angles_5to3 = [ ]
		edges_5to3 =  [ ]

		### add angle
		if b_5p != -1 and b_3p != -1:
			angles_5to3.append([int(is_crossover[b]),b_5p,b,b_3p])

		### add central scaffold 5' side topology
		if b_5p != -1:
			atoms_5to3.append([0,-1,b_5p])
			edges_5to3.append(b_5p)
			if b_5p == backbone_bonds[b][0]:
				bonds_5to3.append([0,b_5p,b])
			else:
				b_5p_D = b_5p + p.n_scaf + p.n_stap*p.staple_copies
				atoms_5to3.append([2,int(is_crossover[b_5p]),b_5p_D])
				bonds_5to3.append([2,b_5p,b_5p_D])
				bonds_5to3.append([2,b,b_5p_D])
				bonds_5to3.append([2,b_5p,bD])

		### add central scaffold 3' side topology
		if b_3p != -1:
			atoms_5to3.append([0,-1,b_3p])
			edges_5to3.append(b_3p)
			if b_3p == backbone_bonds[b][1]:
				bonds_5to3.append([0,b,b_3p])
			else:
				b_3p_D = b_3p + p.n_scaf + p.n_stap*p.staple_copies
				atoms_5to3.append([2,int(is_crossover[b_3p]),b_3p_D])
				bonds_5to3.append([2,b_3p,b_3p_D])
				bonds_5to3.append([2,b,b_3p_D])
				bonds_5to3.append([2,b_3p,bD])

		### add central staple 5' end topology
		bC_5p = backbone_bonds[bC][0]
		if bC_5p != -1:
			atoms_5to3.append([1,-1,bC_5p])
			bonds_5to3.append([0,bC_5p,bC])
			edges_5to3.append(bC_5p)

		### add central staple 3' end topology
		bC_3p = backbone_bonds[bC][1]
		if bC_3p != -1:
			atoms_5to3.append([1,-1,bC_3p])
			bonds_5to3.append([0,bC,bC_3p])
			edges_5to3.append(bC_3p)

		### add central scaffold 5' end hybridization bond
		if b_5p != -1 and bC_3p != -1 and bC_3p == comp_hyb_bonds[b_5p]:
			comp_5p = True
			bonds_5to3.append([1,b_5p,bC_3p])

		### add central scaffold 3' end hybridization bond
		if b_3p != -1 and bC_5p != -1 and bC_5p == comp_hyb_bonds[b_3p]:
			comp_3p = True
			bonds_5to3.append([1,b_3p,bC_5p])

		#-------- working from 3' scaffold end --------#

		### intialize
		b = bi
		bC = comp_hyb_bonds[b]
		bD = b + p.n_scaf + p.n_stap*p.staple_copies

		### core topology
		atoms_3to5 =  [ [0,-1,b], [1,-1,bC], [2,int(is_crossover[b]),bD] ]
		bonds_3to5 =  [ [2,b,bD] ]
		angles_3to5 = [ ]
		edges_3to5 =  [ ]

		### add angle
		if b_3p != -1 and b_5p != -1:
			angles_3to5.append([int(is_crossover[b]),b_3p,b,b_5p])

		### add central scaffold 3' side topology
		if b_3p != -1:
			atoms_3to5.append([0,-1,b_3p])
			edges_3to5.append(b_3p)
			if b_3p == backbone_bonds[b][1]:
				bonds_3to5.append([0,b_3p,b])
			else:
				b_3p_D = b_3p + p.n_scaf + p.n_stap*p.staple_copies
				atoms_3to5.append([2,int(is_crossover[b_3p]),b_3p_D])
				bonds_3to5.append([2,b_3p,b_3p_D])
				bonds_3to5.append([2,b,b_3p_D])
				bonds_3to5.append([2,b_3p,bD])

		### add central scaffold 5' side topology
		if b_5p != -1:
			atoms_3to5.append([0,-1,b_5p])
			edges_3to5.append(b_5p)
			if b_5p == backbone_bonds[b][0]:
				bonds_3to5.append([0,b,b_5p])
			else:
				b_5p_D = b_5p + p.n_scaf + p.n_stap*p.staple_copies
				atoms_3to5.append([2,int(is_crossover[b_5p]),b_5p_D])
				bonds_3to5.append([2,b_5p,b_5p_D])
				bonds_3to5.append([2,b,b_5p_D])
				bonds_3to5.append([2,b_5p,bD])

		### add central staple 3' end topology
		bC_3p = backbone_bonds[bC][1]
		if bC_3p != -1:
			atoms_3to5.append([1,-1,bC_3p])
			bonds_3to5.append([0,bC_3p,bC])
			edges_3to5.append(bC_3p)

		### add central staple 5' end topology
		bC_5p = backbone_bonds[bC][0]
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
		atoms_5to3,bonds_5to3,angles_5to3,edges_5to3 = renumber_bond(atoms_5to3,bonds_5to3,angles_5to3,edges_5to3)
		atoms_3to5,bonds_3to5,angles_3to5,edges_3to5 = renumber_bond(atoms_3to5,bonds_3to5,angles_3to5,edges_3to5)

		### test for symmetry
		abae_zipped = [[a,b,c,d] for a,b,c,d in zip([atoms_5to3,atoms_3to5],[bonds_5to3,bonds_3to5],[angles_5to3,angles_3to5],[edges_5to3,edges_3to5])]
		abae_zipped = remove_duplicate_entries(abae_zipped)
		if len(abae_zipped) == 1:
			symmetric = True
		else:
			symmetric = False

		#-------- add to templates --------#

		atoms_all.append(atoms_5to3)
		bonds_all.append(bonds_5to3)
		angles_all.append(angles_5to3)
		edges_all.append(edges_5to3)
		atoms_all.append(atoms_3to5)
		bonds_all.append(bonds_3to5)
		angles_all.append(angles_3to5)
		edges_all.append(edges_3to5)
		abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angles_all,edges_all)]
		abae_zipped = remove_duplicate_entries(abae_zipped)
		if not symmetric:
			abae_zipped.pop()
		atoms_all,bonds_all,angles_all,edges_all = unzip4(abae_zipped)

		if comp_5p or comp_3p:
			bonds_5to3_copy = bonds_5to3[:-1]
			bonds_3to5_copy = bonds_3to5[:-1]
			atoms_all.append(atoms_5to3)
			bonds_all.append(bonds_5to3_copy)
			angles_all.append(angles_5to3)
			edges_all.append(edges_5to3)
			atoms_all.append(atoms_3to5)
			bonds_all.append(bonds_3to5_copy)
			angles_all.append(angles_3to5)
			edges_all.append(edges_3to5)
			abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angles_all,edges_all)]
			abae_zipped = remove_duplicate_entries(abae_zipped)
			if not symmetric:
				abae_zipped.pop()
			atoms_all,bonds_all,angles_all,edges_all = unzip4(abae_zipped)

		if comp_5p and comp_3p:
			bonds_5to3_copy = bonds_5to3[:-2] + [bonds_5to3[-1]]
			bonds_3to5_copy = bonds_3to5[:-2] + [bonds_3to5[-1]]
			atoms_all.append(atoms_5to3)
			bonds_all.append(bonds_5to3_copy)
			angles_all.append(angles_5to3)
			edges_all.append(edges_5to3)
			atoms_all.append(atoms_3to5)
			bonds_all.append(bonds_3to5_copy)
			angles_all.append(angles_3to5)
			edges_all.append(edges_3to5)
			abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angles_all,edges_all)]
			abae_zipped = remove_duplicate_entries(abae_zipped)
			if not symmetric:
				abae_zipped.pop()
			atoms_all,bonds_all,angles_all,edges_all = unzip4(abae_zipped)

			bonds_5to3_copy = bonds_5to3[:-2]
			bonds_3to5_copy = bonds_3to5[:-2]
			atoms_all.append(atoms_5to3)
			bonds_all.append(bonds_5to3_copy)
			angles_all.append(angles_5to3)
			edges_all.append(edges_5to3)
			atoms_all.append(atoms_3to5)
			bonds_all.append(bonds_3to5_copy)
			angles_all.append(angles_3to5)
			edges_all.append(edges_3to5)
			abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all,bonds_all,angles_all,edges_all)]
			abae_zipped = remove_duplicate_entries(abae_zipped)
			if not symmetric:
				abae_zipped.pop()
			atoms_all,bonds_all,angles_all,edges_all = unzip4(abae_zipped)

	#-------- write hybridization files --------#

	nreact = len(atoms_all)
	len_nreact = len(str(nreact))

	for ri in range(nreact):

		natom = len(atoms_all[ri])
		nbond = len(bonds_all[ri])
		nangle = len(angles_all[ri])
		nedge = len(edges_all[ri])

		if p.debug:
			if ri == 0:
				print("")
			print(f"Bond template {ri+1} (hybridization):")
			print(atoms_all[ri])
			print(bonds_all[ri])
			print(angles_all[ri])
			print(edges_all[ri])
			print("")

		molPreFile = f"{outReactFold}bondHyb{ri+1:0>{len_nreact}}_mol_pre.txt"
		with open(molPreFile,'w') as f:

			atoms = copy.deepcopy(atoms_all[ri])
			bonds = copy.deepcopy(bonds_all[ri])
			angles = copy.deepcopy(angles_all[ri])

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")
			if nangle:
				f.write(f"{nangle} angles\n")
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
		with open(molPstFile,'w') as f:

			atoms = copy.deepcopy(atoms_all[ri])
			bonds = copy.deepcopy(bonds_all[ri])
			angles = copy.deepcopy(angles_all[ri])

			f.write("## Hybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond+1} bonds\n")
			if nangle:
				f.write(f"{nangle} angles\n")
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
		with open(mapFile,'w') as f:

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
			f.write("custom \"rxnsum(v_var1,1) == rxnsum(v_var1,2)\"\n")

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

	### return reaction counts
	return nreact


### write reaction files for hybridization angles
def write_react_angle(outReactFold, strand_index, backbone_bonds, comp_hyb_bonds, is_crossover, p):
	print("Writing angle react files...")

	### initialize
	atoms_all_hyb = []
	bonds_all_hyb = []
	angles_all_hyb = []
	edges_all_hyb = []

	### initialize
	atoms_all_dehyb = []
	bonds_all_dehyb = []
	angles_all_dehyb = []
	edges_all_dehyb = []

	### loop over candidate beads
	for bi in range(p.n_scaf):

		### get neighbors to candidate bead
		bi_5p,bi_3p = get_scaf_neighbors(bi,backbone_bonds,comp_hyb_bonds)

		### skip if core scaffold is not present
		if bi_5p == -1 or bi_3p == -1:
			continue

		### skip if core scaffold is not fully complimentary
		if comp_hyb_bonds[bi_5p] == -1 or comp_hyb_bonds[bi] == -1 or comp_hyb_bonds[bi_3p] == -1:
			continue

		#-------- working from 5' scaffold end --------#

		### intialize
		a = bi_5p
		b = bi
		c = bi_3p
		aC = comp_hyb_bonds[a]
		bC = comp_hyb_bonds[b]
		cC = comp_hyb_bonds[c]
		aD = a + p.n_scaf + p.n_stap*p.staple_copies
		bD = b + p.n_scaf + p.n_stap*p.staple_copies
		cD = c + p.n_scaf + p.n_stap*p.staple_copies

		### core topology
		atoms_5to3 =  [ [0,-1,a], [0,-1,b], [0,-1,c], [1,-1,cC], [1,-1,bC], [1,-1,aC], [2,int(is_crossover[a]),aD], [2,int(is_crossover[b]),bD], [2,int(is_crossover[c]),cD] ]
		bonds_5to3 =  [ [1,a,aC], [1,b,bC], [1,c,cC], [2,a,aD], [2,b,bD], [2,c,cD] ]
		angles_5to3 = [ [int(is_crossover[b]),a,b,c,1] ]
		edges_5to3 =  [ cC, aC, aD, cD ]

		### add core 5' side topology
		if backbone_bonds[b][0] == a:
			bonds_5to3.append([0,a,b])
		else:
			bonds_5to3.append([2,b,aD])
			bonds_5to3.append([2,a,bD])

		### add core 3' side topology
		if backbone_bonds[b][1] == c:
			bonds_5to3.append([0,b,c])
		else:
			bonds_5to3.append([2,b,cD])
			bonds_5to3.append([2,c,bD])

		### add scaffold 5' end topology
		a_5p = get_scaf_neighbors(a, backbone_bonds, comp_hyb_bonds)[0]
		if a_5p != -1:
			if backbone_bonds[a][0] == a_5p:
				atoms_5to3.append([0,-1,a_5p])
				bonds_5to3.append([0,a_5p,a])
				angles_5to3.append([int(is_crossover[a]),a_5p,a,b,0])
				edges_5to3.append(a_5p)
			else:
				a_5p_D = a_5p + p.n_scaf + p.n_stap*p.staple_copies
				atoms_5to3.append([2,int(is_crossover[a_5p]),a_5p_D])
				bonds_5to3.append([2,a,a_5p_D])
				edges_5to3.append(a_5p_D)

		### add scaffold 3' end topology
		c_3p = get_scaf_neighbors(c, backbone_bonds, comp_hyb_bonds)[1]
		if c_3p != -1:
			if backbone_bonds[c][1] == c_3p:
				atoms_5to3.append([0,-1,c_3p])
				bonds_5to3.append([0,c,c_3p])
				angles_5to3.append([int(is_crossover[c]),b,c,c_3p,2])
				edges_5to3.append(c_3p)
			else:
				c_3p_D = c_3p + p.n_scaf + p.n_stap*p.staple_copies
				atoms_5to3.append([2,int(is_crossover[c_3p]),c_3p_D])
				bonds_5to3.append([2,c,c_5p_D])
				edges_5to3.append(c_3p_D)

		### add central staple 5' end topology
		bC_5p = backbone_bonds[bC][0]
		if bC_5p == cC:
			bonds_5to3.append([0,cC,bC])
		elif bC_5p != -1:
			if [1,-1,bC_5p] not in atoms_5to3:
				atoms_5to3.append([1,-1,bC_5p])
				edges_5to3.append(bC_5p)
			bonds_5to3.append([0,bC_5p,bC])

		### add central staple 3' end topology
		bC_3p = backbone_bonds[bC][1]
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
		aC = comp_hyb_bonds[a]
		bC = comp_hyb_bonds[b]
		cC = comp_hyb_bonds[c]
		aD = a + p.n_scaf + p.n_stap*p.staple_copies
		bD = b + p.n_scaf + p.n_stap*p.staple_copies
		cD = c + p.n_scaf + p.n_stap*p.staple_copies

		### core topology
		atoms_3to5 =  [ [0,-1,a], [0,-1,b], [0,-1,c], [1,-1,cC], [1,-1,bC], [1,-1,aC], [2,int(is_crossover[a]),aD], [2,int(is_crossover[b]),bD], [2,int(is_crossover[c]),cD] ]
		bonds_3to5 =  [ [1,a,aC], [1,b,bC], [1,c,cC], [2,a,aD], [2,b,bD], [2,c,cD] ]
		angles_3to5 = [ [int(is_crossover[b]),a,b,c,1] ]
		edges_3to5 =  [ cC, aC, aD, cD ]

		### add core 3' side topology
		if backbone_bonds[b][1] == a:
			bonds_3to5.append([0,a,b])
		else:
			bonds_3to5.append([2,b,aD])
			bonds_3to5.append([2,a,bD])

		### add core 5' side topology
		if backbone_bonds[b][0] == c:
			bonds_3to5.append([0,b,c])
		else:
			bonds_3to5.append([2,b,cD])
			bonds_3to5.append([2,c,bD])

		### add scaffold 3' end topology
		a_3p = get_scaf_neighbors(a, backbone_bonds, comp_hyb_bonds)[1]
		if a_3p != -1:
			if backbone_bonds[a][1] == a_3p:
				atoms_3to5.append([0,-1,a_3p])
				bonds_3to5.append([0,a_3p,a])
				angles_3to5.append([int(is_crossover[a]),a_3p,a,b,0])
				edges_3to5.append(a_3p)
			else:
				a_3p_D = a_3p + p.n_scaf + p.n_stap*p.staple_copies
				atoms_3to5.append([2,int(is_crossover[a_3p]),a_3p_D])
				bonds_3to5.append([2,a,a_3p_D])
				edges_3to5.append(a_3p_D)

		### add scaffold 5' end topology
		c_5p = get_scaf_neighbors(c, backbone_bonds, comp_hyb_bonds)[0]
		if c_5p != -1:
			if backbone_bonds[c][0] == c_5p:
				atoms_3to5.append([0,-1,c_5p])
				bonds_3to5.append([0,c,c_5p])
				angles_3to5.append([int(is_crossover[c]),b,c,c_5p,2])
				edges_3to5.append(c_5p)
			else:
				c_5p_D = c_5p + p.n_scaf + p.n_stap*p.staple_copies
				atoms_3to5.append([2,int(is_crossover[c_5p]),c_5p_D])
				bonds_3to5.append([2,c,c_5p_D])
				edges_3to5.append(c_5p_D)

		### add central staple 3' end topology
		bC_3p = backbone_bonds[bC][1]
		if bC_3p == cC:
			bonds_3to5.append([0,cC,bC])
		elif bC_3p != -1:
			if [1,-1,bC_3p] not in atoms_3to5:
				atoms_3to5.append([1,-1,bC_3p])
				edges_3to5.append(bC_3p)
			bonds_3to5.append([0,bC_3p,bC])

		### add central staple 5' end topology
		bC_5p = backbone_bonds[bC][0]
		if bC_5p == aC:
			bonds_3to5.append([0,bC,aC])
		elif bC_5p != -1:
			if [1,-1,bC_5p] not in atoms_3to5:
				atoms_3to5.append([1,-1,bC_5p])
				edges_3to5.append(bC_5p)
			bonds_3to5.append([0,bC,bC_5p])

		#-------- prepare template for comparison --------#

		### renumber
		atoms_5to3,bonds_5to3,angles_5to3,edges_5to3 = renumber_angle(atoms_5to3,bonds_5to3,angles_5to3,edges_5to3)
		atoms_3to5,bonds_3to5,angles_3to5,edges_3to5 = renumber_angle(atoms_3to5,bonds_3to5,angles_3to5,edges_3to5)

		### test for symmetry
		abae_zipped = [[a,b,c,d] for a,b,c,d in zip([atoms_5to3,atoms_3to5],[bonds_5to3,bonds_3to5],[angles_5to3,angles_3to5],[edges_5to3,edges_3to5])]
		abae_zipped = remove_duplicate_entries(abae_zipped)
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
		abae_zipped = remove_duplicate_entries(abae_zipped)
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
			abae_zipped = remove_duplicate_entries(abae_zipped)
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
				abae_zipped = remove_duplicate_entries(abae_zipped)
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
			abae_zipped = remove_duplicate_entries(abae_zipped)
			if not symmetric:
				abae_zipped.pop()
			atoms_all_hyb,bonds_all_hyb,angles_all_hyb,edges_all_hyb = unzip4(abae_zipped)

		#-------- add to dehybridization templates --------#

		if p.dehyb_angle:
			atoms_all_dehyb.append(atoms_5to3)
			bonds_all_dehyb.append(bonds_5to3)
			angles_all_dehyb.append(angles_5to3)
			edges_all_dehyb.append(edges_5to3)
			atoms_all_dehyb.append(atoms_3to5)
			bonds_all_dehyb.append(bonds_3to5)
			angles_all_dehyb.append(angles_3to5)
			edges_all_dehyb.append(edges_3to5)
			abae_zipped = [[a,b,c,d] for a,b,c,d in zip(atoms_all_dehyb,bonds_all_dehyb,angles_all_dehyb,edges_all_dehyb)]
			abae_zipped = remove_duplicate_entries(abae_zipped)
			if not symmetric:
				abae_zipped.pop()
			atoms_all_dehyb,bonds_all_dehyb,angles_all_dehyb,edges_all_dehyb = unzip4(abae_zipped)

	#-------- write hybridization files --------#

	nreact_hyb = len(atoms_all_hyb)
	nreact_dehyb = len(atoms_all_dehyb)
	len_nreact_hyb = len(str(nreact_hyb))
	len_nreact_dehyb = len(str(nreact_dehyb))

	for ri in range(nreact_hyb):

		natom = len(atoms_all_hyb[ri])
		nbond = len(bonds_all_hyb[ri])
		nangle = len(angles_all_hyb[ri])
		nedge = len(edges_all_hyb[ri])

		if p.debug:
			if ri == 0:
				print("")
			print(f"Angle template {ri+1} (hybridization):")
			print(atoms_all_hyb[ri])
			print(bonds_all_hyb[ri])
			print(angles_all_hyb[ri])
			print(edges_all_hyb[ri])
			print("")

		molFile = f"{outReactFold}angleHyb{ri+1:0>{len_nreact_hyb}}_mol_pre.txt"
		with open(molFile,'w') as f:

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
				for anglei in range(nangle):
					if angles[anglei][0] > 1:
						f.write(f"{anglei+1}\t{angles[anglei][0]+1}\t{angles[anglei][1]+1}\t{angles[anglei][2]+1}\t{angles[anglei][3]+1}\n")
			
			f.write("\nCharges\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

			f.write("\nFragments\n\n")
			f.write("1\t7\n")
			f.write("2\t8\n")
			f.write("3\t9\n")
			f.write("4\t7 8 9\n")

		molFile = f"{outReactFold}angleHyb{ri+1:0>{len_nreact_hyb}}_mol_pst.txt"
		with open(molFile,'w') as f:

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
				for anglei in range(nangle):
					if angles[anglei][0] > 1:
						f.write(f"{anglei+1}\t{angles[anglei][0]+1}\t{angles[anglei][1]+1}\t{angles[anglei][2]+1}\t{angles[anglei][3]+1}\n")
			
			f.write("\nCharges\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

			f.write("\nFragments\n\n")
			f.write("1\t7\n")
			f.write("2\t8\n")
			f.write("3\t9\n")
			f.write("4\t7 8 9\n")

		mapFile = f"{outReactFold}angleHyb{ri+1:0>{len_nreact_hyb}}_map.txt"
		with open(mapFile,'w') as f:

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
			f.write(f"custom \"round(rxnsum(v_var1,{angles_all_hyb[ri][0][4]+1})) == {angles_all_hyb[ri][0][0]+1}\"\n")
			if nangle >= 2:
				f.write(f"custom \"round(rxnsum(v_var1,{angles_all_hyb[ri][1][4]+1})) == {angles_all_hyb[ri][1][0]+1}\"\n")
			if nangle >= 3:
				f.write(f"custom \"round(rxnsum(v_var1,{angles_all_hyb[ri][2][4]+1})) == {angles_all_hyb[ri][2][0]+1}\"\n")
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
			print("")

		molFile = f"{outReactFold}angleDehyb{ri+1:0>{len_nreact_dehyb}}_mol.txt"
		with open(molFile,'w') as f:

			atoms = atoms_all_dehyb[ri]
			bonds = bonds_all_dehyb[ri]
			angles = angles_all_dehyb[ri]

			f.write("## Dehybridization\n")
			f.write(f"{natom} atoms\n")
			f.write(f"{nbond} bonds\n")
			f.write(f"{nangle} angles\n")
			f.write(f"2 fragments\n")

			f.write("\nTypes\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][0]+1}\n")

			f.write("\nBonds\n\n")
			for bondi in range(nbond):
				f.write(f"{bondi+1}\t{bonds[bondi][0]+1}\t{bonds[bondi][1]+1}\t{bonds[bondi][2]+1}\n")

			f.write("\nAngles\n\n")
			for anglei in range(nangle):
				f.write(f"{anglei+1}\t{angles[anglei][0]+1}\t{angles[anglei][1]+1}\t{angles[anglei][2]+1}\t{angles[anglei][3]+1}\n")

			f.write("\nCharges\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atoms[atomi][1]+1}\n")

			f.write("\nFragments\n\n")
			f.write("1\t8\n")
			f.write("2\t7 8 9\n")

		mapFile = f"{outReactFold}angleDehyb{ri+1:0>{len_nreact_dehyb}}_map.txt"
		with open(mapFile,'w') as f:

			f.write("## Dehybridization\n")
			f.write(f"{natom} equivalences\n")
			f.write(f"{nedge} edgeIDs\n")
			f.write(f"1 constraints\n")

			f.write(f"\nInitiatorIDs\n\n")
			f.write("2\n")
			f.write("5\n")

			f.write(f"\nEdgeIDs\n\n")
			for edgei in range(nedge):
				f.write(f"{edges_all_dehyb[ri][edgei]+1}\n")

			f.write("\nConstraints\n\n")
			f.write(f"custom \"round(rxnsum(v_var1,1)) == {angles_all_dehyb[ri][0][0]+2+1}\"\n")

			f.write("\nEquivalences\n\n")
			for atomi in range(natom):
				f.write(f"{atomi+1}\t{atomi+1}\n")

	### return reaction counts
	return nreact_hyb, nreact_dehyb


### write molecule file for hybridization bonds
def write_mol_hyb_bond(molFile, atoms, bonds, angles, frags):
	natom = len(atoms)
	nbond = len(bonds)
	nangle = len(angles)
	nfrag = len(frags)
	with open(molFile,'w') as f:

		f.write("## Hybridization\n")
		f.write(f"{natom} atoms\n")
		f.write(f"{nbond} bonds\n")
		if nangle:
			f.write(f"{nangle} angles\n")
		if nfrag:
			f.write(f"{nfrag} fragments\n")

		f.write("\nTypes\n\n")
		for i in range(natom):
			f.write(f"{i+1}\t{atoms[i]}\n")

		f.write("\nBonds\n\n")
		for i in range(nbond):
			f.write(f"{i+1}\t{bonds[i][0]}\t{bonds[i][1]}\t{bonds[i][2]}\n")

		if nangle:
			f.write("\nAngles\n\n")
			for i in range(nangle):
				f.write(f"{i+1}\t{angles[i][0]}\t{angles[i][1]}\t{angles[i][2]}\t{angles[i][3]}\n")

		if nfrag:
			f.write("\nFragments\n\n")
			for i in range(nfrag):
				f.write(f"{i+1}\t{frags[i]}\n")


### write map file for hybridization bonds
def write_map_hyb_bond(mapFile, atoms, inits, edges):
	natom = len(atoms)
	nedge = len(edges)
	with open(mapFile,'w') as f:

		f.write("## Hybridization\n")
		f.write(f"{natom} equivalences\n")
		f.write(f"{nedge} edgeIDs\n")
		f.write(f"1 constraints\n")

		f.write(f"\nInitiatorIDs\n\n")
		f.write(f"{inits[0]}\n")
		f.write(f"{inits[1]}\n")

		f.write(f"\nEdgeIDs\n\n")
		for i in range(nedge):
			f.write(f"{edges[i]}\n")

		f.write(f"\nConstraints\n\n")
		f.write("custom \"rxnsum(v_var1,1) == rxnsum(v_var1,2)\"\n")

		f.write("\nEquivalences\n\n")
		for i in range(natom):
			f.write(f"{i+1}\t{i+1}\n")


### write table for hybridization bond
def write_bond_hyb(bondFold, bondName, p):
	bondFile = bondFold + "bond_" + bondName + ".txt"
	r12_cut_bond = np.sqrt(3)*p.dbox
	r12_cut_bond = r12_cut_bond - r12_cut_bond%p.bond_res + p.bond_res
	npoint = int(r12_cut_bond/p.bond_res+1)

	### forced binding force (kcal/mol/nm)
	F_force = 1

	with open(bondFile, 'w') as f:
		f.write(f"{bondName}\n")
		f.write(f"N {npoint}\n\n")
		f.write("# r E(r) F(r)\n")
		
		for i in range(npoint):
			r12 = i * r12_cut_bond / (npoint - 1)
			if r12 < p.r12_cut_hyb:
				U = p.U_hyb*(i*p.bond_res/p.r12_cut_hyb-1)
				F = -p.U_hyb/p.r12_cut_hyb
			elif p.force_bind:
				U = F_force*(i*p.bond_res-p.r12_cut_hyb)*6.96
				F = -F_force*6.96
			else:
				U = 0
				F = 0
			f.write(f"{i + 1} {r12:.4f} {U:.4f} {F:.4f}\n")


################################################################################
### Calculation Managers

### translate cadnano design to DNAfold model
def build_dnafold_model(inFile, p):

	### parse the json file
	scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap = parse_json(inFile)
	
	### initial calculations
	print("Building DNAfold model...")
	p.n_scaf = nnt_scaf // p.nnt_per_bead
	p.n_stap = nnt_stap // p.nnt_per_bead
	p.nbead = p.n_scaf + p.n_stap
	print("Using " + str(p.n_scaf) + " scaffold beads and " + str(p.n_stap) + " staple beads.")

	### initialze interaction and geometry arrays
	strand_index =  [0 for i in range(p.nbead)]
	vstrand_index = [0 for i in range(p.nbead)]
	backbone_bonds = [[-1,-1] for i in range(p.nbead)]
	comp_hyb_bonds = [-1 for i in range(p.nbead)]
	is_crossover = [False for i in range(p.nbead)]

	### initialize nucleotide and bead indices
	ni_current = 0
	bi_current = 0

	### kick off nucleotide and bead indexing with 5' scaffold end
	ni_scaffoldArr = find(fiveP_end_scaf[0], fiveP_end_scaf[1], scaffold)
	scaffold[ni_scaffoldArr].extend([ni_current, bi_current])
	vstrand_index[bi_current] = scaffold[ni_scaffoldArr][0]

	### track along scaffold until 3' end eached
	while scaffold[ni_scaffoldArr][4] != -1:
		ni_scaffoldArr = find(scaffold[ni_scaffoldArr][4], scaffold[ni_scaffoldArr][5], scaffold)

		### update nucleotide and bead indices
		ni_current += 1
		bi_current = ni_current // p.nnt_per_bead
		scaffold[ni_scaffoldArr].extend([ni_current, bi_current])

		### store vstrand and backbone bonds for new beads
		if bi_current > (ni_current-1)//p.nnt_per_bead:
			vstrand_index[bi_current] = scaffold[ni_scaffoldArr][0]
			backbone_bonds[bi_current][0] = bi_current-1
			backbone_bonds[bi_current-1][1] = bi_current

	### loop over staples
	nstap = len(fiveP_ends_stap)
	for sti in range(nstap):

		### new nucleotide and bead incides
		ni_current += 1
		bi_current = ni_current // p.nnt_per_bead

		### pick up nucleotide and bead indexing with 5' staple end
		ni_staplesArr = find(fiveP_ends_stap[sti][0],fiveP_ends_stap[sti][1], staples)
		staples[ni_staplesArr].extend([ni_current, bi_current])
		vstrand_index[bi_current] = staples[ni_staplesArr][0]
		strand_index[bi_current] = sti+1

		### identify paired beads
		if scaffold[ni_staplesArr][2] != -1 or scaffold[ni_staplesArr][4] != -1:
			comp_hyb_bonds[scaffold[ni_staplesArr][7]] = bi_current
			comp_hyb_bonds[bi_current] = scaffold[ni_staplesArr][7]

		### track along staple until 3' end eached
		while staples[ni_staplesArr][4] != -1:
			ni_staplesArr = find(staples[ni_staplesArr][4], staples[ni_staplesArr][5], staples)

			### update nucleotide and bead indices
			ni_current += 1
			bi_current = ni_current // p.nnt_per_bead
			staples[ni_staplesArr].extend([ni_current, bi_current])

			### store vstrand, strand, and backbone bonds for new beads
			if bi_current > (ni_current-1)//p.nnt_per_bead:
				strand_index[bi_current] = sti+1
				vstrand_index[bi_current] = scaffold[ni_staplesArr][0]
				backbone_bonds[bi_current][0] = bi_current-1
				backbone_bonds[bi_current-1][1] = bi_current

				### identify paired beads
				if scaffold[ni_staplesArr][2] != -1 or scaffold[ni_staplesArr][4] != -1:
					comp_hyb_bonds[scaffold[ni_staplesArr][7]] = bi_current
					comp_hyb_bonds[bi_current] = scaffold[ni_staplesArr][7]

	### identify crossovers
	for bi in range(1, p.nbead):
		if vstrand_index[bi] != vstrand_index[bi-1]:
			if strand_index[bi] == strand_index[bi-1]:
				is_crossover[bi] = True
				is_crossover[bi-1] = True

	### adjustments for circular scaffold
	if p.circular_scaf:
		backbone_bonds[0][0] = p.n_scaf-1
		backbone_bonds[p.n_scaf-1][1] = 0
		if vstrand_index[0] != vstrand_index[p.n_scaf]:
			is_crossover[0] = True
			is_crossover[p.n_scaf] = True

	### return results			
	return strand_index, backbone_bonds, comp_hyb_bonds, is_crossover, p


### initialize positions, keeping strands together
def init_positions(strand_index, p):
	print("Initializing positions...")
	max_resets = 10
	max_attempts = 10
	scaffold_buffer = 4
	natom = p.n_scaf + p.staple_copies*(p.nbead-p.n_scaf)
	r = np.zeros((natom,3))
	resets = 0
	ai = 1
	while ai < natom:
		attempts = 0
		bi = ai2bi(ai, p)
		while True:
			if strand_index[bi] != strand_index[bi-1]:
				r_propose = ars.rand_pos(p.dbox)
			else:
				r_propose = ars.applyPBC(r[ai-1] + p.r12_eq*ars.unit_vector(ars.box_muller()), p.dbox)
			if not ars.check_overlap(r_propose,r[0:ai],p.sigma,p.dbox):
				if strand_index[bi] != 0:
					break
				elif sum(abs(r_propose)>(p.dbox/2-scaffold_buffer)) == 0:
					break
			if attempts == max_attempts:
				break
			attempts += 1
		if attempts < max_attempts:
			r[ai] = r_propose
			ai += 1
		else:
			resets += 1
			ai = 1
			if resets >= max_resets:
				print("Error: could not place beads, try again with larger box.")
				sys.exit()
	return r


################################################################################
### Utilify Functions

### search for entry in strand/index list that matches given strand/index, return index
def find(strand, index, list):
	for i,item in enumerate(list):
		if item[0] == strand and item[1] == index:
			if item[2] == -1 and item[3] == -1 and item[4] == -1 and item[5] == -1:
				return -1
			return i
	print("Error: index not found, try again.")
	sys.exit()


### renumber atoms starting from 0 (tailored for hyb bonds)
def renumber_bond(atoms, bonds, angles, edges):
	atoms_bi = [row[2] for row in atoms]
	for bondi in range(len(bonds)):
		for i in range(1,len(bonds[bondi])):
			bonds[bondi][i] = atoms_bi.index(bonds[bondi][i])
	for anglei in range(len(angles)):
		for i in range(1,len(angles[anglei])):
			angles[anglei][i] = atoms_bi.index(angles[anglei][i])
	for edgei in range(len(edges)):
		edges[edgei] = atoms_bi.index(edges[edgei])
	for atomi in range(len(atoms)):
		atoms[atomi][2] = atomi
	return atoms,bonds,angles,edges

### renumber atoms starting from 0 (tailored for hyb angles)
def renumber_angle(atoms, bonds, angles, edges):
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
	return atoms,bonds,angles,edges


### remove duplicate elements from array along first dimension
def remove_duplicate_entries(array):
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


### return 5p and 3p neighbors for scaffold bead
def get_scaf_neighbors(bi, backbone_bonds, comp_hyb_bonds):

	### for vast majority of cases, this is the result
	bi_5p = backbone_bonds[bi][0]
	bi_3p = backbone_bonds[bi][1]

	### check for 5' side break in scaffold, adjust accordingly
	if bi_5p == -1:
		if comp_hyb_bonds[bi] != -1:
			if backbone_bonds[comp_hyb_bonds[bi]][1] != -1:
				if comp_hyb_bonds[backbone_bonds[comp_hyb_bonds[bi]][1]] != -1:
					bi_5p = comp_hyb_bonds[backbone_bonds[comp_hyb_bonds[bi]][1]]

	### check for 3' side break in scaffold, adjust accordingly
	if bi_3p == -1:
		if comp_hyb_bonds[bi] != -1:
			if backbone_bonds[comp_hyb_bonds[bi]][0] != -1:
				if comp_hyb_bonds[backbone_bonds[comp_hyb_bonds[bi]][0]] != -1:
					bi_3p = comp_hyb_bonds[backbone_bonds[comp_hyb_bonds[bi]][0]]

	### return result
	return bi_5p,bi_3p


### get bead index from atom index
def ai2bi(ai, p):
	if ai < p.n_scaf or p.nbead == p.n_scaf:
		return ai
	else:
		ci = int((ai-p.n_scaf)/p.n_stap)
		if ci < p.staple_copies:
			return (ai-p.n_scaf)%p.n_stap + p.n_scaf
		else:
			return ai - (p.staple_copies-1)*p.n_stap


### run the script
if __name__ == "__main__":
	main()

