import armament as ars
from scipy import stats
import numpy as np
import pickle
import json
import sys

## Description
# this script contains various functions useful for running and analysing
  # DNAfold simulations.


################################################################################
### File Handlers

### get simulation folders
def getSimFolds(copiesFile=None, simFold=None):

	### use copies file by default
	if copiesFile is not None:
		copyNames = readCopies(copiesFile)[0]
		nsim = len(copyNames)

		### set simulation folders
		simFolds = [ copyNames[i] + "/" for i in range(nsim) ]

		### warning if simulation folder also given
		if simFold is not None:
			print("Flag: Both copies file and simulation folder given, using copies file only.")
	
	### use simulation folder otherwise
	else:
		nsim = 1

		### use given folder
		if simFold is not None:
			simFolds = [ simFold + "/" ]

		### use current folder
		else:
			simFolds = [ "./" ]

	### result
	return simFolds, nsim


### get random seeds
def getRseeds(copiesFile=None, rseed=None):

	### use copies file by default
	if copiesFile is not None:
		rseeds = readCopies(copiesFile)[1]
		nsim = len(rseeds)

		### set random seeds if not copmpletely defined in copies file
		if rseeds.count(None):
			print("Flag: At least one random seed missing in copies file, using range starting from given random seed instead.")
			
			### warning if random seed not given
			if rseed is None:
				print("Flag: Random seed not given, setting to 1.")
				rseed = 1

			### set random seeds
			rseeds = np.arange(rseed, rseed+nsim)

		### warning if rseed also given
		elif rseed is not None:
			print("Flag: Copies file fully defines random seeds, not using given random seed.")
	
	### use random seed otherwise
	else:

		### set random seed if not given
		if rseed is None:
			print("Flag: Random seed not given, setting to 1.")
			rseed = 1

		### set random seeds
		rseeds = [ rseed ]

	### result
	return rseeds


### read file containing simulation folder names and (optionally) random seeds
def readCopies(copiesFile):
	ars.testFileExist(copiesFile, "copies")
	with open(copiesFile, 'r') as f:
		content = f.readlines()
	content = ars.cleanFileContent(content)
	nsim = len(content)
	copyNames = [None]*nsim
	rseeds = [None]*nsim
	for i in range(nsim):
		line = content[i].split()
		copyNames[i] = line[0]
		if len(line) > 1:
			rseeds[i] = int(line[1])
	return copyNames, rseeds


### read oxdna configuration
def readConf(confFile, nba_total):
	ars.testFileExist(confFile, "configuration")
	with open(confFile) as f:
		content = f.readlines()
	dbox3 = [ float(i) for i in content[1].split()[2:] ]
	coms = np.zeros((nba_total,3))
	a1s = np.zeros((nba_total,3))
	for j in range(nba_total):
		line = content[j+3].split()
		for k in range(3):
			coms[j,k] = float(line[k])
			a1s[j,k] = float(line[k+3])
	coms -= np.mean(coms,axis=0)
	return coms, a1s, dbox3


### read oxdna topology
def readTop(topFile):
	ars.testFileExist(topFile, "topology")
	with open(topFile) as f:
		content = f.readlines()
	nba_total = int(content[0].split()[0])
	strands = [int(line.split()[0]) for line in content[1:]]
	return strands, nba_total


### read misbinding cutoffs and energies file
def readMis(misFile):
	ars.testFileExist(misFile, "misbinding")
	with open(misFile) as f:
		content = f.readlines()
	nmisBond = len(content)
	mis_d2_cuts = np.zeros(nmisBond)
	Us_mis = np.zeros(nmisBond)
	for i in range(nmisBond):
		line = content[i].split()
		mis_d2_cuts[i] = line[0]
		Us_mis[i] = line[1]
	return mis_d2_cuts, Us_mis


### read hybridization status file
def readHybStatus(hybFile, nstep_skip=0, coarse_time=1, nstep_max='all', **kwargs):

	### added keyword args
	n_read			= 'all'		if 'n_read' not in kwargs else kwargs['n_read']
	mis_status		= 'none'	if 'mis_status' not in kwargs else kwargs['mis_status']
	getUsedEvery	= False		if 'getUsedEvery' not in kwargs else kwargs['getUsedEvery']

	### notes
	# misbinding status describes how to handle misbonds, specifically whether to count them as 
	  # nothing (none, set to 0), hybridizations (hyb, set to 1), or misbonds (mis, keep at 1.XX)
	
	### load hyb status file
	print("Loading hybridization status...")
	ars.testFileExist(hybFile, "hybridization status")
	with open(hybFile, 'r') as f:
		content = f.readlines()
	print("Parsing hybridization status...")

	### extract metadata
	nbead = 0
	while ars.isnumber(content[nbead+1].split()[0]):
		nbead += 1
	dump_every = int(content[nbead+1].split()[1])
	nstep_recorded = int(len(content)/(nbead+1))
	nstep_trimmed = int((nstep_recorded-nstep_skip-1)/coarse_time)+1
	if nstep_trimmed <= 0:
		print("Error: Cannot read hybridization status - too much initial time cut off.\n")
		sys.exit()

	### interpret input
	if isinstance(nstep_max, str) and nstep_max == 'all':
		nstep_used = nstep_trimmed
	elif ars.isinteger(nstep_max):
		nstep_used = min([nstep_max,nstep_trimmed])
	else:
		print("Error: Cannot read hybridization status - max number of steps must be 'all' or integer.\n")
		sys.exit()
	if isinstance(n_read, str) and n_read == 'all':
		n_read = nbead
	elif not ars.isinteger(n_read):
		print("Error: Cannot read hybridization status - number of beads to read must be 'all' or integer.\n")
		sys.exit()

	### report step counts
	print("{:1.2e} steps in simulation".format(nstep_recorded*dump_every))
	print("{:1.2e} steps recorded".format(nstep_recorded))
	print("{:1.2e} steps used".format(nstep_used))

	### read data
	hyb_status = np.zeros((nstep_used,n_read))
	for i in range(nstep_used):
		for j in range(n_read):
			if j >= nbead:
				print(f"Error: Cannot read hybridization status - requested bead index {j} exceeds the number of beads in the simulation ({nbead}).\n")
				sys.exit()
			hyb_status[i,j] = content[(nbead+1)*(nstep_skip+i*coarse_time)+1+j].split()[1]
		if i%1000 == 0 and i != 0:
			print(f"Processed {i} steps...")

	### adjust misbinding values
	if mis_status == 'hyb':
		hyb_status[hyb_status>1] = 1
	elif mis_status == 'none':
		hyb_status[hyb_status>1] = 0
	elif mis_status != 'mis':
		print("Error: Unrecognized misbinding status.\n")
		sys.exit()

	### result
	output = [ hyb_status ]
	if getUsedEvery: output.append(dump_every*coarse_time)
	if len(output) == 1: output = output[0]
	return output


### extract dump frequency from hyb status file
def getDumpEveryHyb(hybFile):
	ars.testFileExist(hybFile, "hybridization status")
	with open(hybFile, 'r') as f:
		content = f.readlines()

	### extract metadata
	nbead = 0
	while ars.isnumber(content[nbead+1].split()[0]):
		nbead += 1
	dump_every = int(content[nbead+1].split()[1])

	### results
	return dump_every


### prepare basic DNAfold ovito scene
def setOvitoBasics(pipeline):
	from ovito import scene
	from ovito.vis import Viewport

	### disable simulation cell
	vis_element = pipeline.source.data.cell.vis
	vis_element.enabled = False

	### set default particle radius and bond width
	particle_vis = pipeline.source.data.particles.vis
	particle_vis.radius = 1.2
	bonds_vis = pipeline.source.data.particles.bonds.vis
	bonds_vis.width = 2.4

	### first viewport camera
	viewport = scene.viewports[0]
	viewport.type = Viewport.Type.ORTHO
	viewport.camera_dir = (0,0,1)
	viewport.camera_up = (0,-1,0)
	viewport.zoom_all()

	### second viewport camera
	viewport = scene.viewports[1]
	viewport.type = Viewport.Type.ORTHO
	viewport.camera_dir = (-1,0,0)
	viewport.camera_up = (0,-1,0)
	viewport.zoom_all()

	### third viewport camera
	viewport = scene.viewports[2]
	viewport.type = Viewport.Type.PERSPECTIVE
	viewport.camera_dir = (-1,0,0)
	viewport.camera_up = (0,-1,0)
	viewport.zoom_all()

	### fourth viewport camera
	viewport = scene.viewports[3]
	viewport.type = Viewport.Type.PERSPECTIVE
	viewport.camera_dir = (-1,1,1)
	viewport.camera_up = (0,-1,0)
	viewport.zoom_all()

	### results
	return pipeline


################################################################################
### Calculation Managers

### calculate first hybridization time of each scaffold bead
def calcFirstHybTimes(hyb_status, n_scaf):
	nstep = hyb_status.shape[0]
	first_hyb_times = np.ones(n_scaf)*nstep
	for i in range(nstep):
		for j in range(n_scaf):
			if hyb_status[i,j] == 1 and first_hyb_times[j] == nstep:
				first_hyb_times[j] = i
	return first_hyb_times/nstep


### calcualte crystallinity of a chain
def calcCrystallinity(points, dbox):
	nstep = points.shape[0]
	n_scaf = points.shape[1]
	S = np.zeros(nstep)
	for i in range(nstep):
		dX = np.zeros((n_scaf-1,3))
		for j in range(n_scaf-1):
			dX[j] = ars.applyPBC( points[i,j+1] - points[i,j], dbox )
		Q_val = np.zeros((n_scaf-1,3,3))
		for j in range(n_scaf-1):
			dX_mag = np.linalg.norm(dX[j])
			if dX_mag != 0:
				for m in range(3):
					for n in range(3):
						v1 = dX[j,m] / dX_mag
						v2 = dX[j,n] / dX_mag
						Q_val[j,m,n] = 3/2*v1*v2 - 1/2*(i==j)
		Q = np.mean(Q_val, axis=0)
		S[i] = np.max(np.linalg.eigvals(Q).real) - 1/2
	return S


### aligning structures for RMSD
def kabschAlgorithm(r_real, r_ideal, indices='all', getR=False):

	### notes
	# this version of the Kabsch algorithm matches the wikipedia version, which
	  # retains the row-vector convention throughout the calculations; other version
	  # perform the linera algebra with column-vector notation, and others mix them
	  # and produce incorrect results; but rest assured this one is correct.

	### interpret input
	if isinstance(indices, str) and indices == 'all':
		indices = np.arange(len(r_real))
	elif ars.isinteger(indices):
		indices = np.arange(indices)
	elif not ars.isarray(indices) or not ars.isinteger(indices[0]):
		print("Error: indices must be 'auto', integer, or int array.\n")
		sys.exit()

	### enter both sets of points
	com_real = np.mean(r_real[indices], axis=0)
	r_real_centered = r_real - com_real
	com_ideal = np.mean(r_ideal[indices], axis=0)
	r_ideal_centered = r_ideal - com_ideal

	### covariance and SVD
	H = r_real_centered.T @ r_ideal_centered
	U, S, Vt = np.linalg.svd(H)

	### enforce right-handedness
	if np.linalg.det(U@Vt) < 0:
		U[:,-1] *= -1

	### calculate rotation
	R = U @ Vt

	### rotate positions, add ideal center
	r_real_rot = r_real_centered @ R
	r_real_aligned = r_real_rot + com_ideal

	### results
	output = [ r_real_aligned ]
	if getR: output.append(R)
	if len(output) == 1: output = output[0]
	return output


### calculate mean structure of trajectory
def calcMeanStructure(points):
    nstep = points.shape[0]
    nbead = points.shape[1]
    reference = points[0]
    points_aligned = np.zeros_like(points)
    for i in range(nstep):
        points_aligned[i] = kabschAlgorithm(points[i], reference)
    mean_structure = np.mean(points_aligned, axis=0)
    return mean_structure


### calculate RMSD between a trajectory of points and an ideal configuration
def calcRMSD(points, r_ideal):
	nstep = points.shape[0]
	RMSD = np.zeros(nstep)
	for i in range(nstep):
		points_aligned = kabschAlgorithm(points[i], r_ideal)
		RMSD[i] = np.sqrt(np.mean(np.sum((points_aligned - r_ideal) ** 2, axis=1)))
	return RMSD


### initialize positions according to caDNAno positions
def initPositionsCaDNAno(cadFile, scaf_shift=0):
	print("Initializing positions from caDNAno...")

	### parse caDNAno file
	strands, _, row_index, col_index, dep_index = buildDNAfoldModel(cadFile)
	strands = np.array(strands)+1
	n_ori = len(strands)

	### set positions
	row_index = [i - min(row_index) for i in row_index]
	col_index = [i - min(col_index) for i in col_index]
	dep_index = [i - min(dep_index) for i in dep_index]
	row_middle = max(row_index)/2
	col_middle = max(col_index)/2
	dep_middle = max(dep_index)/2
	r = np.zeros((n_ori,3))
	for bi in range(n_ori):
		r[bi,0] = (col_index[bi]-col_middle)*2.4
		r[bi,1] = (row_index[bi]-row_middle)*2.4
		r[bi,2] = (dep_index[bi]-dep_middle)*2.72

	### center about scaffold
	n_scaf = sum(strands==1)
	r -= np.mean(r[:n_scaf], axis=0)

	### shift scaffold
	r[:n_scaf] = np.roll(r[:n_scaf], -scaf_shift, axis=0)

	### results
	return r, strands


### initialize positions according to oxdna configuration file
def initPositionsOxDNA(cadFile, topFile, confFile, scaf_shift=0):
	print("Initializing positions from oxDNA...")
	nnt_bead = 8

	### parse caDNAno file
	strands, complements = buildDNAfoldModel(cadFile)[:2]
	strands = np.array(strands)+1
	complements = np.array(complements)+1
	n_scaf = sum(strands==1)
	n_ori = len(strands)

	### analyze topology
	base_strands, nba_total = readTop(topFile)
	base_strand_scaf = stats.mode(base_strands).mode
	nba_scaf = stats.mode(base_strands).count
	for bai in range(nba_total):
		if base_strands[bai] == base_strand_scaf:
			nba_offset = bai
			break

	### read configuration
	coms, a1s, dbox3 = readConf(confFile, nba_total)

	### get oxDNA scaffold positions
	r = np.zeros((n_ori,3))
	for bi in range(n_scaf):
		bais = [ nba_offset+nba_scaf-bi*nnt_bead-j-1 for j in range(nnt_bead)]
		r[bi] = np.mean( ars.applyPBC( coms[bais]+0.6*a1s[bais], dbox3 ), axis=0) *0.8518

	### get oxDNA staple positions
	n_stap_5p = (nba_total - nba_scaf - nba_offset) // nnt_bead
	n_stap_3p = nba_offset // nnt_bead
	n_stap = n_stap_5p + n_stap_3p
	r_ox_stap = np.zeros((n_stap,3))
	strands_ox_stap = np.zeros(n_stap,dtype=int)
	for bi in range(n_stap_5p):
		bais = [ nba_total-bi*nnt_bead-j-1 for j in range(nnt_bead) ]
		r_ox_stap[bi] = np.mean( ars.applyPBC( coms[bais]+0.6*a1s[bais], dbox3 ), axis=0) *0.8518
		strands_ox_stap[bi] = base_strands[bais[0]]
	for bi in range(n_stap_3p):
		bais = [ nba_offset-bi*nnt_bead-j-1 for j in range(nnt_bead) ]
		r_ox_stap[bi+n_stap_5p] = np.mean( ars.applyPBC( coms[bais]+0.6*a1s[bais], dbox3 ), axis=0) *0.8518
		strands_ox_stap[bi+n_stap_5p] = base_strands[bais[0]]

	### find mapping between DNAfold strand indices and oxDNA strand indices
	nstrand = max(strands)
	strand_ox_to_strand = np.ones(nstrand)
	for scbi in range(n_scaf):
		for stbi in range(n_stap):
			if np.linalg.norm(r[scbi]-r_ox_stap[stbi]) < 1E-3:
				strand_ox = strands_ox_stap[stbi]
				strand = strands[complements[scbi]-1]
				if strand_ox_to_strand[strand_ox-1] == 1:
					strand_ox_to_strand[strand_ox-1] = strand
				elif strand_ox_to_strand[strand_ox-1] != strand:
					print("Error: Conflicting information relating DNAfold strands to oxDNA strands.\n")

	### assign staple positions
	bi = n_scaf
	for strand in range(2,nstrand+1):
		for stbi in range(n_stap):
			if strand_ox_to_strand[strands_ox_stap[stbi]-1] == strand:
				r[bi] = r_ox_stap[stbi]
				bi += 1

	### shift scaffold
	r[:n_scaf] = np.roll(r[:n_scaf], -scaf_shift, axis=0)

	### center about scaffold
	r -= np.mean(r[:n_scaf], axis=0)

	### align with principal components of scaffold
	r = ars.alignPCs(r, n_scaf, [2,1,0])

	### flip such that 5p scaffold end is in back (min x) upper (min y) left (min z) corner
	r *= np.sign(r[0])*[-1,-1,-1]

	### results
	return r, strands


### simulation time in seconds
def getTime(nstep, dump_every, dt=0.01, scale=5200):
	return np.arange(nstep)*dump_every*dt*scale*1E-9



################################################################################
### DNAfold

### translate caDNAno design to DNAfold model
def buildDNAfoldModel(cadFile):

	### parse the caDNAno file
	scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap, vstrand_rows, vstrand_cols = parseCaDNAno(cadFile)

	### initial calculations
	print("Building DNAfold model...")
	nnt_bead = 8
	n_scaf = nnt_scaf // nnt_bead
	n_stap = nnt_stap // nnt_bead
	n_ori = n_scaf + n_stap
	print("Using " + str(n_scaf) + " scaffold beads and " + str(n_stap) + " staple beads.")

	### initialze interaction and geometry arrays
	strands = [0 for i in range(n_ori)]
	backbone_neighbors = [[-1,-1] for i in range(n_ori)]
	complements = [-1 for i in range(n_ori)]
	row_index = [0 for i in range(n_ori)]
	col_index = [0 for i in range(n_ori)]
	dep_index = [0 for i in range(n_ori)]

	### initialize nucleotide and bead indices
	ni_current = 0
	bi_current = 0

	### kick off nucleotide and bead indexing with 5' scaffold end
	ni_scaffoldArr = find(fiveP_end_scaf[0], fiveP_end_scaf[1], scaffold)
	scaffold[ni_scaffoldArr].extend([ni_current, bi_current])
	vstrand = scaffold[ni_scaffoldArr][0]
	vstrand_prev = vstrand

	### gather 3D positions
	row_index[bi_current] = vstrand_rows[vstrand]
	col_index[bi_current] = vstrand_cols[vstrand]
	dep_index[bi_current] = scaffold[ni_scaffoldArr][1] // nnt_bead

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

			### gather 3D positions
			row_index[bi_current] = vstrand_rows[vstrand]
			col_index[bi_current] = vstrand_cols[vstrand]
			dep_index[bi_current] = scaffold[ni_scaffoldArr][1] // nnt_bead

		### error message
		elif vstrand != vstrand_prev:
			print(f"Error: Scaffold crossover not located at nultiple-of-8 position (vstrand {vstrand}).\n")
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
		ni_staplesArr = find(fiveP_ends_stap[sti][0], fiveP_ends_stap[sti][1], staples)
		staples[ni_staplesArr].extend([ni_current, bi_current])
		strands[bi_current] = sti+1
		vstrand = staples[ni_staplesArr][0]
		vstrand_prev = vstrand

		### gather 3D positions
		row_index[bi_current] = vstrand_rows[vstrand]
		col_index[bi_current] = vstrand_cols[vstrand]
		dep_index[bi_current] = staples[ni_staplesArr][1] // nnt_bead

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

				### gather 3D positions
				row_index[bi_current] = vstrand_rows[vstrand]
				col_index[bi_current] = vstrand_cols[vstrand]
				dep_index[bi_current] = staples[ni_staplesArr][1] // nnt_bead

				### identify paired beads
				if scaffold[ni_staplesArr][2] != -1 or scaffold[ni_staplesArr][4] != -1:
					complements[scaffold[ni_staplesArr][7]] = bi_current
					complements[bi_current] = scaffold[ni_staplesArr][7]

				### error message
				elif vstrand != vstrand_prev:
					print(f"Error: Staple crossover not located at nultiple-of-8 position (vstrand {vstrand}).\n")
					sys.exit()
				vstrand_prev = vstrand

		### error message
		if staples[ni_staplesArr][0] % 2 == 0:
			if staples[ni_staplesArr][1] % nnt_bead != 0:
				print(f"Error: Staple 3' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
				sys.exit()
		elif staples[ni_staplesArr][1] % nnt_bead != 7:
			print(f"Error: Staple 3' end not located at multiple-of-8 position (vstrand {vstrand}).\n")
			sys.exit()

	### results
	return strands, complements, row_index, col_index, dep_index


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
	vstrand_rows = {}
	vstrand_cols = {}
	
	### loop over virtual strands
	for el1 in j["vstrands"]:
		
		### loop over the elements of the virtual strand
		for el2_key, el2 in el1.items():
			
			### read virtual strand index
			if el2_key == "num":
				vstrand_current = el2
				vstrand_rows[vstrand_current] = vstrand_row_current
				vstrand_cols[vstrand_current] = vstrand_col_current

			### read virtual strand row index
			if el2_key == "row":
				vstrand_row_current = el2

			### read virtual strand col index
			if el2_key == "col":
				vstrand_col_current = el2
			
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
		print("Error: Scaffold 5' end not found.\n")
		sys.exit()

	### results
	print(f"Found {nnt_scaf} scaffold nucleotides and {nnt_stap} staple nucleotides.")
	return scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap, vstrand_rows, vstrand_cols


### search for entry in strand/index list that matches given strand/index, return index
def find(strand, index, list):
	for i,item in enumerate(list):
		if item[0] == strand and item[1] == index:
			if item[2] == -1 and item[3] == -1 and item[4] == -1 and item[5] == -1:
				return -1
			return i
	print("Error: Index not found in strand/index list.\n")
	sys.exit()

