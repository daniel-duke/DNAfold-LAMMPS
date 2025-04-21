import armament as ars
from scipy import stats
import numpy as np
import json
import sys

## Description
# this script contains various functions useful for analyzing DNAfold simulations.


################################################################################
### File Handlers

### get simulation folders
def getSimFolds(copiesFile=None, simFold=None, rseed=1):
	if copiesFile is not None:
		copyNames, nsim = ars.readCopies(copiesFile)
		simFolds = [ copyNames[i] + "/" for i in range(nsim) ]
	else:
		nsim = 1
		if simFold is not None:
			simFolds = [ args.simFold + "/" ]
		else:
			simFolds = [ f"sim{rseed:02.0f}" + "/" ]
	return simFolds, nsim


### read oxdna configuration
def readConf(confFile, nba_total):
	ars.testFileExist(confFile,"configuration")
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
	ars.testFileExist(topFile,"topology")
	with open(topFile) as f:
		content = f.readlines()
	nba_total = int(content[0].split()[0])
	strands = [int(line.split()[0]) for line in content[1:]]
	return strands, nba_total


### read hybridization times file
def readHybStatus(inHybFile, nstep_skip=0, coarse_time=1, nstep_max="all"):
	ars.testFileExist(inHybFile,"hybridization status")
	with open(inHybFile, 'r') as f:
		content = f.readlines()

	### extract metadata
	nbead = 0
	while ars.isnumber(content[nbead+1].split()[0]):
		nbead += 1
	nstep_recorded = int(len(content)/(nbead+1))
	nstep_trimmed = int((nstep_recorded-nstep_skip-1)/coarse_time)+1
	if nstep_trimmed <= 0:
		print("Error: Cannot read hyb status - too much initial time cut off.")
		sys.exit()

	### interpret input
	if isinstance(nstep_max, str) and nstep_max == "all":
		nstep_used = nstep_trimmed
	elif isinstance(nstep_max, int):
		nstep_used = min([nstep_max,nstep_trimmed])
	else:
		print("Error: Cannot read hyb status - max number of steps must be \"all\" or integer.")
		sys.exit()

	### read data
	hyb_status = np.zeros((nstep_used,nbead),dtype=int)
	for i in range(nstep_used):
		for j in range(nbead):
			hyb_status[i,j] = int(content[(nbead+1)*(nstep_skip+i*coarse_time)+1+j].split()[1])

	### results
	return hyb_status


### extract dump frequency from hyb status file
def getDumpEveryHyb(inHybFile):
	ars.testFileExist(inHybFile,"hybridization status")
	with open(inHybFile, 'r') as f:
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

	### set active viewport to top perspective
	viewport = scene.viewports.active_vp
	viewport.type = Viewport.Type.PERSPECTIVE
	viewport.camera_dir = (-1,0,0)
	viewport.camera_up = (0,1,0)
	viewport.zoom_all()

	### results
	return pipeline


################################################################################
### Calculation Managers

### calculate first bind times from hybridization status
# integer for timestep of first hybridization
# 0 for never hybridized
# -1 for no complement
def calcFirstHybTimes(hyb_status, complements, n_scaf, dump_every):
	nstep = hyb_status.shape[0]
	first_hyb_times = np.zeros(n_scaf)
	for i in range(n_scaf):
		if len(complements[i]) == 0:
			first_hyb_times[i] = -1
	for i in range(nstep):
		for j in range(n_scaf):
			if hyb_status[i,j] == 1 and first_hyb_times[j] == 0:
				first_hyb_times[j] = i*dump_every
	first_hyb_times_scaled = np.zeros(n_scaf)
	for i in range(n_scaf):
		if first_hyb_times[i] == -1:
			first_hyb_times_scaled[i] = -1
		elif first_hyb_times[i] == 0:
			first_hyb_times_scaled[i] = 1
		else:
			first_hyb_times_scaled[i] = first_hyb_times[i]/nstep/dump_every
	return first_hyb_times, first_hyb_times_scaled


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


### initialize positions according to caDNAno positions
def initPositionsCaDNAno(cadFile):
	print("Initializing positions from caDNAno...")

	### parse caDNAno file
	strands, _, row_index, col_index, dep_index = buildDNAfoldModel(cadFile)
	strands = np.array(strands)+1
	n_ori = len(row_index)

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
		r[bi,1] = -(row_index[bi]-row_middle)*2.4
		r[bi,2] = -(dep_index[bi]-dep_middle)*2.72

	### results
	return r, strands


### initialize positions according to oxdna configuration file
def initPositionsOxDNA(cadFile, topFile, confFile):
	print("Initializing positions from oxDNA...")
	nnt_per_bead = 8

	### parse caDNAno file
	strands, complements = buildDNAfoldModel(cadFile)[:2]
	strands = np.array(strands)+1
	complements = np.array(complements)+1
	n_scaf = np.sum(strands==1)
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
		bais = [ nba_offset+nba_scaf-bi*nnt_per_bead-j-1 for j in range(nnt_per_bead)]
		r[bi] = np.mean( ars.applyPBC( coms[bais]+0.6*a1s[bais], dbox3 ), axis=0) *0.8518

	### get oxDNA staple positions
	n_stap_5p = (nba_total - nba_scaf - nba_offset) // nnt_per_bead
	n_stap_3p = nba_offset // nnt_per_bead
	n_stap = n_stap_5p + n_stap_3p
	r_ox_stap = np.zeros((n_stap,3))
	strands_ox_stap = np.zeros(n_stap,dtype=int)
	for bi in range(n_stap_5p):
		bais = [ nba_total-bi*nnt_per_bead-j-1 for j in range(nnt_per_bead) ]
		r_ox_stap[bi] = np.mean( ars.applyPBC( coms[bais]+0.6*a1s[bais], dbox3 ), axis=0) *0.8518
		strands_ox_stap[bi] = base_strands[bais[0]]
	for bi in range(n_stap_3p):
		bais = [ nba_offset-bi*nnt_per_bead-j-1 for j in range(nnt_per_bead) ]
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
					print("Error: conflicting information relating DNAfold strands to oxDNA strands.")

	### assign staple positions
	bi = n_scaf
	for strand in range(2,nstrand+1):
		for stbi in range(n_stap):
			if strand_ox_to_strand[strands_ox_stap[stbi]-1] == strand:
				r[bi] = r_ox_stap[stbi]
				bi += 1

	### align positions
	r -= np.mean(r,axis=0)
	r = np.flip(ars.alignPC(r,np.arange(n_scaf)),axis=1)

	### flip such that 5p scaffold end is in upper left corner
	r = r*np.sign(r[0])*[-1,1,1]

	### results
	return r, strands


################################################################################
### DNAfold

### translate caDNAno design to DNAfold model
def buildDNAfoldModel(cadFile):

	### parse the caDNAno file
	scaffold, staples, fiveP_end_scaf, fiveP_ends_stap, nnt_scaf, nnt_stap, vstrand_rows, vstrand_cols = parseCaDNAno(cadFile)

	### initial calculations
	print("Building DNAfold model...")
	nnt_per_bead = 8
	n_scaf = nnt_scaf // nnt_per_bead
	n_stap = nnt_stap // nnt_per_bead
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
	dep_index[bi_current] = scaffold[ni_scaffoldArr][1] // nnt_per_bead

	### error message
	if scaffold[ni_scaffoldArr][0] % 2 == 0:
		if scaffold[ni_scaffoldArr][1] % nnt_per_bead != 0:
			print(f"Error: Scaffold 5' end not located at multiple-of-8 position (vstrand {vstrand}).")
			sys.exit()
	elif scaffold[ni_scaffoldArr][1] % nnt_per_bead != 7:
		print(f"Error: Scaffold 5' end not located at multiple-of-8 position (vstrand {vstrand}).")
		sys.exit()

	### track along scaffold until 3' end eached
	while scaffold[ni_scaffoldArr][4] != -1:
		ni_scaffoldArr = find(scaffold[ni_scaffoldArr][4], scaffold[ni_scaffoldArr][5], scaffold)

		### update nucleotide and bead indices
		ni_current += 1
		bi_current = ni_current // nnt_per_bead
		scaffold[ni_scaffoldArr].extend([ni_current, bi_current])
		vstrand = scaffold[ni_scaffoldArr][0]

		### store vstrand and backbone bonds for new beads
		if bi_current > (ni_current-1) // nnt_per_bead:
			backbone_neighbors[bi_current][0] = bi_current-1
			backbone_neighbors[bi_current-1][1] = bi_current

			### gather 3D positions
			row_index[bi_current] = vstrand_rows[vstrand]
			col_index[bi_current] = vstrand_cols[vstrand]
			dep_index[bi_current] = scaffold[ni_scaffoldArr][1] // nnt_per_bead

		### error message
		elif vstrand != vstrand_prev:
			print(f"Error: Scaffold crossover not located at nultiple-of-8 position (vstrand {vstrand}).")
			sys.exit()
		vstrand_prev = vstrand

	### error message
	if scaffold[ni_scaffoldArr][0] % 2 == 0:
		if scaffold[ni_scaffoldArr][1] % nnt_per_bead != 7:
			print(f"Error: Scaffold 3' end not located at multiple-of-8 position (vstrand {vstrand}).")
			sys.exit()
	elif scaffold[ni_scaffoldArr][1] % nnt_per_bead != 0:
		print(f"Error: Scaffold 3' end not located at multiple-of-8 position (vstrand {vstrand}).")
		sys.exit()

	### loop over staples
	nstap = len(fiveP_ends_stap)
	for sti in range(nstap):

		### new nucleotide and bead incides
		ni_current += 1
		bi_current = ni_current // nnt_per_bead

		### pick up nucleotide and bead indexing with 5' staple end
		ni_staplesArr = find(fiveP_ends_stap[sti][0], fiveP_ends_stap[sti][1], staples)
		staples[ni_staplesArr].extend([ni_current, bi_current])
		strands[bi_current] = sti+1
		vstrand = staples[ni_staplesArr][0]
		vstrand_prev = vstrand

		### gather 3D positions
		row_index[bi_current] = vstrand_rows[vstrand]
		col_index[bi_current] = vstrand_cols[vstrand]
		dep_index[bi_current] = staples[ni_staplesArr][1] // nnt_per_bead

		### identify paired beads
		if scaffold[ni_staplesArr][2] != -1 or scaffold[ni_staplesArr][4] != -1:
			complements[scaffold[ni_staplesArr][7]] = bi_current
			complements[bi_current] = scaffold[ni_staplesArr][7]

		### error message
		if staples[ni_staplesArr][0] % 2 == 0:
			if staples[ni_staplesArr][1] % nnt_per_bead != 7:
				print(f"Error: Staple 5' end not located at multiple-of-8 position (vstrand {vstrand}).")
				sys.exit()
		elif staples[ni_staplesArr][1] % nnt_per_bead != 0:
			print(f"Error: Staple 5' end not located at multiple-of-8 position (vstrand {vstrand}).")
			sys.exit()

		### track along staple until 3' end eached
		while staples[ni_staplesArr][4] != -1:
			ni_staplesArr = find(staples[ni_staplesArr][4], staples[ni_staplesArr][5], staples)

			### update nucleotide and bead indices
			ni_current += 1
			bi_current = ni_current // nnt_per_bead
			staples[ni_staplesArr].extend([ni_current, bi_current])
			vstrand = staples[ni_staplesArr][0]

			### store vstrand, strand, and backbone bonds for new beads
			if bi_current > (ni_current-1) // nnt_per_bead:
				strands[bi_current] = sti+1
				backbone_neighbors[bi_current][0] = bi_current-1
				backbone_neighbors[bi_current-1][1] = bi_current

				### gather 3D positions
				row_index[bi_current] = vstrand_rows[vstrand]
				col_index[bi_current] = vstrand_cols[vstrand]
				dep_index[bi_current] = staples[ni_staplesArr][1] // nnt_per_bead

				### identify paired beads
				if scaffold[ni_staplesArr][2] != -1 or scaffold[ni_staplesArr][4] != -1:
					complements[scaffold[ni_staplesArr][7]] = bi_current
					complements[bi_current] = scaffold[ni_staplesArr][7]

				### error message
				elif vstrand != vstrand_prev:
					print(f"Error: Staple crossover not located at nultiple-of-8 position (vstrand {vstrand}).")
					sys.exit()
				vstrand_prev = vstrand

		### error message
		if staples[ni_staplesArr][0] % 2 == 0:
			if staples[ni_staplesArr][1] % nnt_per_bead != 0:
				print(f"Error: Staple 3' end not located at multiple-of-8 position (vstrand {vstrand}).")
				sys.exit()
		elif staples[ni_staplesArr][1] % nnt_per_bead != 7:
			print(f"Error: Staple 3' end not located at multiple-of-8 position (vstrand {vstrand}).")
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
		print("Error: Scaffold 5' end not found.")
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
	print("Error: index not found, try again.")
	sys.exit()

