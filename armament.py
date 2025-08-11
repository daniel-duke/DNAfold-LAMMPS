import armament as ars
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
import shutil
import os
import sys

### read lammps-style trajectory
def readAtomDump(datFile, nstep_skip=0, coarse_time=1, bdis='all', coarse_points=1, nstep_max='all'):

	### notes
	# assumes the bdis array stores the atom indices starting from 1.
	# assumes the timestep of the second step is the dump frequency.
	# assumes the box diameter is uniform accross dimensions and does not change.

	### load trajectory file
	print("Loading LAMMPS-style trajectory...")
	ars.testFileExist(datFile, "trajectory")
	with open(datFile, 'r') as f:
		content = f.readlines()
	print("Parsing trajectory...")

	### extract metadata
	nbd_total = int(content[3].split()[0])
	dbox = 2*float(content[5].split()[1])
	dump_every = int(content[nbd_total+10].split()[0])
	nstep_recorded = int(len(content)/(nbd_total+9))
	nstep_trimmed = int((nstep_recorded-nstep_skip-1)/coarse_time)+1
	if nstep_trimmed <= 0:
		print("Error: Cannot read atom dump - too much initial time cut off.")
		sys.exit()

	### interpret input
	if isinstance(nstep_max, str) and nstep_max == 'all':
		nstep_used = nstep_trimmed
	elif isinstance(nstep_max, int):
		nstep_used = min([nstep_max,nstep_trimmed])
	else:
		print("Error: Cannot read atom dump - max number of steps must be \"all\" or integer.")
		sys.exit()

	### report step counts
	print("{:1.2e} steps in simulation".format(nstep_recorded*dump_every))
	print("{:1.2e} steps recorded".format(nstep_recorded))
	print("{:1.2e} steps used".format(nstep_used))

	### interpret input
	if isinstance(bdis, str) and bdis == 'all':
		bdis = [list( range(1, int(np.ceil(nbd_total/coarse_points))+1) )]
	elif isinstance(bdis, int):
		bdis = [[bdis]]
	elif ars.isarray(bdis) and isinstance(bdis[0], int):
		bdis = [bdis]
	elif ars.isarray(bdis) and ars.isarray(bdis[0]):
		for i in range(len(bdis)):
			bdis[i] = bdis[i][::coarse_points]
	else:
		print("Error: Cannot read atom dump - bead indices must be \"all\", an int, a 1D int array, or 2D int array.")
		sys.exit()

	### count total number of beads to use
	nbd_used = 0
	for i in range(len(bdis)):
		nbd_used += len(bdis[i])

	### extract the data
	points = np.zeros((nstep_used,nbd_used,3))
	col2s = np.zeros(nbd_used, dtype=int)
	groups = np.zeros(nbd_used, dtype=int)
	for i in range(nstep_used):
		point_count = 0
		for g in range(len(bdis)):
			for j in range(len(bdis[g])):
				bdi = bdis[g][j]-1
				if bdi >= nbd_total:
					print(f"Error: Cannot read atom dump - requested bead index {bdi} exceeds the number of beads in the simulation ({nbd_total}).")
					sys.exit()
				line = content[(nbd_total+9)*(nstep_skip+i*coarse_time)+9+bdi].split()
				if i == 0:
					col2s[point_count] = int(line[1])
					groups[point_count] = g + 1
				point = np.array(line[2:5],dtype=float)
				points[i,point_count] = ars.applyPBC(point-1/2,1)*dbox
				point_count += 1
		if i%1000 == 0 and i != 0:
			print(f"Processed {i} steps...")

	### results
	if len(bdis) == 1:
		return points, col2s, dbox
	else:
		return points, col2s, dbox, groups


### extract number of steps from lammps-style trajectory
def getNstep(datFile, nstep_skip=0, coarse_time=1):

	### load trajectory file
	ars.testFileExist(datFile, "trajectory")
	with open(datFile, 'r') as f:
		content = f.readlines()

	### extract dump frequency
	nbd_total = int(content[3].split()[0])
	nstep_recorded = int(len(content)/(nbd_total+9))
	nstep_trimmed = int((nstep_recorded-nstep_skip-1)/coarse_time)+1
	return nstep_trimmed


### extract dump frequency from lammps-style trajectory
def getDumpEvery(datFile):

	### notes
	# assumes the timestep of the second step is the dump frequency.

	### load trajectory file
	ars.testFileExist(datFile, "trajectory")
	with open(datFile, 'r') as f:
		content = f.readlines()

	### extract dump frequency
	nbd_total = int(content[3].split()[0])
	dump_every = int(content[nbd_total+10].split()[0])
	return dump_every


### read lammps-style geometry
def readGeo(geoFile, **kwargs):

	### keyword arguments
	extraLabel = None if 'extraLabel' not in kwargs else kwargs['extraLabel']

	### notes
	# all indices are stored directly as they appear in the geometry file; in
	  # other words, the molecule indices, type indices, and atom indices in 
	  # the bonds all start from 1.

	### load geometry file
	ars.testFileExist(geoFile, "geometry")
	with open(geoFile, 'r') as f:
		content = f.readlines()

	### get atom and bond counts
	natom = -1
	nbond = -1
	nangle = -1
	for i in range(len(content)):
		if len(content[i].split()) == 2:
			if content[i].split()[1] == 'atoms':
				natom = int(content[i].split()[0])
			if content[i].split()[1] == 'bonds':
				nbond = int(content[i].split()[0])
			if content[i].split()[1] == 'angles':
				nangle = int(content[i].split()[0])
		if natom != -1 and nbond != -1 and nangle != -1:
			break
		if i == len(content)-1:
			if natom == -1:
				natom = 0
			if nbond == -1:
				nbond = 0
			if nangle == -1:
				nangle = 0

	### get atom information
	r = np.zeros((natom,3))
	ids = np.zeros(natom, dtype=int)
	molecules = np.zeros(natom, dtype=int)
	types = np.zeros(natom, dtype=int)
	charges = np.zeros(natom)
	readCharge = False
	if natom:
		for i in range(len(content)):
			if len(content[i].split()) > 0 and content[i].split()[0] == 'Atoms':
				line_index = i+2
				break	
		for i in range(natom):
			line = content[line_index].split()
			line_index += 1

			### identification
			ai = int(line[0])-1
			molecules[ai] = line[1]
			types[ai] = line[2]

			### assume molecular atom style
			if len(line) == 6 or len(line) == 9:
				r[ai] = line[3:6]

			### assume full atom style
			elif len(line) == 7 or len(line) == 10:
				if i == 0:
					readCharge = True
				charges[ai] = line[3]
				r[ai] = line[4:7]

			### throw error
			else:
				print("Error: Cannot read geometry - unable to surmise atom style.")
				sys.exit()

	### get bond information
	bonds = np.zeros((nbond,3), dtype=int)
	if nbond:
		for i in range(len(content)):
			if len(content[i].split()) > 0 and content[i].split()[0] == 'Bonds':
				line_index = i+2
				break
		for i in range(nbond):
			bonds[i,0] = content[line_index].split()[1]
			bonds[i,1] = content[line_index].split()[2]
			bonds[i,2] = content[line_index].split()[3]
			line_index += 1

	### get angle information
	angles = np.zeros((nangle,4), dtype=int)
	if nangle:
		for i in range(len(content)):
			if len(content[i].split()) > 0 and content[i].split()[0] == 'Angles':
				line_index = i+2
				break
		for i in range(nangle):
			angles[i,0] = content[line_index].split()[1]
			angles[i,1] = content[line_index].split()[2]
			angles[i,2] = content[line_index].split()[3]
			angles[i,3] = content[line_index].split()[4]
			line_index += 1


	### get bond information
	if extraLabel != None:
		for i in range(len(content)):
			if len(content[i].split()) > 0 and content[i].split()[0] == extraLabel:
				line_index = i+2
				break

		nextra = len(content[line_index].split())-1
		extras = np.zeros((natom,nextra))
		for i in range(natom):
			line = content[line_index].split()
			line_index += 1

			ai = int(line[0])-1
			for j in range(nextra):
				extras[ai,j] = line[j+1]

	### results
	if not readCharge:
		if extraLabel == None:
			return r, molecules, types, bonds, angles
		else:
			return r, molecules, types, bonds, angles, extras
	else:
		if extraLabel == None:
			return r, molecules, types, charges, bonds, angles
		else:
			return r, molecules, types, charges, bonds, angles, extras


### read 3D box diameter from geometry file
def getDbox3(geoFile):

	### load geometry file
	ars.testFileExist(geoFile, "geometry")
	with open(geoFile, 'r') as f:
		content = f.readlines()

	### extract the info
	dbox3 = [0]*3
	for i in range(len(content)):
		line = content[i].split()
		if len(line) == 4 and line[2] == 'xlo':
			dbox3[0] = float(line[1]) - float(line[0])
		if len(line) == 4 and line[2] == 'ylo':
			dbox3[1] = float(line[1]) - float(line[0])
		if len(line) == 4 and line[2] == 'zlo':
			dbox3[2] = float(line[1]) - float(line[0])
		if all(x > 0 for x in dbox3):
			break
	return dbox3


### read cluster file
def readCluster(clusterFile):
	ars.testFileExist(clusterFile, "cluster")
	with open(clusterFile, 'r') as f:
		content = f.readlines()
	ncluster = int(len(content)/2)
	clusters = [None]*ncluster
	for c in range(ncluster):
		indices = content[1+c*2].split()
		clusters[c] = [None]*len(indices)
		for i in range(len(indices)):
			clusters[c][i] = int(indices[i])
	return clusters


### read file containing names of simulation copies
def readCopies(copiesFile):
	ars.testFileExist(copiesFile, "copies")
	with open(copiesFile, 'r') as f:
		content = f.readlines()
	nsim = len(content)
	simFoldNames = [None]*nsim
	for i in range(nsim):
		simFoldNames[i] = content[i].split()[0]
	return simFoldNames, nsim


### write lammps-style geometry
def writeGeo(geoFile, dbox3, r, molecules='auto', types='auto', bonds=None, angles=None, **kwargs):

	### added keyword args
	natomType	= 'auto'	if 'natomType' not in kwargs else kwargs['natomType']
	nbondType	= 'auto'	if 'nbondType' not in kwargs else kwargs['nbondType']
	nangleType	= 'auto'	if 'nangleType' not in kwargs else kwargs['nangleType']
	masses		= 'auto'	if 'masses' not in kwargs else kwargs['masses']
	charges		= None		if 'charges' not in kwargs else kwargs['charges']
	extras		= None		if 'extras' not in kwargs else kwargs['extras']
	x_precision	= 2			if 'x_precision' not in kwargs else kwargs['x_precision']
	q_precision	= 4			if 'q_precision' not in kwargs else kwargs['q_precision']
	e_precision	= 4			if 'e_precision' not in kwargs else kwargs['e_precision']

	### notes
	# by convention, molecule/type/bond/angle indexing starts at 1, however, there is
	  # nothing in this function that requires this to be the case (it will print the
	  # values it is given).

	### count atoms
	natom = len(r)

	### interpret input
	if ars.isnumber(dbox3):
		dbox3 = [dbox3,dbox3,dbox3]
	elif not ars.isarray(dbox3) or len(dbox3) != 3:
		print("Flag: Not writing geometry file - dbox3 must be number or 3-element list.")
		return
	if isinstance(molecules, str) and molecules == 'auto':
		molecules = np.zeros(natom, dtype=int)
	if isinstance(types, str) and types == 'auto':
		types = np.ones(natom, dtype=int)
	if bonds is None:
		bonds = np.zeros((0,3), dtype=int)
	if angles is None:
		angles = np.zeros((0,4), dtype=int)

	### inerpret charges
	if charges is None:
		includeCharge = False
	elif isinstance(charges, str) and charges == 'auto':
		includeCharge = True
		charges = np.zeros(natom, dtype=int)
		len_charge = 1
	else:
		includeCharge = True
		len_charge = len(str(int(max(charges))))

	### inerpret extras
	if extras is None:
		includeExtra = False
	else:
		includeExtra = True
		nextra = extras.shape[1]
		len_extra = [None]*nextra
		for i in range(nextra):
			len_extra[i] = len(str(int(max(extras[:,i]))))

	### count objects
	nmolecule = int(max(molecules))
	nbond = len(bonds)
	nangle = len(angles)

	### some more input interpretation
	if isinstance(natomType, str) and natomType == 'auto':
		natomType = int(max(types))
	if isinstance(nbondType, str) and nbondType == 'auto':
		if nbond > 0:
			nbondType = int(max(bonds[:,0]))
		else:
			nbondType = 0
	if isinstance(nangleType, str) and nangleType == 'auto':
		if nangle > 0:
			nangleType = int(max(angles[:,0]))
		else:
			nangleType = 0
	if isinstance(masses, str) and masses == 'auto':
		masses = np.ones(natomType, dtype=int)

	### count digits
	len_natom = len(str(natom))
	len_nbond = len(str(nbond))
	len_nangle = len(str(nangle))
	len_nobject = max([len_natom,len_nbond,len_nangle])
	len_nmolecule = len(str(nmolecule))
	len_natomType = len(str(natomType))
	len_nbondType = len(str(nbondType))
	len_nangleType = len(str(nangleType))
	len_nobjectType = max([len_natomType,len_nbondType,len_nangleType])
	len_dbox3 = len(str(int(max(dbox3)/2)))

	### write to file
	with open(geoFile, 'w') as f:

		f.write("## Number of Objects\n")
		f.write(f"\t{natom:<{len_nobject}} atoms\n")
		if nbond:
			f.write(f"\t{nbond:<{len_nobject}} bonds\n")
		if nangle:
			f.write(f"\t{nangle:<{len_nobject}} angles\n")

		f.write("\n## Number of Object Types\n")
		f.write(f"\t{natomType:<{len_nobjectType}} atom types\n")
		if nbondType:
			f.write(f"\t{nbondType:<{len_nobjectType}} bond types\n")
		if nangleType:
			f.write(f"\t{nangleType:<{len_nobjectType}} angle types\n")

		f.write("\n## Simulation Box\n")
		f.write(f"\t{-dbox3[0]/2:>{len_dbox3+x_precision+2}.{x_precision}f} {dbox3[0]/2:>{len_dbox3+x_precision+1}.{x_precision}f} xlo xhi\n")
		f.write(f"\t{-dbox3[1]/2:>{len_dbox3+x_precision+2}.{x_precision}f} {dbox3[1]/2:>{len_dbox3+x_precision+1}.{x_precision}f} ylo yhi\n")
		f.write(f"\t{-dbox3[2]/2:>{len_dbox3+x_precision+2}.{x_precision}f} {dbox3[2]/2:>{len_dbox3+x_precision+1}.{x_precision}f} zlo zhi\n")

		f.write("\nMasses\n\n")
		for i in range(natomType):
			f.write(f"\t{i+1:<{len_natomType}} {masses[i]}\n")

		f.write("\nAtoms\n\n")
		for i in range(natom):
			f.write(f"\t{i+1:<{len_natom}}" + \
					f" {int(molecules[i]):<{len_nmolecule}}" + \
					f" {int(types[i]):<{len_natomType}}")
			if includeCharge:
				f.write(f" {charges[i]:>{len_charge+q_precision+1}.{q_precision}f}") 
			f.write(f"  {r[i,0]:>{len_dbox3+x_precision+2}.{x_precision}f}" + \
					 f" {r[i,1]:>{len_dbox3+x_precision+2}.{x_precision}f}" + \
					 f" {r[i,2]:>{len_dbox3+x_precision+2}.{x_precision}f}\n")

		if nbond:
			f.write("\nBonds\n\n")
			for i in range(nbond):
				f.write(f"\t{i+1:<{len_nbond}} " + \
						f"{int(bonds[i,0]):<{len_nbondType}}  " + \
						f"{int(bonds[i,1]):<{len_natom}} " + \
						f"{int(bonds[i,2]):<{len_natom}}\n")

		if nangle:
			f.write("\nAngles\n\n")
			for i in range(nangle):
				f.write(f"\t{i+1:<{len_nangle}} " + \
						f"{int(angles[i,0]):<{len_nangleType}}  " + \
						f"{int(angles[i,1]):<{len_natom}} " + \
						f"{int(angles[i,2]):<{len_natom}} " + \
						f"{int(angles[i,3]):<{len_natom}}\n")

		if includeExtra:
			f.write("\nExtras\n\n")
			for i in range(natom):
				f.write(f"\t{i+1:<{len_natom}}")
				for j in range(nextra):
					f.write(f" {extras[i][j]:>{len_extra[j]+e_precision+1}.{e_precision}f}")
				f.write("\n")


### set pretty matplotlib defaults
def magicPlot(pubReady=False):

	### set default magic settings
	params = {
		'figure.figsize'	: '8,6',
		'font.family'		: 'Times',
		'text.usetex'		: True,
		'errorbar.capsize'	: 3,
		'lines.markersize'	: 6,
		'legend.fontsize'	: 14,
		'xtick.labelsize'	: 14,
		'ytick.labelsize'	: 14,
		'axes.labelsize'	: 14,
		'axes.titlesize'	: 16
	}
	plt.rcParams.update(params)

	### set default magic settings
	paramsPub = {
		'figure.figsize'	: '8,6',
		'font.family'		: 'Times',
		'text.usetex'		: True,
		'errorbar.capsize'	: 3,
		'lines.markersize'	: 6,
		'legend.fontsize'	: 20,
		'xtick.labelsize'	: 20,
		'ytick.labelsize'	: 20,
		'axes.labelsize'	: 20,
		'axes.titlesize'	: 24
	}

	if not pubReady:
		plt.rcParams.update(params)
	else:
		plt.rcParams.update(paramsPub)


### plot a nice histogram
def plotHist(A, Alabel=None, title=None, figLabel="Hist", nbin='auto', Alim_bin='auto', Alim_plot='auto', Ylabel='auto', useDensity=False, **kwargs):

	### added keyword args
	weights			= None		if 'weights' not in kwargs else kwargs['weights']
	plotAsLine		= False		if 'plotAsLine' not in kwargs else kwargs['plotAsLine']
	plotAvgLine		= False		if 'plotAvgLine' not in kwargs else kwargs['plotAvgLine']
	alpha			= 0.6		if 'alpha' not in kwargs else kwargs['alpha']

	### notes
	# if plotting multiple histograms in the same plot, be careful with plotAvgLine (the legend might be wonky)

	### interpret input
	if isinstance(nbin, str) and nbin == 'auto':
		nbin = ars.optbins(A, 50)
	elif not isinstance(nbin, int):
		print("Flag: Skipping histogram plot - number of histogram bins must be either \"auto\" or integer.")
		return
	if isinstance(Alim_bin, str) and Alim_bin == 'auto':
		Alim_bin = [ min(A), max(A) ]
	elif not ars.isarray(Alim_bin) or len(Alim_bin) != 2:
		print("Flag: Skipping histogram plot - variable limits must be either \"auto\" or 2-element list.")
		return
	if isinstance(Alim_plot, str) and Alim_plot == 'auto':
		dAbin = (Alim_bin[1]-Alim_bin[0])/nbin
		Alim_plot = [ Alim_bin[0]-dAbin/2, Alim_bin[1]+dAbin/2 ]
	elif not ars.isarray(Alim_plot) or len(Alim_plot) != 2:
		print("Flag: Skipping histogram plot - variable limits must be either \"auto\" or 2-element list.")
		return

	### plot histogram
	plt.figure(figLabel)
	if plotAsLine == False:
		plt.hist(A, nbin, weights=weights, range=Alim_bin, density=useDensity, alpha=alpha, edgecolor='black')
	else:
		heights, edges = np.histogram(A, nbin, weights=weights, range=Alim_bin, useDensity=True)
		edges = edges[:len(edges)-1] + 1/2*(edges[1]-edges[0])
		plt.plot(edges, heights, color='black')
	if plotAvgLine:
		plt.axvline(np.mean(A), color='red', linestyle='--', label=f"Avg = {np.mean(A):0.2f}")
	plt.xlim(Alim_plot)
	if Alabel is not None:
		plt.xlabel(Alabel)
	if Ylabel == 'auto':
		if useDensity:
			plt.ylabel("Density")
		else:
			plt.ylabel("Count")
	elif Ylabel is not None:
		plt.ylabel(Ylabel)
	if title is not None:
		plt.title(title)
	if plotAvgLine:
		plt.legend()


### shift trajectory, placing the given point at the center, optionally unwrapping molecules at boundary
def centerPointsMolecule(points, molecules, dboxs, center=1, unwrap=True):

	### notes
	# as the notation suggests, the indices contained in molecules must start at 1.
	# particles located perfectly at the origin are assumed to be dummy, and are thus
	  # not included in the center of mass calculation.

	### get counts
	nstep = points.shape[0]
	npoint = points.shape[1]
	nmolecule = int(max(molecules))

	### interpret input
	if ars.isnumber(dboxs):
		dboxs = nstep*[dboxs]
	elif not ars.isarray(dboxs) or len(dboxs) != nstep:
		print("Error: Cannot center points - dboxs must be number or nstep-element list.")
		sys.exit()

	### sort points by molecule
	points_moleculed = ars.sortPointsByMolecule(points, molecules)

	### initialize
	molecule_coms = np.zeros((nmolecule,3))
	points_centered = np.zeros((nstep,npoint,3))

	### loop over steps
	print("Centering trajectory...")
	for i in range(nstep):

		### calculate molecule coms
		for j in range(nmolecule):
			molecule_coms[j] = ars.calcCOMnoDummy(points_moleculed[j][i,:,:], dboxs[i])

		### set centering point
		if center == "none":
			com = np.zeros(3)
		elif center == "com_points" or center == "com_beads" or center == "com_bases":
			com = ars.calcCOMnoDummy(points[i,:,:], dboxs[i])
		elif center == "com_molecules" or center == "com_clusters":
			com = ars.calcCOMnoDummy(molecule_coms, dboxs[i])
		elif isinstance(center, int) and center <= nmolecule:
			com = molecule_coms[center-1,:]
		else:
			print("Error: Cannot center points - center must be either \"none\", \"com_points\", \"com_molecules\", or integer <= nmolecule.")
			sys.exit()

		### center the points
		for j in range(npoint):
			if (molecule_coms[molecules[j]-1]==[0,0,0]).all():
				points_centered[i,j,:] = points[i,j,:]
			else:
				points_centered[i,j,:] = ars.applyPBC(points[i,j,:]-com, dboxs[i])

		### unwrap molecules at boundary
		if unwrap:
			molecule_coms_centered = np.zeros((nmolecule,3))
			for j in range(nmolecule):
				molecule_coms_centered[j,:] = ars.applyPBC(molecule_coms[j,:]-com, dboxs[i])
			for j in range(npoint):
				ref = molecule_coms_centered[molecules[j]-1,:]
				points_centered[i,j,:] = ref + ars.applyPBC(points_centered[i,j,:]-ref, dboxs[i])

		### progress update
		if i%1000 == 0 and i != 0:
			print(f"Centered {i} steps...")

	### result
	return points_centered


### calculate center of mass, excluding dummy particles (any particle exactly at origin)
def calcCOMnoDummy(r, dbox):
	r_trim = np.zeros((0,3))
	for i in range(len(r)):
		if any(r[i]!=[0,0,0]):
			r_trim = np.append(r_trim, [r[i]], axis=0)
	if len(r_trim) == 0:
		return np.zeros(3)
	else:
		com = ars.calcCOM(r_trim, dbox)
		return com 


### calculate center of mass, using method from Bai and Breen 2008
def calcCOM(r, dbox):
	xi_bar = np.mean( np.cos(2*np.pi*(r/dbox+1/2)), axis=0 )
	zeta_bar = np.mean( np.sin(2*np.pi*(r/dbox+1/2)), axis=0 )
	theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
	r_ref = dbox*(theta_bar/(2*np.pi)-1/2)
	com = r_ref + np.mean( ars.applyPBC(r-r_ref, dbox), axis=0 )
	return com


### calculate optimum number of histogram bins
def optbins(A, maxM):
	N = len(A)
	logp = np.zeros(maxM)
	for M in range(1, maxM+1):
		n = np.histogram(A,bins=M)[0]
		part1 = N*np.log(M) + gammaln(M/2) - gammaln(N+M/2)
		part2 = -M*gammaln(1/2) + np.sum(gammaln(n+1/2))
		logp[M-1] = part1 + part2
	optM = np.argmax(logp) + 1
	return optM


### calculate sem assuming independent measurements
def calcSEM(A):
	return np.std(A) / np.sqrt(len(A))


### check for overlap between a bead and a list of beads
def checkOverlap(r0, r_other, sigma, dbox):
	return np.any(np.linalg.norm(ars.applyPBC(r_other-r0, dbox),axis=1) < sigma)


### align positions with principal components
def alignPC(r, indices='auto'):

	### interpret input
	if isinstance(indices, str) and indices == 'auto':
		indices = np.arange(len(r))

	### calculate principal components
	r -= np.mean(r[indices], axis=0)
	cov = np.cov(r[indices], rowvar=False)
	eigenvalues, eigenvectors = np.linalg.eigh(cov)
	sorted_indices = np.argsort(eigenvalues)[::-1]
	PCs = eigenvectors[:,sorted_indices]

	### return rotated positions
	r_rot = r @ PCs
	return r_rot


### calculate moving average of an array
def movingAvg(A, stride=1):
	if stride < 1:
		print("Flag: Skipping moving average, stride must be positive integer.")
	avg = np.convolve(A, np.ones(stride)/stride, mode='valid')
	pad_left = (stride - 1) // 2
	pad_right = stride // 2
	return np.pad(avg, (pad_left,pad_right), mode='edge')


### unit vector
def unitVector(vector):
	return vector / np.linalg.norm(vector)


### apply periodic boundary condition
def applyPBC(r, dbox):

	### notes
	# handles single values, single points, or arrays of points
	# arrays (both 1D and 2D) must be numpy arrays

	return r - dbox*np.round(r/dbox)


### vector with random orientation and (on average) unit magnitude
def boxMuller(rng=np.random):
	return np.sqrt(-2 * np.log(rng.uniform(size=3))) * np.cos(2 * np.pi * rng.uniform(size=3))


### vector randomly placed within box
def randPos(dbox3, rng=np.random):
	if ars.isnumber(dbox3):
		dbox3 = [dbox3,dbox3,dbox3]
	x = np.zeros(3)
	for i in range(3):
		x[i] = rng.uniform(-dbox3[i]/2, dbox3[i]/2)
	return x


### get common nice colors
def getColor(color):
	if color == 'teal':
		return np.array([0,145,147])
	elif color == 'orchid':
		return np.array([122,129,255])
	elif color == 'silver':
		return np.array([214,241,241])
	elif color == 'purple':
		return np.array([68,1,84])
	elif color == 'grey':
		return np.array([153,153,153])
	else:
		print("Error: Unknown color.")
		sys.exit()


### sort points into molecules
def sortPointsByMolecule(points, molecules):
	nstep = points.shape[0]
	npoint = points.shape[1]
	nmolecule = int(max(molecules))
	points_moleculed = [None]*nmolecule
	for m in range(nmolecule):
		points_moleculed[m] = np.zeros((nstep,sum(molecules==m+1),3))
	for i in range(nstep):
		n_molecule_count = np.zeros(nmolecule, dtype=int)
		for j in range(npoint):
			points_moleculed[molecules[j]-1][i,n_molecule_count[molecules[j]-1],:] = points[i,j,:]
			n_molecule_count[molecules[j]-1] += 1
	return points_moleculed


### test if files exist
def testFileExist(file, name="the", required=True):
	if os.path.isfile(file):
		return True
	else:
		if required:
			print(f"Error: Could not find {name} file:")
			print(file + "\n")
			sys.exit()
		else:
			print(f"Flag: Could not find {name} file.")
			return False


### creates new folder, only if it doesn't already exist
def createSafeFold(newFold):
	os.makedirs(newFold, exist_ok=True)


### creates new empty folder
def createEmptyFold(newFold):
	if os.path.exists(newFold):
		shutil.rmtree(newFold)
	os.makedirs(newFold)


### test if variable is numeric (both float and integer count)
def isnumber(x):
	try:
		value = float(x)
		return True
	except:
		return False


### check if variable is an array (both list and numpy array work)
def isarray(x):
	if isinstance(x, list):
		return True
	elif isinstance(x, np.ndarray):
		return True
	else:
		return False


