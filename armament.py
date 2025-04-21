import arsenal as ars
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import os
import sys

### read lammps-style trajectory
def readAtomDump(datFile,nstep_skip=0,coarse_time=1,bdis="all",coarse_points=1):

	### notes
	# assumes the bdis array stores the atom indices starting from 1.
	# assumes the time step of the second step is the dump frequency.
	# assumes the box diameter does not change.

	### load trajectory file
	print("Loading LAMMPS-style trajectory...")
	ars.testFileExist(datFile,"trajectory")
	with open(datFile) as f:
		content = f.readlines()
	print("Parsing trajectory...")

	### extract metadata
	dbox = 2*float(content[5].split()[1])
	nbd_total = int(content[3].split()[0])
	nstep_recorded = int(len(content)/(nbd_total+9))
	nstep_used = int((nstep_recorded-nstep_skip-1)/coarse_time)+1
	if nstep_used <= 0:
		print("Error: Cannot read atom dump - too much initial time cut off.")
		sys.exit()

	### report step counts
	nstep_per_record = int(content[nbd_total+10].split()[0])
	print("{:1.2e} steps in simulation".format(nstep_recorded*nstep_per_record))
	print("{:1.2e} steps recorded".format(nstep_recorded))
	print("{:1.2e} steps used".format(nstep_used))

	### interpret input
	if isinstance(bdis,str) and bdis == "all":
		bdis = [list(range(1,int(np.ceil(nbd_total/coarse_points))+1))]
	elif isinstance(bdis,int):
		bdis = [[bdis]]
	elif ars.isarray(bdis) and isinstance(bdis[0],int):
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
	col2s = np.zeros(nbd_used,dtype=int)
	groups = np.zeros(nbd_used,dtype=int)
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
				points[i,point_count] = [ ars.applyPBC(float(i)%1-1/2,1)*dbox for i in line[2:5] ]
				point_count += 1
		if i%1000 == 0 and i != 0:
			print(f"processed {i} steps...")

	### return
	if len(bdis) == 1:
		return points,col2s,dbox
	else:
		return points,col2s,groups,dbox


### extract dump frequency from lammps-style trajectory
def getDumpEvery(datFile):

	### notes
	# assumes the time step of the second step is the dump frequency.

	### load trajectory file
	ars.testFileExist(datFile,"trajectory")
	with open(datFile) as f:
		content = f.readlines()

	### extract dump frequency
	nbd_total = int(content[3].split()[0])
	dump_every = int(content[nbd_total+10].split()[0])
	return dump_every


### read lammps-style geometry
def readGeo(geoFile):

	### notes
	# all indices are stored directly as they appear in the geometry file; in
	  # other words, the molecule indices, type indices, and atom indices in 
	  # the bonds all start from 1.

	### load geometry file
	ars.testFileExist(geoFile,"geometry")
	with open(geoFile) as f:
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
	points = np.zeros((natom,3))
	ids = np.zeros(natom,dtype=int)
	molecules = np.zeros(natom,dtype=int)
	types = np.zeros(natom,dtype=int)
	charges = np.zeros(natom)
	is_charged = False
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
				points[ai] = line[3:6]

			### assume full atom style
			elif len(line) == 7 or len(line) == 10:
				if i == 0:
					is_charged = True
				charges[ai] = line[3]
				points[ai] = line[4:7]

			### throw error
			else:
				print("Error: Cannot read geometry - unable to surmise atom style.")
				sys.exit()

	### get bond information
	bonds = np.zeros((nbond,3),dtype=int)
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
	angles = np.zeros((nangle,4),dtype=int)
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

	### return results
	if not is_charged:
		return points,molecules,types,bonds,angles
	else:
		return points,molecules,types,charges,bonds,angles


### read 3D box diameter from geometry file
def getDbox3(geoFile):

	### load geometry file
	ars.testFileExist(geoFile,"geometry")
	with open(geoFile) as f:
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


### read file containing names of simulation copies
def readCopies(copiesFile):
	ars.testFileExist(copiesFile,"copies")
	with open(copiesFile,'r') as f:
		content = f.readlines()
	nsim = len(content)
	simFoldNames = [None]*nsim
	for i in range(nsim):
		simFoldNames[i] = content[i].split()[0]
	return simFoldNames,nsim


### write lammps-style geometry
def writeGeo(geoFile,dbox3,points,molecules="auto",types="auto",bonds="none",angles="none",natomType="auto",nbondType="auto",nangleType="auto",masses="auto",charges="none",precision=2):

	### notes
	# by convention, molecule/type/bond/angle indexing starts at 1, however, there is
	  # nothing in this function that requires this to be the case (it will print the
	  # values it is given).
	# assumes four decimal points is sufficiently precise for setting charge.

	### count atoms
	natom = len(points)

	### interpret input
	if ars.isnumber(dbox3):
		dbox3 = [dbox3,dbox3,dbox3]
	elif not ars.isarray(dbox3) or len(dbox3) != 3:
		print("Flag: Not writing geometry file - dbox3 must be number or 3-element list.")
		return
	if isinstance(molecules,str) and molecules == "auto":
		molecules = np.zeros(natom,dtype=int)
	if isinstance(types,str) and types == "auto":
		types = np.ones(natom,dtype=int)
	if isinstance(bonds,str) and bonds == "none":
		bonds = np.zeros((0,3),dtype=int)
	if isinstance(angles,str) and angles == "none":
		angles = np.zeros((0,4),dtype=int)

	### inerpret charges
	if isinstance(charges,str) and charges == "none":
		is_charged = False
	elif isinstance(charges,str) and charges == "auto":
		is_charged = True
		charges = np.zeros(natom,dtype=int)
		len_charge = 1
	else:
		is_charged = True
		len_charge = len(str(int(max(charges))))

	### count objects
	nmolecule = int(max(molecules))
	nbond = len(bonds)
	nangle = len(angles)

	### some more input interpretation
	if isinstance(natomType,str) and natomType == "auto":
		natomType = int(max(types))
	if isinstance(nbondType,str) and nbondType == "auto":
		if nbond > 0:
			nbondType = int(max(bonds[:,0]))
		else:
			nbondType = 0
	if isinstance(nangleType,str) and nangleType == "auto":
		if nangle > 0:
			nangleType = int(max(angles[:,0]))
		else:
			nangleType = 0
	if isinstance(masses,str) and masses == "auto":
		masses = np.ones(natomType,dtype=int)

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
	with open(geoFile,'w') as f:

		f.write("## Number of Objects\n")
		f.write(f"\t{natom:<{len_nobject}} atoms\n")
		if nbond:
			f.write(f"\t{nbond:<{len_nobject}} bonds\n")
		if nangle:
			f.write(f"\t{nangle:<{len_nobject}} angles\n")
		f.write("\n")

		f.write("## Number of Object Types\n")
		f.write(f"\t{natomType:<{len_nobjectType}} atom types\n")
		if nbondType:
			f.write(f"\t{nbondType:<{len_nobjectType}} bond types\n")
		if nangleType:
			f.write(f"\t{nangleType:<{len_nobjectType}} angle types\n")
		f.write("\n")

		f.write("## Simulation Box\n")
		f.write(f"\t{-dbox3[0]/2:>{len_dbox3+precision+2}.{precision}f} {dbox3[0]/2:>{len_dbox3+precision+1}.{precision}f} xlo xhi\n")
		f.write(f"\t{-dbox3[1]/2:>{len_dbox3+precision+2}.{precision}f} {dbox3[1]/2:>{len_dbox3+precision+1}.{precision}f} ylo yhi\n")
		f.write(f"\t{-dbox3[2]/2:>{len_dbox3+precision+2}.{precision}f} {dbox3[2]/2:>{len_dbox3+precision+1}.{precision}f} zlo zhi\n\n")

		f.write("Masses\n\n")
		for i in range(natomType):
			f.write(f"\t{i+1:<{len_natomType}} {masses[i]}\n")
		f.write("\n")

		f.write("Atoms\n\n")
		for i in range(natom):
			f.write(f"\t{i+1:<{len_natom}} " + \
					f"{int(molecules[i]):<{len_nmolecule}} " + \
					f"{int(types[i]):<{len_natomType}} ")
			if is_charged:
				f.write(f"{charges[i]:>{len_charge+6}.4f}  ") 
			f.write(f"{points[i,0]:>{len_dbox3+precision+2}.{precision}f} " + \
					f"{points[i,1]:>{len_dbox3+precision+2}.{precision}f} " + \
					f"{points[i,2]:>{len_dbox3+precision+2}.{precision}f}\n")
		f.write("\n")

		if nbond:
			f.write("Bonds\n\n")
			for i in range(nbond):
				f.write(f"\t{i+1:<{len_nbond}} " + \
						f"{int(bonds[i,0]):<{len_nbondType}}  " + \
						f"{int(bonds[i,1]):<{len_natom}} " + \
						f"{int(bonds[i,2]):<{len_natom}}\n")
			f.write("\n")

		if nangle:
			f.write("Angles\n\n")
			for i in range(nangle):
				f.write(f"\t{i+1:<{len_nangle}} " + \
						f"{int(angles[i,0]):<{len_nangleType}}  " + \
						f"{int(angles[i,1]):<{len_natom}} " + \
						f"{int(angles[i,2]):<{len_natom}} " + \
						f"{int(angles[i,3]):<{len_natom}}\n")
			f.write("\n")


### set pretty matplotlib defaults
def magicPlot():
	plt.style.use("/Users/dduke/Files/analysis/arsenal/magic.mplstyle")


### shift trajectory, placing the given point at the center, optionally unwrapping molecules at boundary
def centerPointsMolecule(points,molecules,dboxs,center=1,unwrap=True):

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
	points_moleculed = ars.sortPointsByMolecule(points,molecules)

	### initialize
	molecule_coms = np.zeros((nmolecule,3))
	points_centered = np.zeros((nstep,npoint,3))

	### loop over steps
	print("Centering trajectory...")
	for i in range(nstep):

		### calculate molecule coms
		for j in range(nmolecule):
			molecule_coms[j] = ars.calcCOMnoDummy(points_moleculed[j][i,:,:],dboxs[i])

		### set centering point
		if center == "none":
			com = np.zeros(3)
		elif center == "com_points" or center == "com_beads" or center == "com_bases":
			com = ars.calcCOMnoDummy(points[i,:,:],dboxs[i])
		elif center == "com_molecules" or center == "com_clusters":
			com = ars.calcCOMnoDummy(molecule_coms,dboxs[i])
		elif isinstance(center,int) and center <= nmolecule:
			com = molecule_coms[center-1,:]
		else:
			print("Error: Cannot center points - center must be either \"none\", \"com_points\", \"com_molecules\", or integer <= nmolecule.")
			sys.exit()

		### center the points
		for j in range(npoint):
			if (molecule_coms[molecules[j]-1]==[0,0,0]).all():
				points_centered[i,j,:] = points[i,j,:]
			else:
				points_centered[i,j,:] = ars.applyPBC(points[i,j,:] - com, dboxs[i])

		### unwrap molecules at boundary
		if unwrap:
			molecule_coms_centered = np.zeros((nmolecule,3))
			for j in range(nmolecule):
				molecule_coms_centered[j,:] = ars.applyPBC(molecule_coms[j,:] - com, dboxs[i])
			for j in range(npoint):
				ref = molecule_coms_centered[molecules[j]-1,:]
				points_centered[i,j,:] = ref + ars.applyPBC(points_centered[i,j,:] - ref, dboxs[i])
	return points_centered


### calculate sem assuming independent measurements
def calcSEM(A):
	return np.std(A) / np.sqrt(len(A))


### check for overlap between a bead and a list of beads
def checkOverlap(r0,r_other,sigma,dbox):
	for bi in range(len(r_other)):
		if np.linalg.norm(ars.applyPBC(r0-r_other[bi],dbox)) < sigma:
			return True
	return False


### align positions with principal components
def alignPC(r, indices="auto"):

	### interpret input
	if isinstance(indices,str) and indices == "auto":
		indices = np.arange(len(r))

	### calculate principal components
	r -= np.mean(r[indices], axis=0)
	cov = np.cov(r[indices], rowvar=False)
	eigenvalues, eigenvectors = np.linalg.eigh(cov)
	sorted_indices = np.argsort(eigenvalues)[::-1]
	PCs = eigenvectors[:, sorted_indices]

	### return rotated positions
	r_rot = r @ PCs
	return r_rot


### return unit vector
def unitVector(vector):
	return vector / np.linalg.norm(vector)


# arrays (both 1D and 2D) must be numpy arrays
def applyPBC(r,dbox):
	return r - dbox*np.round(r/dbox)


### vector with random orientation and (on average) unit magnitude
def boxMuller():
	x = np.zeros(3)
	x[0] = np.sqrt(-2 * np.log(random.uniform(0,1))) * np.cos(2 * np.pi * random.uniform(0,1))
	x[1] = np.sqrt(-2 * np.log(random.uniform(0,1))) * np.cos(2 * np.pi * random.uniform(0,1))
	x[2] = np.sqrt(-2 * np.log(random.uniform(0,1))) * np.cos(2 * np.pi * random.uniform(0,1))
	return x


### vector randomly placed within box
def randPos(dbox3):
	if ars.isnumber(dbox3):
		dbox3 = [dbox3,dbox3,dbox3]
	x = np.zeros(3)
	x[0] = random.uniform(-dbox3[0]/2,dbox3[0]/2)
	x[1] = random.uniform(-dbox3[1]/2,dbox3[1]/2)
	x[2] = random.uniform(-dbox3[2]/2,dbox3[2]/2)
	return x


### test if files exist
def testFileExist(file,name="the"):
	if os.path.isfile(file) == False:
		print("Error: Could not find " + name + " file:")
		print(file + "\n")
		sys.exit()


### creates new folder, only if it doesn't already exist
def createSafeFold(newFold):
	os.makedirs(newFold,exist_ok=True)


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
	if isinstance(x,list):
		return True
	elif isinstance(x,np.ndarray):
		return True
	else:
		return False


### calculate center of mass, excluding dummy particles
def calcCOMnoDummy(r,dbox):
	r_trim = np.zeros((0,3))
	for i in range(len(r)):
		if (r[i]!=[0,0,0]).any():
			r_trim = np.append(r_trim,[r[i]],axis=0)
	if len(r_trim) == 0:
		return np.zeros(3)
	else:
		com = ars.calcCOM(r_trim,dbox)
		return com 


### sort points into molecules
def sortPointsByMolecule(points,molecules):
	nstep = points.shape[0]
	npoint = points.shape[1]
	nmolecule = int(max(molecules))
	points_moleculed = [None]*nmolecule
	for m in range(nmolecule):
		points_moleculed[m] = np.zeros((nstep,sum(molecules==m+1),3))
	for i in range(nstep):
		n_molecule_count = np.zeros(nmolecule,dtype=int)
		for j in range(npoint):
			points_moleculed[molecules[j]-1][i,n_molecule_count[molecules[j]-1],:] = points[i,j,:]
			n_molecule_count[molecules[j]-1] += 1
	return points_moleculed


### calculate center of mass, using method from Bai and Breen 2008
def calcCOM(r,dbox):
	xi_bar = np.mean( np.cos(2*np.pi*(r/dbox+1/2)), axis=0 )
	zeta_bar = np.mean( np.sin(2*np.pi*(r/dbox+1/2)), axis=0 )
	theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
	r_ref = dbox*(theta_bar/(2*np.pi)-1/2)
	com = r_ref + np.mean( ars.applyPBC(r-r_ref,dbox), axis=0 )
	return com


