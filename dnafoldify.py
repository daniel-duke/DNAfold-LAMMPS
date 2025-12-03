import armament as ars
import utils
import argparse
import numpy as np
import sys
import json
import copy

## Description
# this script reads a caDNAno file, makes the structure DNAfold compatible
  # (if possible), and writes a new caDNAno file.


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--cadFile',	type=str, required=True,	help='name of caDNAno file')
	parser.add_argument('--debug',		type=int, default=False,	help='whether to print debugging output')

	### set arguments
	args = parser.parse_args()
	cadFile = args.cadFile
	debug = args.debug


################################################################################
### Heart

	### read the caDNAno file
	j, scaffold, staples, colors_scaffold, colors_staples = parseCaDNAno(cadFile)

	### shift things around
	scaffold, staples, colors_scaffold, colors_staples = DNAfoldifyOld(scaffold, staples, colors_scaffold, colors_staples, debug)

	### write edited file
	cadEditFile = cadFile[:-5] + "_edited" + cadFile[-5:]
	writeCaDNAno(cadEditFile, j, scaffold, staples, colors_scaffold, colors_staples)


################################################################################
### File Handlers

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
	colors_scaffold = None
	colors_staples = []
	
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

			### read staple side of helix
			elif el2_key == "stap":
				
				### loop over nucleotides
				for ni_vstrand, neighbors in enumerate(el2):
					
					### store virtual strand index and nucleotide index for current nucleotide and its neighbors
					staple_current = [vstrand_current, int(ni_vstrand)]
					for s in neighbors:
						staple_current.append(int(s))
					staples.append(staple_current)

			### read scaffold colors
			elif el2_key == "scaf_colors":

				### check if color defined
				if len(el2) > 0:

					### check for multiple scaffolds
					if colors_scaffold is not None:
						print("Error: Multiple scaffolds detected.")
						sys.exit()

					### set scaffold color
					cID = [vstrand_current, el2[0][0], el2[0][1]]
					colors_scaffold = [cID]

			### read staple colors
			elif el2_key == "stap_colors":

				### set staple colors
				for ci in range(len(el2)):
					cID = [vstrand_current, el2[ci][0], el2[ci][1]]
					colors_staples.append(cID)

	### results
	return j, scaffold, staples, colors_scaffold, colors_staples


### write edited caDNAno file
def writeCaDNAno(cadEditFile, j, scaffold, staples, colors_scaffold, colors_staples):
	print("Writing caDNAno file...")

	### initialize edited json
	j_edit = copy.deepcopy(j)
	j_edit["name"] = cadEditFile

	### loop over virtual strands
	for vi in range(len(j["vstrands"])):

		### edit scaffold
		for ni in range(len(scaffold)):
			if scaffold[ni][0] == j["vstrands"][vi]["num"]:
				ni_vstrand = scaffold[ni][1]
				j_edit["vstrands"][vi]["scaf"][ni_vstrand] = scaffold[ni][2:]

		### edit staples
		for ni in range(len(staples)):
			if staples[ni][0] == j["vstrands"][vi]["num"]:
				ni_vstrand = staples[ni][1]
				j_edit["vstrands"][vi]["stap"][ni_vstrand] = staples[ni][2:]

		### edit scaffold color
		cID = colors_scaffold[0]
		j_edit["vstrands"][cID[0]]["scaf_colors"] = [[cID[1],cID[2]]]

		### edit staple colors
		for cID in colors_staples:
			j_edit["vstrands"][cID[0]]["stap_colors"] = [[cID[1],cID[2]]]

	### write caDNAno file
	with open(cadEditFile, 'w') as f:
		json.dump(j_edit,f,separators=(',',':'))


################################################################################
### Calculations Old

### adjust all nucleotides one-by-one
def DNAfoldifyOld(scaffold, staples, colors_scaffold, colors_staples, debug):

	### loop over scaffold spots
	print("Shifting scaffold features...")
	if debug: print("")
	for ni in range(len(scaffold)):
		if isNucleotide(scaffold[ni]):
			dirs53 = getDirsScaf(scaffold, ni)
			scaffold, colors_scaffold = shiftFeature(scaffold, staples, colors_scaffold, ni, dirs53, debug)

	### loop over staple spots
	print("Shifting staple features...")
	if debug: print("")
	for ni in range(len(staples)):
		if isNucleotide(staples[ni]):
			dirs53 = getDirsStap(staples, ni)
			staples, colors_staples = shiftFeature(staples, scaffold, colors_staples, ni, dirs53, debug)

	### results
	return scaffold, staples, colors_scaffold, colors_staples


### check if nucleotide is end that needs shifting, then shift if necessary
def shiftFeature(arr, arr_comp, colors, ni, dirs53, debug):

	### check for misplaced 5p end
	if arr[ni][2] != arr[ni][0] and (2*ni+dirs53[0]+1)%16 != 0:
		dir_head = dirs53[0]
		dir_tail = dirs53[1]
		col_head = 2
		col_tail = 4

	### check for misplaced 3p end
	elif arr[ni][4] != arr[ni][0] and (2*ni+dirs53[1]+1)%16 != 0:
		dir_head = dirs53[1]
		dir_tail = dirs53[0]
		col_head = 4
		col_tail = 2

	### otherwise, all is well
	else:
		return arr, colors

	### initialize
	stuck_ext = False
	stuck_con = False
	found_ext = False
	found_con = False
	nnt_shift = 0

	### look for suitable shift location
	while True:
		nnt_shift += 1

		### extension
		if not stuck_ext:
			stuck_ext, found_ext = checkExtension(arr, arr_comp, ni, nnt_shift, dir_head, dir_tail, col_head, col_tail)

		### contraction
		if not stuck_con:
			stuck_con, found_con = checkContraction(arr, arr_comp, ni, nnt_shift, dir_head, dir_tail, col_head, col_tail)

		### check for success
		if found_ext or found_con:
			if found_ext and found_con:
				dir_shift = dirs53[0]
			elif found_ext:
				dir_shift = dir_head
			else:
				dir_shift = dir_tail
			break

		### check for failure:
		if stuck_ext and stuck_con:
			print("Warning: No suitable shift location found.")
			if debug: print("")
			return arr, colors

	### determine connection
	if arr[ni][col_head] == -1:
		n_conn = [-1,-1]
		ni_conn = -1
	else:
		n_conn = arr[ni][col_head:col_head+2]
		ni_conn = findNucleotide(n_conn, arr)

	### debug output
	if debug:
		print('------ shift ------')
		print(f"nucleotide index: {ni}")
		print(f"shift direction:  {dir_shift}")
		print(f"shift distance:   {nnt_shift}\n")

	### update color identifiers if necessary
	if arr[ni][col_head] == -1 and dir_head == dirs53[0] and dir_shift == dir_tail:
		ci = findNucleotide(arr[ni][:2], colors)
		colors[ci][:2] = [ arr[ni][0], arr[ni][1]+nnt_shift*dir_shift ]

	### shift, one-by-one
	for s in range(nnt_shift):
		ni_back = ni + (s-1)*dir_shift
		ni_curr = ni + (s+0)*dir_shift
		ni_spot = ni + (s+1)*dir_shift
		ni_next = ni + (s+2)*dir_shift

		### extension
		if dir_shift == dir_head:

			### check nucleotide in spot about to be overtaken exists
			if isNucleotide(arr[ni_spot]):

				### if end
				if arr[ni_spot][col_tail] == -1:
					arr[ni_next][col_tail+0] = -1
					arr[ni_next][col_tail+1] = -1

				### if crossover
				else:
					n_spot_conn = arr[ni_spot][col_tail:col_tail+2]
					ni_spot_conn = findNucleotide(n_spot_conn, arr)
					arr[ni_next][col_tail+0] = arr[ni_spot_conn][0]
					arr[ni_next][col_tail+1] = arr[ni_spot_conn][1]
					arr[ni_spot_conn][col_head+0] = arr[ni_next][0]
					arr[ni_spot_conn][col_head+1] = arr[ni_next][1]

			### shift head forward
			arr[ni_curr][col_head+0] = arr[ni_curr][0]
			arr[ni_curr][col_head+1] = arr[ni_curr][1]+dir_shift
			arr[ni_spot][col_tail+0] = arr[ni_spot][0]
			arr[ni_spot][col_tail+1] = arr[ni_spot][1]-dir_shift
			arr[ni_spot][col_head+0] = n_conn[0]
			arr[ni_spot][col_head+1] = n_conn[1]

		### contraction
		if dir_shift == dir_tail:

			### shift head backward
			arr[ni_spot][col_head+0] = n_conn[0]
			arr[ni_spot][col_head+1] = n_conn[1]
			arr[ni_curr][col_tail+0] = -1
			arr[ni_curr][col_tail+1] = -1

			### check if nucleotide in spot about to be left alone
			if isNucleotide(arr[ni_back]):

				### if end
				if arr[ni_back][col_tail] == -1:
					arr[ni_curr][col_head+0] = arr[ni_back][0]
					arr[ni_curr][col_head+1] = arr[ni_back][1]
					arr[ni_back][col_tail+0] = arr[ni_curr][0]
					arr[ni_back][col_tail+1] = arr[ni_curr][1]

				### if crossover
				else:
					n_back_conn = arr[ni_back][col_tail:col_tail+2]
					ni_back_conn = findNucleotide(n_back_conn, arr)
					arr[ni_curr][col_tail+0] = arr[ni_back_conn][0]
					arr[ni_curr][col_tail+1] = arr[ni_back_conn][1]
					arr[ni_back_conn][col_head+0] = arr[ni_curr][0]
					arr[ni_back_conn][col_head+1] = arr[ni_curr][1]
			else:
				arr[ni_curr][col_head+0] = -1
				arr[ni_curr][col_head+1] = -1

	### make the other side of crossover point to new location
	if ni_conn != -1:
		arr[ni_conn][col_tail+1] = arr[ni][1] + nnt_shift*dir_shift

	### results
	return arr, colors


################################################################################
### Utility Functions

### determine whether extension is feasible
def checkExtension(arr, arr_comp, ni, nnt_shift, dir_head, dir_tail, col_head, col_tail):
	ni_shift = ni + nnt_shift*dir_head
	ni_shift_next = ni + (nnt_shift+1)*dir_head
	stuck = False
	found = False

	### check if nucleotide in potential spot exists and has complement with head-direction feature
	if isNucleotide(arr[ni_shift]) and isNucleotide(arr_comp[ni_shift]) and arr_comp[ni_shift][col_tail] != arr_comp[ni_shift][0]:
		stuck = True

	### check if nucleotide in potential spot complement has anti-head-direction feature
	elif isNucleotide(arr_comp[ni_shift]) and arr_comp[ni_shift][col_head] != arr_comp[ni_shift][0]:
		stuck = True

	### check if nucleotide in next potential spot has head-direction feature
	elif isNucleotide(arr[ni_shift_next]) and arr[ni_shift_next][col_head] != arr_comp[ni_shift_next][0]:
		stuck = True

	### check if features are allowed at potential spot
	if not stuck and (2*ni_shift+dir_head+1)%16 == 0:
		found = True

	### result
	return stuck, found


### determine whether contraction is feasible
def checkContraction(arr, arr_comp, ni, nnt_shift, dir_head, dir_tail, col_head, col_tail):
	ni_curr = ni + (nnt_shift-1)*dir_tail
	ni_shift = ni + nnt_shift*dir_tail
	stuck = False
	found = False

	### check if nucleotide in potential spot has anti-head-direction feature
	if arr[ni_shift][col_tail] != arr[ni_shift][0]:
		stuck = True

	### check if nucleotide in current spot complement has anti-head-direction feature
	elif isNucleotide(arr_comp[ni_curr]) and arr_comp[ni_curr][col_head] != arr_comp[ni_curr][0]:
		stuck = True

	### check if features are allowed at potential spot
	elif not stuck and (2*ni_shift+dir_head+1)%16 == 0:
		found = True

	### result
	return stuck, found


### check if nucleotide exists
def isNucleotide(n):
	return (n[2] != -1 or n[4] != -1)


### check if nucleotide has crossover on given side (2 for 5p, 4 for 3p)
def isCrossover(n, col):
	return (n[col] != n[0] and n[col] != -1)


### get 5p and 3p directions for scaffold nucletide
def getDirsScaf(scaffold, ni):
	dirs53 = [0,0]
	dirs53[0] = 2*(scaffold[ni][0] % 2) - 1
	dirs53[1] = 1 - 2*(scaffold[ni][0] % 2)
	return dirs53


### get 5p and 3p directions for staple nucleotide
def getDirsStap(staples, ni):
	dirs53 = [0,0]
	dirs53[0] = 1 - 2*(staples[ni][0] % 2)
	dirs53[1] = 2*(staples[ni][0] % 2) - 1
	return dirs53


### find array index where the first two values match the nucleotide
def findNucleotide(n, arr):
	for ai, item in enumerate(arr):
		if item[:2] == n:
			return ai
	print("Error: No matching entry found")
	sys.exit()


### run the script
if __name__ == "__main__":
	main()
	print()

