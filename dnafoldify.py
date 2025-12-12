import armament as ars
import utils
import argparse
from dataclasses import dataclass
import numpy as np
import sys
import json
import copy

## Description
# this script reads a caDNAno file, makes the structure DNAfold compatible
  # (if possible), and writes a new caDNAno file.


################################################################################
### Parameters

@dataclass
class parameters:
	cadFile: str
	debug: bool = False
	allowBulge: bool = False
	justClean: bool = False

### get things started
def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--cadFile',	type=str, required=True, 	help='name of caDNAno file')
	parser.add_argument('--debug',		action='store_true',		help='whether to print debugging output')
	parser.add_argument('--justClean',	action='store_true',		help='whether to only check compatability and clean up skips and loops')
	parser.add_argument('--allowBulge',	action='store_true',		help='whether to allow scafLoop or stapLoop')

	### set arguments
	args = parser.parse_args()
	p = parameters(
		cadFile = args.cadFile,
		debug = args.debug,
		justClean = args.justClean,
		allowBulge = args.allowBulge)


################################################################################
### Heart

	### read the caDNAno file
	j, scaffold, staples, colors_scaffold, colors_staples, skips, loops = parseCaDNAno(p)

	### check compatability and clean up skips and loops
	scaffold, staples, colors_scaffold, colors_staples, loops = preprocess(scaffold, staples, colors_scaffold, colors_staples, skips, loops, p)

	### shift things around
	if not p.justClean:
		scaffold, staples, colors_scaffold, colors_staples = DNAfoldify(scaffold, staples, colors_scaffold, colors_staples, p)

	### write edited file
	writeCaDNAno(j, scaffold, staples, colors_scaffold, colors_staples, loops, p)


################################################################################
### File Handlers

### extract necessary info from caDNAno file
def parseCaDNAno(p):
	print("Parsing caDNAno file...")
	
	### load caDNAno file
	ars.checkFileExist(p.cadFile,'caDNAno')
	with open(p.cadFile, 'r') as f:
		json_string = f.read()
	j = json.loads(json_string)

	### initialize
	scaffold = []
	staples = []
	skips = []
	loops = []
	colors_scaffold = None
	colors_staples = []
	found5pEndScaf = False

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

					### check for multiple scaffolds
					dir_3p = getDir3pScaf(nt)
					if isEnd(nt,-dir_3p,dir_3p):
						if not found5pEndScaf:
							found5pEndScaf = True
						else:
							print("Error: Multiple scaffolds detected.\n")
							sys.exit()

			### read staple side of helix
			elif el2_key == "stap":
				
				### loop over nucleotides
				for ni_vstrand, neighbors in enumerate(el2):
					
					### store virtual strand index and nucleotide index for current nucleotide and its neighbors
					nt = [vi, int(ni_vstrand)]
					for s in neighbors:
						nt.append(int(s))
					staples.append(nt)

			### read skips
			elif el2_key == "skip":

				### loop over nucleotides
				for ni_vstrand, skip in enumerate(el2):
					skips.append([vi, ni_vstrand, skip])

			### read loops
			elif el2_key == "loop":

				### loop over nucleotides
				for ni_vstrand, loop in enumerate(el2):
					loops.append([vi, ni_vstrand, loop])

			###  check for bulges
			elif el2_key == "scafLoop" or el2_key == "stapLoop":
				if len(el2) != 0 and not p.allowBulge:
					print("Error: one-sided loop detected.\n")
					sys.exit()

			### read scaffold colors
			elif el2_key == "scaf_colors":

				### check if color defined
				if len(el2) > 0:

					### check for multiple scaffolds
					if colors_scaffold is not None:
						print("Error: Multiple scaffold colors detected.\n")
						sys.exit()

					### set scaffold color
					cID = [vi, el2[0][0], el2[0][1]]
					colors_scaffold = [cID]

			### read staple colors
			elif el2_key == "stap_colors":

				### set staple colors
				for ci in range(len(el2)):
					cID = [vi, el2[ci][0], el2[ci][1]]
					colors_staples.append(cID)

	### check if scaffold 5p end found
	if not found5pEndScaf:
		print("Error: No break found in scaffold.\n")
		sys.exit()

	### results
	return j, scaffold, staples, colors_scaffold, colors_staples, skips, loops


### write edited caDNAno file
def writeCaDNAno(j, scaffold, staples, colors_scaffold, colors_staples, loops, p):
	print("Writing caDNAno file...")

	### initialize edited json
	j_edit = copy.deepcopy(j)

	### change name
	cadEditFile = p.cadFile[:-5] + "_edited" + p.cadFile[-5:]
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

		### reset colors
		j_edit["vstrands"][vi]["scaf_colors"] = []
		j_edit["vstrands"][vi]["stap_colors"] = []

		### remove skips
		for ni_vstrand in range(len(j["vstrands"][vi]["skip"])):
			j_edit["vstrands"][vi]["skip"][ni_vstrand] = 0

	### add scaffold color
	cID = colors_scaffold[0]
	for vi in range(len(j["vstrands"])):
		if j["vstrands"][vi]["num"] == cID[0]:
			j_edit["vstrands"][vi]["scaf_colors"].append([cID[1],cID[2]])

	### add staple colors
	for cID in colors_staples:
		for vi in range(len(j["vstrands"])):
			if j["vstrands"][vi]["num"] == cID[0]:
				j_edit["vstrands"][vi]["stap_colors"].append([cID[1],cID[2]])

	### write caDNAno file
	with open(cadEditFile, 'w') as f:
		json.dump(j_edit,f,separators=(',',':'))


################################################################################
### Preprocessing

def preprocess(scaffold, staples, colors_scaffold, colors_staples, skips, loops, p):
	print("Checking compatibility and cleaning up...")

	### look for high skip frequency
	for ni in range(len(skips)-7):
		if scaffold[ni+7][0] == scaffold[ni][0]:
			if sum(1 for s in skips[ni:ni+8] if s[2] != 0) > 1:
				print(f"Error: Detected skips too close together (must be at least 8 nucleotides apart).\n")
				sys.exit()

	### fix select skip scenarios
	scaffold, staples, colors_scaffold, colors_staples = fixSkipsCase1(scaffold, staples, colors_scaffold, colors_staples, skips, p)
	scaffold, staples, colors_scaffold, colors_staples = fixSkipsCase2(scaffold, staples, colors_scaffold, colors_staples, skips, p)
	scaffold, staples = fixSkipsCase3(scaffold, staples, skips, p)
	scaffold, staples = fixSkipsCase4(scaffold, staples, skips, p)

	### look for bulges
	for ni in range(len(scaffold)):
		dir_3p_scaf = getDir3pScaf(scaffold[ni])
		dir_3p_stap = getDir3pStap(staples[ni])
		if (isBridgeConn(scaffold[ni],1,dir_3p_scaf) and isBridgeConn(staples[ni],1,dir_3p_stap) and
			getDntConn(scaffold[ni],1,dir_3p_scaf) != getDntConn(staples[ni],1,dir_3p_stap)):
				print("Error: Detected bulge.\n")
				sys.exit()

	### look for misaligned dsDNA crossovers
	# for ni in range(len(scaffold)-1):

	# 	### check scaffold
	# 	dir_3p_scaf = getDir3pScaf(scaffold[ni])
	# 	if isCrossover(scaffold[ni],dir_3p_scaf,dir_3p_scaf) and isNucleotide(staples[ni]):
	# 		nt_conn = scaffold[ni][4:6]
	# 		ni_conn = findNucleotide(nt_conn, scaffold)
	# 		if isNucleotide(staples[ni_conn]) and scaffold[ni][5] != scaffold[ni][1]:
	# 			print("Error: Detected misaligned crossover between to dsDNA strands.\n")

	# 	### check staples
	# 	dir_3p_stap = getDir3pStap(staples[ni])
	# 	if isCrossover(staples[ni],dir_3p_stap,dir_3p_stap) and isNucleotide(staples[ni]):
	# 		nt_conn = staples[ni][4:6]
	# 		ni_conn = findNucleotide(nt_conn, staples)
	# 		if isNucleotide(scaffold[ni_conn]) and (staples[ni][5]-staples[ni][1])%8 != 0:
	# 			print("Error: Detected misaligned crossover between to dsDNA strands.\n")

	### results
	return scaffold, staples, colors_scaffold, colors_staples, loops


### case 1: skipped nucleotide that creates blunt end when removed
def fixSkipsCase1(scaffold, staples, colors_scaffold, colors_staples, skips, p):
	for ni in range(len(scaffold)):
		if skips[ni][2] != 0:

			### check for ssDNA skip
			nts_ss, nts_hole, dir_3p_ss = isDNAss(scaffold, staples, ni)
			if dir_3p_ss == 0:
				continue
			
			### set moving strand
			nts = copy.deepcopy(nts_ss)
			nts_comp = copy.deepcopy(nts_hole)
			dir_3p = dir_3p_ss

			### check skip strand for end
			if isEnd(nts[ni],1,dir_3p):
				dir_head = 1
			elif isEnd(nts[ni],-1,dir_3p):
				dir_head = -1
			else:
				continue

			### check hole strand for correct topology
			if isEnd(nts_comp[ni-dir_head],dir_head,-dir_3p) and isDirectConn(nts[ni],-dir_head,dir_3p):

				### debug output
				if p.debug:
					print("Fixing case 1 skip.")

				### apply fix
				col_head, col_tail = getCols(dir_head, dir_3p)
				nts[ni][col_tail+0] = -1
				nts[ni][col_tail+1] = -1
				nts[ni-dir_head][col_head+0] = -1
				nts[ni-dir_head][col_head+1] = -1

				### apply changes
				if dir_3p == getDir3pScaf(scaffold[ni]):
					colors_scaffold = updateColors(scaffold[ni], colors_scaffold, -dir_head, dir_3p)
					scaffold = nts
				else:
					colors_staples = updateColors(staples[ni], colors_staples, -dir_head, dir_3p)
					staples = nts

	### result
	return scaffold, staples, colors_scaffold, colors_staples


### case 2: skipped nucleotide that creates nick when removed
def fixSkipsCase2(scaffold, staples, colors_scaffold, colors_staples, skips, p):
	for ni in range(len(scaffold)):
		if skips[ni][2] != 0:

			### check for ssDNA skip
			nts_ss, nts_hole, dir_3p_ss = isDNAss(scaffold, staples, ni)
			if dir_3p_ss == 0:
				continue

			### set moving strand
			nts = copy.deepcopy(nts_hole)
			nts_comp = copy.deepcopy(nts_ss)
			dir_3p = -dir_3p_ss

			### avoid similar but different topologies
			if isCrossover(nts[ni-1],1,dir_3p) and getDntConn(nts[ni-1],1,dir_3p) == 1:
				continue
			if isCrossover(nts[ni+1],-1,dir_3p) and getDntConn(nts[ni+1],-1,dir_3p) == -1:
				continue
			if (isCrossover(nts[ni-1],1,dir_3p) and isCrossover(nts[ni+1],-1,dir_3p) and 
				getDntConn(nts[ni-1],1,dir_3p) == 0 and getDntConn(nts[ni+1],-1,dir_3p) == 0):
				continue

			### check hole strand for for correct topology
			if isEnd(nts[ni-1],1,dir_3p) and isEnd(nts[ni+1],-1,dir_3p):
				dir_head = 1
			elif isEnd(nts[ni-1],1,dir_3p) and isCrossover(nts[ni+1],-1,dir_3p):
				dir_head = 1
			elif isCrossover(nts[ni-1],1,dir_3p) and isEnd(nts[ni+1],-1,dir_3p):
				dir_head = -1
			else:
				continue
				
			### check skip strand for direct connections
			if isDirectConn(nts_comp[ni],1,-dir_3p) and isDirectConn(nts_comp[ni],-1,-dir_3p):

				### debug output
				if p.debug:
					print("Fixing case 2 skip.")

				### apply fix
				col_head, col_tail = getCols(dir_head, dir_3p)
				nts[ni-dir_head][col_head+0] = nts[ni][0]
				nts[ni-dir_head][col_head+1] = nts[ni][1]
				nts[ni][col_tail+0] = nts[ni-dir_head][0]
				nts[ni][col_tail+1] = nts[ni-dir_head][1]

				### apply changes
				if dir_3p == getDir3pScaf(scaffold[ni]):
					colors_scaffold = updateColors(scaffold[ni-dir_head], colors_scaffold, dir_head, dir_3p)
					scaffold = nts
				else:
					colors_staples = updateColors(staples[ni-dir_head], colors_staples, dir_head, dir_3p)
					staples = nts

	### result
	return scaffold, staples, colors_scaffold, colors_staples


### case 3: skipped nucleotide that creates aligned crossover when removed
def fixSkipsCase3(scaffold, staples, skips, p):
	for ni in range(len(scaffold)):
		if skips[ni][2] != 0:

			### check for ssDNA skip
			nts_ss, nts_hole, dir_3p_ss = isDNAss(scaffold, staples, ni)
			if dir_3p_ss == 0:
				continue

			### set moving strand
			nts = copy.deepcopy(nts_hole)
			nts_comp = copy.deepcopy(nts_ss)
			dir_3p = -dir_3p_ss

			### check hole strand for crossover misaligned by 1 nucleotide
			if isCrossover(nts[ni-1],1,dir_3p) and getDntConn(nts[ni-1],1,dir_3p) == 1:
				dir_head = 1
			elif isCrossover(nts[ni+1],-1,dir_3p) and getDntConn(nts[ni+1],-1,dir_3p) == -1:
				dir_head = -1
			else:
				continue
				
			### check hole strand for direct connections
			if isDirectConn(nts_comp[ni],1,-dir_3p) and isDirectConn(nts_comp[ni],-1,-dir_3p):

				### debug output
				if p.debug:
					print("Fixing case 3 skip.")

				### apply fix
				col_head, col_tail = getCols(dir_head, dir_3p)
				nt_conn = nts[ni-dir_head][col_head:col_head+2]
				ni_conn = findNucleotide(nt_conn, nts)
				nts[ni-dir_head][col_head+0] = nts[ni][0]
				nts[ni-dir_head][col_head+1] = nts[ni][1]
				nts[ni][col_tail+0] = nts[ni-dir_head][0]
				nts[ni][col_tail+1] = nts[ni-dir_head][1]
				nts[ni][col_head+0] = nt_conn[0]
				nts[ni][col_head+1] = nt_conn[1]
				nts[ni_conn][col_tail+0] = nts[ni][0]
				nts[ni_conn][col_tail+1] = nts[ni][1]

				### apply changes
				if dir_3p == getDir3pScaf(scaffold[ni]):
					scaffold = nts
				else:
					staples = nts

	### result
	return scaffold, staples


### case 4: skipped pair of nucleotides that creates aligned double crossover when removed
def fixSkipsCase4(scaffold, staples, skips, p):
	for ni in range(len(scaffold)):
		if skips[ni][2] != 0:

			### check for ssDNA skip
			nts_ss, nts_hole, dir_3p_ss = isDNAss(scaffold, staples, ni)
			if dir_3p_ss == 0:
				continue

			### set moving strand
			nts = copy.deepcopy(nts_hole)
			nts_comp = copy.deepcopy(nts_ss)
			dir_3p = -dir_3p_ss

			### check hole strand for correct topology
			if (isCrossover(nts[ni-1],1,dir_3p) and isCrossover(nts[ni+1],-1,dir_3p) and
				getDntConn(nts[ni-1],1,dir_3p) == 0 and getDntConn(nts[ni+1],-1,dir_3p) == 0 and
				nts[ni-1][getCol(1,dir_3p)] == nts[ni+1][getCol(-1,dir_3p)]):


				### check other side of crossovers for skip
				nt_other = [ nts[ni-1][getCol(1,dir_3p)], nts[ni][1]]
				ni_other = findNucleotide(nt_other, nts)
				if skips[ni_other] != 0:

					### check skip strands for direct connections
					if (isDirectConn(nts_comp[ni],1,-dir_3p) and isDirectConn(nts_comp[ni],-1,-dir_3p) and
						isDirectConn(nts_comp[ni_other],1,dir_3p) and isDirectConn(nts_comp[ni_other],-1,dir_3p)):

						### debug output
						if p.debug:
							print("Fixing case 4 skip.")

						### apply fix
						dir_head = 1
						col_head, col_tail = getCols(dir_head, dir_3p)
						nts[ni-dir_head][col_head+0] = nts[ni][0]
						nts[ni-dir_head][col_head+1] = nts[ni][1]
						nts[ni][col_tail+0] = nts[ni-dir_head][0]
						nts[ni][col_tail+1] = nts[ni-dir_head][1]
						nts[ni][col_head+0] = nts[ni_other][0]
						nts[ni][col_head+1] = nts[ni_other][1]
						nts[ni_other][col_tail+0] = nts[ni][0]
						nts[ni_other][col_tail+1] = nts[ni][1]
						nts[ni_other][col_head+0] = nts[ni_other-dir_head][0]
						nts[ni_other][col_head+1] = nts[ni_other-dir_head][1]
						nts[ni_other-dir_head][col_tail+0] = nts[ni_other][0]
						nts[ni_other-dir_head][col_tail+1] = nts[ni_other][1]

						### apply changes
						if dir_3p == getDir3pScaf(scaffold[ni]):
							scaffold = nts
						else:
							staples = nts

	### result
	return scaffold, staples


################################################################################
### Calculations

### adjust all nucleotides one-by-one
def DNAfoldify(scaffold, staples, colors_scaffold, colors_staples, p):

	### loop over scaffold spots
	print("Shifting scaffold crossovers...")
	if p.debug: print("")
	for ni in range(len(scaffold)):
		dir_3p = getDir3pScaf(scaffold[ni])
		if isIllegalCrossover(scaffold[ni], dir_3p):
			scaffold, colors_scaffold = shiftFeature(scaffold, staples, colors_scaffold, ni, dir_3p, p)

	### loop over staple spots
	print("Shifting staple crossovers...")
	if p.debug: print("")
	for ni in range(len(staples)):
		dir_3p = getDir3pStap(staples[ni])
		if isIllegalCrossover(staples[ni], dir_3p):
			staples, colors_staples = shiftFeature(staples, scaffold, colors_staples, ni, dir_3p, p)

	### loop over scaffold spots
	print("Shifting scaffold ends...")
	if p.debug: print("")
	for ni in range(len(scaffold)):
		dir_3p = getDir3pScaf(scaffold[ni])
		if isIllegalEnd(scaffold[ni], dir_3p):
			scaffold, colors_scaffold = shiftFeature(scaffold, staples, colors_scaffold, ni, dir_3p, p)

	### loop over staple spots
	print("Shifting staple ends...")
	if p.debug: print("")
	for ni in range(len(staples)):
		dir_3p = getDir3pStap(staples[ni])
		if isIllegalEnd(staples[ni], dir_3p):
			staples, colors_staples = shiftFeature(staples, scaffold, colors_staples, ni, dir_3p, p)

	### results
	return scaffold, staples, colors_scaffold, colors_staples


### check if nucleotide is end that needs shifting, then shift if necessary
def shiftFeature(nts, nts_comp, colors, ni, dir_3p, p):

	### check for 3p feature
	if not isDirectConn(nts[ni],dir_3p,dir_3p):
		dir_head = dir_3p

	### check for 5p feature
	elif not isDirectConn(nts[ni],-dir_3p,dir_3p):
		dir_head = -dir_3p

	### error
	else:
		print("Error: Cannot shift nucleotide without any features.")

	### determine connection
	if isEnd(nts[ni],dir_head,dir_3p):
		nt_conn = [-1,-1]
		ni_conn = -1
	else:
		col_head = getCol(dir_head, dir_3p)
		nt_conn = nts[ni][col_head:col_head+2]
		ni_conn = findNucleotide(nt_conn, nts)

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
			stuck_ext = checkExtension(nts, nts_comp, ni, nnt_shift, dir_head, dir_3p)

			if not stuck_ext and ni_conn != -1 and isNucleotide(nts_comp[ni_conn]):
				stuck_ext = checkExtension(nts, nts_comp, ni_conn, nnt_shift, dir_head, -dir_3p)

			if not stuck_ext:
				found_ext = isFeatureAllowed(ni+dir_head*nnt_shift, dir_head)

		### contraction
		if not stuck_con:
			stuck_con = checkContraction(nts, nts_comp, ni, nnt_shift, dir_head, dir_3p)

			if not stuck_con and ni_conn != -1:
				stuck_con = checkContraction(nts, nts_comp, ni_conn, nnt_shift, dir_head, -dir_3p)

			if not stuck_con:
				found_con = isFeatureAllowed(ni-dir_head*nnt_shift, dir_head)

		### check for success
		if found_ext or found_con:
			if found_ext and found_con:
				dir_shift = 1
			elif found_ext:
				dir_shift = dir_head
			else:
				dir_shift = -dir_head
			break

		### check for failure:
		if (stuck_ext and stuck_con) or nnt_shift > 8:
			print(f"Warning: No suitable shift location found for nucleotide {nts[ni][1]} on vstrand {nts[ni][0]}.")

			if p.debug: print("")
			return nts, colors

	### debug output
	if p.debug:
		print('------ shift ------')
		print(f"vstrand:         {nts[ni][0]}")
		print(f"nucleotide:      {nts[ni][1]}")
		print(f"shift direction: {dir_shift}")
		print(f"shift distance:  {nnt_shift}\n")

	### do the shifting
	nts, colors = shift(nts, colors, ni, ni_conn, nt_conn, dir_head, dir_shift, nnt_shift, dir_3p)
	if ni_conn != -1 and isNucleotide(nts_comp[ni_conn]):
		ni_new = ni + dir_shift*nnt_shift
		nt_new = nts[ni_new][:2]
		nts, colors = shift(nts, colors, ni_conn, ni_new, nt_new, dir_head, dir_shift, nnt_shift, -dir_3p)

	### results
	return nts, colors


### actually do the shifitng
def shift(nts, colors, ni, ni_conn, nt_conn, dir_head, dir_shift, nnt_shift, dir_3p):

	### determine head and tail columns
	col_head, col_tail = getCols(dir_head, dir_3p)

	### shift, one-by-one
	for s in range(nnt_shift):
		ni_back = ni + (s-1)*dir_shift
		ni_curr = ni + (s+0)*dir_shift
		ni_spot = ni + (s+1)*dir_shift
		ni_next = ni + (s+2)*dir_shift

		### extension
		if dir_shift == dir_head:

			### update color identifiers if necessary
			colors = updateColors(nts[ni_curr], colors, dir_shift, dir_3p)
			colors = updateColors(nts[ni_spot], colors, dir_shift, dir_3p)

			### check nucleotide exists in head direction spot
			if isNucleotide(nts[ni_spot]):

				### if end
				if isEnd(nts[ni_spot],-dir_head,dir_3p):
					nts[ni_next][col_tail+0] = -1
					nts[ni_next][col_tail+1] = -1

				### if crossover or bridge
				else:
					nt_spot_conn = nts[ni_spot][col_tail:col_tail+2]
					ni_spot_conn = findNucleotide(nt_spot_conn, nts)
					nts[ni_next][col_tail+0] = nts[ni_spot_conn][0]
					nts[ni_next][col_tail+1] = nts[ni_spot_conn][1]
					nts[ni_spot_conn][col_head+0] = nts[ni_next][0]
					nts[ni_spot_conn][col_head+1] = nts[ni_next][1]

			### shift head forward
			nts[ni_curr][col_head+0] = nts[ni_curr][0]
			nts[ni_curr][col_head+1] = nts[ni_curr][1]+dir_shift
			nts[ni_spot][col_tail+0] = nts[ni_spot][0]
			nts[ni_spot][col_tail+1] = nts[ni_spot][1]-dir_shift
			nts[ni_spot][col_head+0] = nt_conn[0]
			nts[ni_spot][col_head+1] = nt_conn[1]

		### contraction
		if dir_shift == -dir_head:

			### update color identifiers if necessary
			colors = updateColors(nts[ni_curr], colors, dir_shift, dir_3p)
			colors = updateColors(nts[ni_back], colors, dir_shift, dir_3p)

			### shift head backward
			nts[ni_spot][col_head+0] = nt_conn[0]
			nts[ni_spot][col_head+1] = nt_conn[1]
			nts[ni_curr][col_tail+0] = -1
			nts[ni_curr][col_tail+1] = -1
			nts[ni_curr][col_head+0] = -1
			nts[ni_curr][col_head+1] = -1

			### check if nucleotide exists in head direction spot
			if isNucleotide(nts[ni_back]):

				### if end
				if isEnd(nts[ni_back],-dir_head,dir_3p):
					nts[ni_curr][col_head+0] = nts[ni_back][0]
					nts[ni_curr][col_head+1] = nts[ni_back][1]
					nts[ni_back][col_tail+0] = nts[ni_curr][0]
					nts[ni_back][col_tail+1] = nts[ni_curr][1]

				### if crossover or bridge
				else:
					nt_back_conn = nts[ni_back][col_tail:col_tail+2]
					ni_back_conn = findNucleotide(nt_back_conn, nts)
					nts[ni_back_conn][col_head+0] = nts[ni_curr][0]
					nts[ni_back_conn][col_head+1] = nts[ni_curr][1]
					nts[ni_curr][col_tail+0] = nts[ni_back_conn][0]
					nts[ni_curr][col_tail+1] = nts[ni_back_conn][1]
					nts[ni_curr][col_head+0] = nts[ni_back][0]
					nts[ni_curr][col_head+1] = nts[ni_back][1]
					nts[ni_back][col_tail+0] = nts[ni_curr][0]
					nts[ni_back][col_tail+1] = nts[ni_curr][1]

	### make the other side of crossover point to new location
	if ni_conn != -1:
		nts[ni_conn][col_tail+1] = nts[ni][1] + dir_shift*nnt_shift

	### results
	return nts, colors


################################################################################
### Utility Functions

### determine whether extension is feasible
def checkExtension(nts, nts_comp, ni, nnt_shift, dir_head, dir_3p):
	ni_spot = ni + dir_head*nnt_shift
	ni_next = ni + dir_head*(nnt_shift+1)
	stuck = False

	### check if nucleotide in potential spot exists and has complement with head-direction feature
	if isNucleotide(nts[ni_spot]) and isNucleotide(nts_comp[ni_spot]) and not isDirectConn(nts_comp[ni_spot],dir_head,-dir_3p):
		stuck = True

	### check if nucleotide in complement to potential spot has anti-head-direction feature
	elif isNucleotide(nts_comp[ni_spot]) and not isDirectConn(nts_comp[ni_spot],-dir_head,-dir_3p):
		stuck = True

	### check if nucleotide in next potential spot has anti-head-direction feature
	elif isNucleotide(nts[ni_next]) and not isDirectConn(nts[ni_next],-dir_head,dir_3p):
		stuck = True

	### check if nucleotide in next potential spot has head-direction feature
	elif isNucleotide(nts[ni_next]) and not isDirectConn(nts[ni_next],dir_head,dir_3p):
		stuck = True

	### result
	return stuck


### determine whether contraction is feasible
def checkContraction(nts, nts_comp, ni, nnt_shift, dir_head, dir_3p):
	ni_curr = ni - dir_head*(nnt_shift-1)
	ni_spot = ni - dir_head*nnt_shift
	stuck = False

	### check if nucleotide in potential spot has anti-head-direction feature
	if not isDirectConn(nts[ni_spot],-dir_head,dir_3p):
		stuck = True

	### check if nucleotide in current spot complement has anti-head-direction feature
	elif isNucleotide(nts_comp[ni_curr]) and not isDirectConn(nts_comp[ni_curr],-dir_head,-dir_3p):
		stuck = True

	### result
	return stuck


### check if feature at given location is compatible with DNAfold
def isFeatureAllowed(ni, dir_feature):
	return (2*ni+dir_feature+1)%16 == 0


### check if nucleotide exists
def isNucleotide(nt):
	return (nt[2] != -1 or nt[4] != -1)


### check if nucleotide exists and has direct connection on given side (2 for 5p, 4 for 3p)
def isDirectConn(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	return (nt[col] == nt[0] and nt[col+1] == nt[1]+dir_feature)


### check if nucleotide exists and has bridge connection on given side (2 for 5p, 4 for 3p)
def isBridgeConn(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	return (nt[col] == nt[0] and nt[col+1] != nt[1])


### check if nucleotide exists and has crossover on given side (2 for 5p, 4 for 3p)
def isCrossover(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	return (nt[col] != nt[0] and nt[col] != -1)


### check if nucleotide exists and has end on given side (2 for 5p, 4 for 3p)
def isEnd(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	return (isNucleotide(nt) and nt[col] == -1 and nt[col+1] == -1)


### get difference in positions between nucleotide and its connection on the given side
def getDntConn(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	if nt[col] == -1:
		print("Error: no connection on given side.\n")
	return nt[col+1]-nt[1]


### get 5p and 3p directions for scaffold nucletide
def getDir3pScaf(nt):
	return 1 - 2*(nt[0] % 2)


### get 5p and 3p directions for staple nucleotide
def getDir3pStap(nt):
	return 2*(nt[0] % 2) - 1


### get column that corresponds to given direction
def getCol(dir_feature, dir_3p):
	if dir_feature == dir_3p:
		return 4
	else:
		return 2


### get column that corresponds to given direction, and other column
def getCols(dir_feature, dir_3p):
	if dir_feature == dir_3p:
		return 4,2
	else:
		return 2,4


### find array index where the first two values match the nucleotide first two values
def findNucleotide(nt, arr):
	for ai, item in enumerate(arr):
		if item[:2] == nt[:2]:
			return ai
	print("Error: No matching entry found.\n")
	sys.exit()


### update color identifiers if moving 5p end
def updateColors(nt, colors, dir_shift, dir_3p):
	if isEnd(nt,-dir_3p,dir_3p):
		ci = findNucleotide(nt, colors)
		colors[ci][:2] = [ nt[0], nt[1]+dir_shift ]
	return colors


def isDNAss(scaffold, staples, ni):
	if isNucleotide(scaffold[ni]) and not isNucleotide(staples[ni]):
		nts_ss = scaffold
		nts_hole = staples
		dir_3p_ss = getDir3pScaf(scaffold[ni])
		return nts_ss, nts_hole, dir_3p_ss
	elif not isNucleotide(scaffold[ni]) and isNucleotide(staples[ni]):
		nts_ss = staples
		nts_hole = scaffold
		dir_3p_ss = getDir3pStap(staples[ni])
		return nts_ss, nts_hole, dir_3p_ss
	else:
		return scaffold, staples, 0


################################################################################
### Logic Functions

def isIllegalCrossover(nt, dir_3p):
	return ( (isCrossover(nt,dir_3p,dir_3p) and not isFeatureAllowed(nt[1],dir_3p)) or 
			 (isCrossover(nt,-dir_3p,dir_3p) and not isFeatureAllowed(nt[1],-dir_3p)) )


def isIllegalEnd(nt, dir_3p):
	return ( (isEnd(nt,dir_3p,dir_3p) and not isFeatureAllowed(nt[1],dir_3p)) or 
			 (isEnd(nt,-dir_3p,dir_3p) and not isFeatureAllowed(nt[1],-dir_3p)) )



### run the script
if __name__ == "__main__":
	main()
	print()

