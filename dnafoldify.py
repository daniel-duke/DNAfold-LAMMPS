import armament as ars
import argparse
from dataclasses import dataclass
import math
import sys
import json
import copy

## Description
# this script reads a caDNAno file, makes the structure DNAfold compatible, 
  # and writes a new caDNAno file.

## To Do
# make sure no more than 2 features / 8 nucleotides
# broaden scope of shifting logic to include potentially conflicting features:
  # search until next feature
  # if feature is terminal, include that feature in shift and return
  # otherwise, if feature is before next legal location, add to shift and keep searching
  # otherwise, if feature is after next legal location, ignore and return
# take care of loops

## Fundamental Assumptions
# if both sides of a crossover have complements and the crossover is aligned,
  # the crossover is on-lattice, and thus any shift to one end of the crossover
  # must also shift the other end.
# if either side of a crossover has no complement or the crossover is misaligned,
  # the crossover is off-lattice, and thus the ends shift independently.
# scafLoops and stapLoops are relics of old code, and this if they contain
  # any values, they are removed and a warning is printed.


################################################################################
### Parameters

@dataclass
class parameters:
	cadFile: str
	cleanOnly: bool = False
	keepScafLoop: bool = False
	keepStapLoop: bool = False
	avoidAligningEnds: bool = False
	avoidHalfBeads: bool = False
	debug: bool = False

### start
def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--cadFile',			type=str, required=True, 	help='name of caDNAno file')
	parser.add_argument('--debug',				action='store_true',		help='whether to print debugging output')
	parser.add_argument('--cleanOnly',			action='store_true',		help='whether to only check compatability and clean up skips and loops')
	parser.add_argument('--keepScafLoop',		action='store_true',		help='whether to keep scafLoop entries')
	parser.add_argument('--keepStapLoop',		action='store_true',		help='whether to keep stapLoop entries')
	parser.add_argument('--avoidAligningEnds',	action='store_true',		help='whether to avoid aligning two complementary but unstacked ends')
	parser.add_argument('--avoidHalfBeads',		action='store_true',		help='whether to use half beads')

	### set arguments
	args = parser.parse_args()
	p = parameters(
		cadFile = args.cadFile,
		cleanOnly = args.cleanOnly,
		keepScafLoop = args.keepScafLoop,
		keepStapLoop = args.keepStapLoop,
		avoidAligningEnds = args.avoidAligningEnds,
		avoidHalfBeads = args.avoidHalfBeads,
		debug = args.debug)


################################################################################
### Heart

	### read the caDNAno file
	j, scaffold, staples, colors_scaffold, colors_staples, skips, loops = parseCaDNAno(p)

	### check compatability and clean up skips and loops
	p.nnt_bead = 8 if p.avoidHalfBeads else 4
	scaffold, staples, colors_scaffold, colors_staples, loops = preprocess(scaffold, staples, colors_scaffold, colors_staples, skips, loops, p)

	### shift things around
	if not p.cleanOnly:

		### full bead shift
		p.nnt_bead = 8
		p.finalShift = True if p.avoidHalfBeads else False
		scaffold, staples, colors_scaffold, colors_staples = DNAfoldify(scaffold, staples, colors_scaffold, colors_staples, p)

		### half bead shift
		if not p.avoidHalfBeads:
			p.nnt_bead = 4
			p.finalShift = True
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

		### get virtual strand index
		vi = el1["num"]

		### loop over the elements of the virtual strand
		for el2_key, el2 in el1.items():
			
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
					dir_3p = getDir3pScaf(nt[0])
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

			### check for the mysterious scafLoop
			elif el2_key == "scafLoop":
				if len(el2) != 0 and not p.allowScafLoop:
					print("Warning: scafLoop detected, removing.\n")
					j["vstrands"][el2]["scafLoop"] = []

			###  check for the mysterious stapLoop
			elif el2_key == "stapLoop":
				if len(el2) != 0 and not p.allowStapLoop:
					print("Warning: stapLoop detected, removing.\n")
					j["vstrands"][el2]["stapLoop"] = []

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

	### check if scaffold 5p end found
	if colors_scaffold is None:
		print("Error: No scaffold color found.\n")
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

		### write scaffold
		for ni in range(len(scaffold)):
			if scaffold[ni][0] == j["vstrands"][vi]["num"]:
				ni_vstrand = scaffold[ni][1]
				j_edit["vstrands"][vi]["scaf"][ni_vstrand] = scaffold[ni][2:]

		### write staples
		for ni in range(len(staples)):
			if staples[ni][0] == j["vstrands"][vi]["num"]:
				ni_vstrand = staples[ni][1]
				j_edit["vstrands"][vi]["stap"][ni_vstrand] = staples[ni][2:]

		### reset colors
		j_edit["vstrands"][vi]["scaf_colors"] = []
		j_edit["vstrands"][vi]["stap_colors"] = []

		### write skips (all zero)
		for ni_vstrand in range(len(j["vstrands"][vi]["skip"])):
			j_edit["vstrands"][vi]["skip"][ni_vstrand] = 0

		### write loops
		for ni in range(len(loops)):
			if loops[ni][0] == j["vstrands"][vi]["num"]:
				ni_vstrand = loops[ni][1]
				j_edit["vstrands"][vi]["loop"][ni_vstrand] = 0
				if loops[ni][2] != 0:
					print(f"Warning: Removing loop @ vstrand {loops[ni][0]}, nucleotide {loops[ni][1]}")

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

	### check for high skip frequency
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

	### take care of bridges
	scaffold, staples = fixBridges(scaffold, staples, 0, getDir3pScaf(0), p)
	staples, scaffold = fixBridges(staples, scaffold, 0, getDir3pStap(0), p)

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
			ntCs = copy.deepcopy(nts_hole)
			dir_3p = dir_3p_ss

			### check skip strand for end
			if isEnd(nts[ni],1,dir_3p):
				dir_head = 1
			elif isEnd(nts[ni],-1,dir_3p):
				dir_head = -1
			else:
				continue

			### check hole strand for correct topology
			if isEnd(ntCs[ni-dir_head],dir_head,-dir_3p) and isDirectConn(nts[ni],-dir_head,dir_3p):

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
				if dir_3p == getDir3pScaf(scaffold[ni][0]):
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
			ntCs = copy.deepcopy(nts_ss)
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
			if isDirectConn(ntCs[ni],1,-dir_3p) and isDirectConn(ntCs[ni],-1,-dir_3p):

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
				if dir_3p == getDir3pScaf(scaffold[ni][0]):
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
			ntCs = copy.deepcopy(nts_ss)
			dir_3p = -dir_3p_ss

			### check hole strand for crossover misaligned by 1 nucleotide
			if isCrossover(nts[ni-1],1,dir_3p) and getDntConn(nts[ni-1],1,dir_3p) == 1:
				dir_head = 1
			elif isCrossover(nts[ni+1],-1,dir_3p) and getDntConn(nts[ni+1],-1,dir_3p) == -1:
				dir_head = -1
			else:
				continue
				
			### check hole strand for direct connections
			if isDirectConn(ntCs[ni],1,-dir_3p) and isDirectConn(ntCs[ni],-1,-dir_3p):

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
				if dir_3p == getDir3pScaf(scaffold[ni][0]):
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
			ntCs = copy.deepcopy(nts_ss)
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
					if (isDirectConn(ntCs[ni],1,-dir_3p) and isDirectConn(ntCs[ni],-1,-dir_3p) and
						isDirectConn(ntCs[ni_other],1,dir_3p) and isDirectConn(ntCs[ni_other],-1,dir_3p)):

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
						if dir_3p == getDir3pScaf(scaffold[ni][0]):
							scaffold = nts
						else:
							staples = nts

	### result
	return scaffold, staples


### check for incompatible bridges and unite ssDNA bridges
def fixBridges(nts, ntCs, vstrand_ref, dir_3p_ref, p):

	### loop over nucleotides
	for ni in range(len(nts)):
		dir_3p = getDir3pRef(nts[ni][0], vstrand_ref, dir_3p_ref)

		### check first bridge end
		if isNucleotideBridge(nts[ni], dir_3p):

			### get bridge metadata
			dir_head = getDirHead(nts[ni], dir_3p)
			col_head = getCol(dir_head, dir_3p)
			nt_conn = nts[ni][col_head:col_head+2]
			ni_conn = findNucleotide(nt_conn, nts)

			### check if bridge end is shiftable and compatible
			if not isBridgeEndShiftable(ntCs, ni, dir_head, dir_3p):
				continue

			### check if connected bridge end is shiftable and compatible
			if not isBridgeEndShiftable(ntCs, ni_conn, -dir_head, dir_3p):
				continue

			### try extension of first bridge end (contraction of second)
			cost_ext, nts_ext, ntCs_ext = shiftBridge(nts, ntCs, ni, dir_3p, p)

			### try contraction of first bridge end (extension of second)
			cost_con, nts_con, ntCs_con = shiftBridge(nts, ntCs, ni_conn, dir_3p, p)

			### check for invalid solution
			if cost_ext == math.inf and cost_con == math.inf:
				print(f"Error: Could not find suitable shift location for bridge @ vstrand {nts[ni][0]}, nucleotide {nts[ni][1]}\n")
				sys.exit()

			### chose extension
			if cost_ext < cost_con:
				nts = nts_ext
				ntCs = ntCs_ext

			### choose contraction
			elif cost_con < cost_ext:
				nts = nts_con
				ntCs = ntCs_con

			### chose extension
			elif dir_head == 1:
				nts = nts_ext
				ntCs = ntCs_ext

			### choose contraction
			else:
				nts = nts_con
				ntCs = ntCs_con

	### result
	return nts, ntCs


### determine whether bridge end is shiftable, given complementary strand
def isBridgeEndShiftable(ntCs, ni, dir_head, dir_3p):

	### bridge end complement does not exist: shiftable, if compatible
	if not isNucleotide(ntCs[ni]):

		### search for complementary strand
		while True:
			ni -= 1

			### complementary strand ends with end or crossover: shiftable
			if isEnd(ntCs[ni], dir_head, -dir_3p) or isCrossover(ntCs[ni], dir_head, -dir_3p):
				return True

			### complementary strand ends with bridge: incompatible
			if isBridge(ntCs[ni], dir_head, -dir_3p):
				print(f"Error: Mismatched bridge ends @ vstrand {ntCs[ni][0]}, nucleotide {ntCs[ni][1]}\n")
				sys.exit()		

	### bridge end complement has head-direction end or crossover: shiftable
	if isEnd(ntCs[ni], dir_head, -dir_3p) or isCrossover(ntCs[ni], dir_head, -dir_3p):
		return True

	### bridge end complement has head direction bridge: non-shiftable
	if isBridge(ntCs[ni], dir_head, -dir_3p):
		return False

	### search along complementary strand
	while True:
		ni += 1

		### complementary strand ends with end or crossover: non-shiftable
		if isEnd(ntCs[ni], dir_head, -dir_3p) or isCrossover(ntCs[ni], dir_head, -dir_3p):
			return False

		### complementary strand ends with bridge: incompatible
		if isBridge(ntCs[ni], dir_head, -dir_3p):
			print(f"Error: Mismatched bridge ends @ vstrand {ntCs[ni][0]}, nucleotide {ntCs[ni][1]}\n")
			sys.exit()


### attempt to unite bridge by extending the given end
def shiftBridge(nts, ntCs, ni_ext, dir_3p, p):

	### get head/shift direction
	dir_shift = getDirHead(nts[ni_ext], dir_3p)

	### get connected nucleotide
	col_withShift, col_antiShift = getCols(dir_shift, dir_3p)
	nt_con = nts[ni_ext][col_withShift:col_withShift+2]
	ni_con = findNucleotide(nt_con, nts)

	### initialize
	nts = copy.deepcopy(nts)
	ntCs = copy.deepcopy(ntCs)
	cost = 0	

	### loop over single-spot shifts
	while True:

		### check if contracting end has complement
		if isNucleotide(ntCs[ni_con]):
			return cost, nts, ntCs

		### check if extending end has room to extend
		if isNucleotide(nts[ni_ext+dir_shift]) or isNucleotide(ntCs[ni_ext+dir_shift]) or not isRoomVstrand(ni_ext, dir_shift, nts):
			return math.inf, None, None

		### shift
		nts[ni_ext][col_withShift+0] = nts[ni_ext+dir_shift][0]
		nts[ni_ext][col_withShift+1] = nts[ni_ext+dir_shift][1]
		nts[ni_ext+dir_shift][col_antiShift+0] = nts[ni_ext][0]
		nts[ni_ext+dir_shift][col_antiShift+1] = nts[ni_ext][1]
		nts[ni_ext+dir_shift][col_withShift+0] = nts[ni_con+dir_shift][0]
		nts[ni_ext+dir_shift][col_withShift+1] = nts[ni_con+dir_shift][1]
		nts[ni_con][col_antiShift+0] = -1
		nts[ni_con][col_antiShift+1] = -1
		nts[ni_con][col_withShift+0] = -1
		nts[ni_con][col_withShift+1] = -1
		nts[ni_con+dir_shift][col_antiShift+0] = nts[ni_ext+dir_shift][0]
		nts[ni_con+dir_shift][col_antiShift+1] = nts[ni_ext+dir_shift][1]

		### update bridge ends
		ni_ext += dir_shift
		ni_con += dir_shift

		### decrease cost if extension introduces new legal spot for complement
		if isFeatureLegal(ni_ext, dir_shift, p):
			cost -= 1

		### increase cost if contraction removes legal spot for complement
		if isFeatureLegal(ni_con-dir_shift, -dir_shift, p):
			cost += 1

		### check if contracted end has head-direction end
		if isEnd(nts[ni_con], dir_shift, dir_3p):

			### shift
			nts[ni_ext][col_withShift+0] = nts[ni_ext+dir_shift][0]
			nts[ni_ext][col_withShift+1] = nts[ni_ext+dir_shift][1]
			nts[ni_ext+dir_shift][col_antiShift+0] = nts[ni_ext][0]
			nts[ni_ext+dir_shift][col_antiShift+1] = nts[ni_ext][1]
			nts[ni_ext+dir_shift][col_withShift+0] = -1
			nts[ni_ext+dir_shift][col_withShift+1] = -1
			nts[ni_con][col_antiShift+0] = -1
			nts[ni_con][col_antiShift+1] = -1
			nts[ni_con][col_withShift+0] = -1
			nts[ni_con][col_withShift+1] = -1

			### check for dangling complement
			if isNucleotide(ntCs[ni_con]):
				print(f"Warning: Breaking single-nucleotide complementary connection to accomodate bridge @ vstrand {nts[ni_con][0]}, nucleotide {nts[ni_con][1]}\n")

			### finished
			return -math.inf, nts, ntCs

		### check if contracted nucleotide has head-direction crossover
		elif isCrossover(nts[ni_con], dir_shift, dir_3p):
			nt_conn = nts[ni][col_withShift:col_withShift+2]
			ni_conn = findNucleotide(nt_conn, nts)

			### shift
			nts[ni_ext][col_withShift+0] = nts[ni_ext+dir_shift][0]
			nts[ni_ext][col_withShift+1] = nts[ni_ext+dir_shift][1]
			nts[ni_ext+dir_shift][col_antiShift+0] = nts[ni_ext][0]
			nts[ni_ext+dir_shift][col_antiShift+1] = nts[ni_ext][1]
			nts[ni_ext+dir_shift][col_withShift+0] = nts[ni_conn][0]
			nts[ni_ext+dir_shift][col_withShift+1] = nts[ni_conn][1]
			nts[ni_con][col_antiShift+0] = -1
			nts[ni_con][col_antiShift+1] = -1
			nts[ni_con][col_withShift+0] = -1
			nts[ni_con][col_withShift+1] = -1

			### update crossover nucleotide
			dir_3p_conn = getDir3pRef(nts[ni_conn][0], nts[ni_ext], dir_3p)
			dir_head_conn = getDirHead(nts[ni_conn], dir_3p_conn)
			col_head_conn = getCol(dir_head_conn, dir_3p_conn)
			nts[ni_conn][col_head_conn+0] = nts[ni_ext+dir_shift][0]
			nts[ni_conn][col_head_conn+1] = nts[ni_ext+dir_shift][1]

			### check for dangling complement
			if isNucleotide(ntCs[ni_con]):
				print(f"Warning: Breaking single-nucleotide complementary connection to accomodate bridge @ vstrand {nts[ni_con][0]}, nucleotide {nts[ni_con][1]}\n")

			### finished
			return -math.inf, nts, ntCs

		### check if contracted nucleotide has head-direction bridge
		elif isBridge(nts[ni_con], dir_shift, dir_3p):
			nt_conn = nts[ni_con][col_withShift:col_withShift+2]
			ni_conn = findNucleotide(nt_conn, nts)

			### shift
			nts[ni_ext][col_withShift+0] = nts[ni_ext+dir_shift][0]
			nts[ni_ext][col_withShift+1] = nts[ni_ext+dir_shift][1]
			nts[ni_ext+dir_shift][col_antiShift+0] = nts[ni_ext][0]
			nts[ni_ext+dir_shift][col_antiShift+1] = nts[ni_ext][1]
			nts[ni_ext+dir_shift][col_withShift+0] = nts[ni_conn][0]
			nts[ni_ext+dir_shift][col_withShift+1] = nts[ni_conn][1]
			nts[ni_conn][col_antiShift+0] = nts[ni_ext+dir_shift][0]
			nts[ni_conn][col_antiShift+1] = nts[ni_ext+dir_shift][1]
			nts[ni_con][col_antiShift+0] = -1
			nts[ni_con][col_antiShift+1] = -1
			nts[ni_con][col_withShift+0] = -1
			nts[ni_con][col_withShift+1] = -1

			### check if new connected bridge end is non-shiftable
			if not isBridgeEndShiftable(ntCs, ni_conn, -dir_shift, dir_3p):
				return -math.inf, nts, ntCs

			### update bridge ends
			ni_ext +=1
			ni_con = ni_conn

			### check for dangling complement
			if isNucleotide(ntCs[ni_con]):
				print(f"Warning: Breaking single-nucleotide complementary connection to accomodate bridge @ vstrand {nts[ni_con][0]}, nucleotide {nts[ni_con][1]}\n")


################################################################################
### Calculations

### adjust all nucleotides one-by-one
def DNAfoldify(scaffold, staples, colors_scaffold, colors_staples, p):

	### initialize
	scaffold_fixed = set()
	staples_fixed = set()

	### loop over scaffold spots
	print("Shifting scaffold crossovers...")
	if p.debug: print("")
	for ni in range(len(scaffold)):
		dir_3p = getDir3pScaf(scaffold[ni][0])
		if isIllegalCrossover(scaffold[ni], dir_3p, p):
			scaffold, staples, colors_scaffold, colors_staples, scaffold_fixed, staples_fixed = fixGroup(
			scaffold, staples, colors_scaffold, colors_staples, scaffold_fixed, staples_fixed, ni, dir_3p, p)
		elif isNucleotideCrossover(scaffold[ni],dir_3p):
			scaffold_fixed.add(ni)

	### loop over staple spots
	print("Shifting staple crossovers...")
	if p.debug: print("")
	for ni in range(len(staples)):
		dir_3p = getDir3pStap(staples[ni][0])
		if isIllegalCrossover(staples[ni], dir_3p, p):
			staples, scaffold, colors_staples, colors_scaffold, staples_fixed, scaffold_fixed = fixGroup(
			staples, scaffold, colors_staples, colors_scaffold, staples_fixed, scaffold_fixed, ni, dir_3p, p)
		elif isNucleotideCrossover(staples[ni],dir_3p):
			staples_fixed.add(ni)


	## loop over scaffold spots
	print("Shifting scaffold ends...")
	if p.debug: print("")
	for ni in range(len(scaffold)):
		dir_3p = getDir3pScaf(scaffold[ni][0])
		if isIllegalEnd(scaffold[ni], dir_3p, p):
			scaffold, staples, colors_scaffold, colors_staples, scaffold_fixed, staples_fixed = fixGroup(
			scaffold, staples, colors_scaffold, colors_staples, scaffold_fixed, staples_fixed, ni, dir_3p, p)
		elif isNucleotideEnd(scaffold[ni],dir_3p):
			scaffold_fixed.add(ni)

	### loop over staple spots
	print("Shifting staple ends...")
	if p.debug: print("")
	for ni in range(len(staples)):
		dir_3p = getDir3pStap(staples[ni][0])
		if isIllegalEnd(staples[ni], dir_3p, p):
			staples, scaffold, colors_staples, colors_scaffold, staples_fixed, scaffold_fixed = fixGroup(
			staples, scaffold, colors_staples, colors_scaffold, staples_fixed, scaffold_fixed, ni, dir_3p, p)
		elif isNucleotideEnd(staples[ni],dir_3p):
			staples_fixed.add(ni)

	## loop over scaffold spots
	print("Shifting scaffold bridges...")
	if p.debug: print("")
	for ni in range(len(scaffold)):
		dir_3p = getDir3pScaf(scaffold[ni][0])
		if isIllegalBridge(scaffold[ni], dir_3p, p):
			scaffold, staples, colors_scaffold, colors_staples, scaffold_fixed, staples_fixed = fixGroup(
			scaffold, staples, colors_scaffold, colors_staples, scaffold_fixed, staples_fixed, ni, dir_3p, p)
		elif isNucleotideBridge(scaffold[ni],dir_3p):
			scaffold_fixed.add(ni)

	### loop over staple spots
	print("Shifting staple bridges...")
	if p.debug: print("")
	for ni in range(len(staples)):
		dir_3p = getDir3pStap(staples[ni][0])
		if isIllegalBridge(staples[ni], dir_3p, p):
			staples, scaffold, colors_staples, colors_scaffold, staples_fixed, scaffold_fixed = fixGroup(
			staples, scaffold, colors_staples, colors_scaffold, staples_fixed, scaffold_fixed, ni, dir_3p, p)
		elif isNucleotideBridge(staples[ni],dir_3p):
			staples_fixed.add(ni)

	### results
	return scaffold, staples, colors_scaffold, colors_staples


### build a group of nucleotides connected to a feature and find the optimal shift
def fixGroup(nts, ntCs, colors, colorCs, nis_fixed, niCs_fixed, ni_init, dir_3p, p):

	### check for 3p feature
	dir_head = getDirHead(nts[ni_init], dir_3p)

	### get reference identity
	vstrand_ref = nts[ni_init][0]

	### try extension
	dir_shift = dir_head
	output_ext = attemptFixGroup(nts, ntCs, colors, colorCs, nis_fixed, niCs_fixed, ni_init, dir_shift, vstrand_ref, dir_3p, p)
	cost_ext = output_ext[0]
	nnt_shift_ext = output_ext[1]

	### try contraction
	dir_shift = -dir_head
	output_con = attemptFixGroup(nts, ntCs, colors, colorCs, nis_fixed, niCs_fixed, ni_init, dir_shift, vstrand_ref, dir_3p, p)
	cost_con = output_con[0]
	nnt_shift_con = output_con[1]

	### check for failure
	if cost_con == math.inf and cost_ext == math.inf:
		if p.finalShift:
			print(f"Warning: Could not find suitable shift location @ vstrand {nts[ni_init][0]}, nucleotide {nts[ni_init][1]}")
		return nts, ntCs, colors, colorCs, nis_fixed, niCs_fixed

	### choose extension
	if cost_ext < cost_con:
		result = output_ext[2:]
		dir_shift = dir_head
		nnt_shift = nnt_shift_ext

	### choose contraction
	elif cost_con < cost_ext:
		result = output_con[2:]
		dir_shift = -dir_head
		nnt_shift = nnt_shift_con

	### chose extension
	elif nnt_shift_ext < nnt_shift_con:
		result = output_ext[2:]
		dir_shift = dir_head
		nnt_shift = nnt_shift_ext

	### chose contraction
	elif nnt_shift_con < nnt_shift_ext:
		result = output_con[2:]
		dir_shift = -dir_head
		nnt_shift = nnt_shift_con

	### chose right
	else:
		dir_head == 1
		if dir_head == 1:
			result = output_ext[2:]
			nnt_shift = nnt_shift_ext
		else:
			result = output_con[2:]
			nnt_shift = nnt_shift_con

	### debug output
	if p.debug:
		print('------ shift ------')
		print(f"init vstrand:     {nts[ni_init][0]}")
		print(f"init nucleotide:  {nts[ni_init][1]}")
		print(f"shift direction   {dir_shift}")
		print(f"shift distance:   {nnt_shift}\n")

	### results
	return result


### shift a nucleotide in a given direction until its feature is legal, moving other features if necessary and possible
def attemptFixGroup(nts, ntCs, colors, colorCs, nis_fixed, niCs_fixed, ni_init, dir_shift, vstrand_ref, dir_3p_ref, p):
	
	### initialize
	ni_init_curr = ni_init
	nnt_shift = 0
	cost = 0

	### initialize
	nts = copy.deepcopy(nts)
	ntCs = copy.deepcopy(ntCs)
	colors = copy.deepcopy(colors)
	colorCs = copy.deepcopy(colorCs)
	nis_fixed = copy.deepcopy(nis_fixed)
	niCs_fixed = copy.deepcopy(niCs_fixed)

	### get core goup of moving nucleotides
	nis_group_core, niCs_group_core = buildGroupCore(nts, ntCs, ni_init, vstrand_ref, dir_3p_ref)

	### loop over single-spot shifts
	while True:

		### get group of nucleotides that will need to move
		success, nis_group, niCs_group = buildGroupFull(nts, ntCs, nis_fixed, niCs_fixed, ni_init_curr, dir_shift, vstrand_ref, dir_3p_ref, p)
		if not success:
			return math.inf, None, None, None, None, None, None, None

		### shift the nucleotides
		nts, colors = shiftGroup(nts, colors, nis_group, dir_shift, vstrand_ref, dir_3p_ref)
		ntCs, colorCs = shiftGroup(ntCs, colorCs, niCs_group, dir_shift, vstrand_ref, -dir_3p_ref)

		### update core group
		nis_group_core = { ni+dir_shift for ni in nis_group_core }
		niCs_group_core = { ni+dir_shift for ni in niCs_group_core }

		### update 
		ni_init_curr += dir_shift
		cost += len(nis_group - nis_group_core)
		cost += len(niCs_group - niCs_group_core)
		nnt_shift += 1

		### check for completion
		if isNucleotideLegal(nts[ni_init_curr], dir_3p_ref, p):
			break

	### get moved nucleotices
	nis_moved = { ni+dir_shift for ni in nis_group }
	niCs_moved = { ni+dir_shift for ni in niCs_group }

	### update fixed nucleotides list
	for ni in nis_moved:
		dir_3p = getDir3pRef(nts[ni][0], vstrand_ref, dir_3p_ref)
		if isNucleotideLegal(nts[ni], dir_3p, p):
			nis_fixed.add(ni)
	for ni in niCs_moved:
		dir_3p = getDir3pRef(ntCs[ni][0], vstrand_ref, -dir_3p_ref)
		if isNucleotideLegal(ntCs[ni], dir_3p, p):
			niCs_fixed.add(ni)

	### result
	return cost, nnt_shift, nts, ntCs, colors, colorCs, nis_fixed, niCs_fixed


### create list of nucleotides that would need to move with initializing nucleotide, no matter the direction
def buildGroupCore(nts, ntCs, ni_init, vstrand_ref, dir_3p_ref):
	nis_group = {ni_init}
	niCs_group = set()
	nis_curr = {ni_init}
	niCs_curr = set()

	### grow group layer-by-layer
	while True:

		### initialize next layer
		nis_next = set()
		niCs_next = set()

		### loop over nucleotides in current layer
		for ni in nis_curr:
			dir_3p = getDir3pRef(nts[ni][0], vstrand_ref, dir_3p_ref)
			nis_add, niCs_add = extendGroupCore(nts, ntCs, ni, dir_3p)
			nis_next.update(nis_add)
			niCs_next.update(niCs_add)

		### loop over complementary nucleotides in current layer
		for ni in niCs_curr:
			dir_3p = getDir3pRef(ntCs[ni][0], vstrand_ref, -dir_3p_ref)
			niCs_add, nis_add = extendGroupCore(ntCs, nts, ni, dir_3p)
			nis_next.update(nis_add)
			niCs_next.update(niCs_add)

		### remove potential next layer nucleotides that are already in group
		nis_next -= nis_group
		niCs_next -= niCs_group

		### check if finished
		if not nis_next and not niCs_next:
			break

		### add new nucleotides to group
		nis_group |= nis_next
		niCs_group |= niCs_next
		nis_curr = nis_next
		niCs_curr = niCs_next

	### result
	return nis_group, niCs_group


### get list of connected features
def extendGroupCore(nts, ntCs, ni, dir_3p):

	### find head direction
	dir_head = getDirHead(nts[ni], dir_3p)

	### initialize
	nis_add = set()
	niCs_add = set()

	### check if current nucleotide has complement and crossover connection
	if isNucleotide(ntCs[ni]) and isCrossover(nts[ni],dir_head,dir_3p):
		col_head = getCol(dir_head,dir_3p)
		nt_conn = nts[ni][col_head:col_head+2]
		ni_conn = findNucleotide(nt_conn, nts)

		### check if crossover nucleotide is aligned and has complement
		if nts[ni_conn][1] == nts[ni][1] and isNucleotide(ntCs[ni_conn]):
			if getDir3pRef(nts[ni_conn][0], nts[ni][0], dir_3p) == dir_3p:
				nis_add.add(ni_conn)

	### check if nucleotide complementary to current spot has head-direction feature
	if isNucleotide(ntCs[ni]) and not isDirectConn(ntCs[ni],dir_head,-dir_3p):
		niCs_add.add(ni)

	### check if nucleotide opposite to head exists
	if isNucleotide(nts[ni+dir_head]):
		nis_add.add(ni+dir_head)

	### result
	return nis_add, niCs_add


### create list of nucleotides that would need to move to accomodate shifting the initializing nucleotide
def buildGroupFull(nts, ntCs, nis_fixed, niCs_fixed, ni_init, dir_shift, vstrand_ref, dir_3p_ref, p):
	nis_group = {ni_init}
	niCs_group = set()
	nis_curr = {ni_init}
	niCs_curr = set()

	### grow group layer-by-layer
	while True:

		### initialize next layer
		nis_next = set()
		niCs_next = set()

		### loop over nucleotides in current layer
		for ni in nis_curr:

			dir_3p = getDir3pRef(nts[ni][0], vstrand_ref, dir_3p_ref)
			nis_add, niCs_add = extendGroupFull(nts, ntCs, ni, dir_shift, dir_3p, p)
			nis_next.update(nis_add)
			niCs_next.update(niCs_add)

		### loop over complementary nucleotides in current layer
		for ni in niCs_curr:

			dir_3p = getDir3pRef(ntCs[ni][0], vstrand_ref, -dir_3p_ref)
			niCs_add, nis_add = extendGroupFull(ntCs, nts, ni, dir_shift, dir_3p, p)
			nis_next.update(nis_add)
			niCs_next.update(niCs_add)

		### remove potential next layer nucleotides that are already in group
		nis_next -= nis_group
		niCs_next -= niCs_group

		### check if finished
		if not nis_next and not niCs_next:
			break

		### check new nucleotides for conflicts (fixed nucleotides or vstrand end)
		for ni in nis_next:
			if ni in nis_fixed or not isRoomVstrand(ni,dir_shift,nts):
				return False, None, None
		for ni in niCs_next:
			if ni in niCs_fixed or not isRoomVstrand(ni,dir_shift,ntCs):
				return False, None, None

		### add new nucleotides to group
		nis_group |= nis_next
		niCs_group |= niCs_next
		nis_curr = nis_next
		niCs_curr = niCs_next

	### result
	return True, nis_group, niCs_group


### get list of connected features
def extendGroupFull(nts, ntCs, ni, dir_shift, dir_3p, p):

	### find head direction
	dir_head = getDirHead(nts[ni], dir_3p)

	### initialize
	nis_add = set()
	niCs_add = set()

	### check if current nucleotide has crossover connection
	if isNucleotide(ntCs[ni]) and isCrossover(nts[ni],dir_head,dir_3p):
		col_head = getCol(dir_head,dir_3p)
		nt_conn = nts[ni][col_head:col_head+2]
		ni_conn = findNucleotide(nt_conn, nts)

		### check if crossover nucleotide is aligned and has complement
		if nts[ni_conn][1] == nts[ni][1] and isNucleotide(ntCs[ni_conn]):
			if getDir3pRef(nts[ni_conn][0], nts[ni][0], dir_3p) == dir_3p:
				nis_add.add(ni_conn)

	### check if nucleotide complementary to current spot has head-direction feature
	if isNucleotide(ntCs[ni]) and not isDirectConn(ntCs[ni],dir_head,-dir_3p):
		niCs_add.add(ni)

	### check if nucleotide opposite to head exists
	if isNucleotide(nts[ni+dir_head]):
		nis_add.add(ni+dir_head)

	### extension cases
	if dir_head == dir_shift:

		### check if nucleotide complementary to spot about to be overtaken has anti-head-direction feature
		if isNucleotide(ntCs[ni+dir_shift]) and not isDirectConn(ntCs[ni+dir_shift],-dir_head,-dir_3p):
			niCs_add.add(ni+dir_shift)

		### check if nucleotide in next spot to be taken has anti-head-direction feature
		if isNucleotide(nts[ni+dir_shift*2]) and not isDirectConn(nts[ni+dir_shift*2],-dir_head,dir_3p):

			### check if complements are connected
			if isDirectConn(ntCs[ni],dir_head,-dir_3p) and isDirectConn(ntCs[ni+dir_shift],dir_head,-dir_3p):
				nis_add.add(ni+dir_shift*2)

		### check if nucleotide complementary to spot about to be overtaken has head-direction feature
		if isNucleotide(ntCs[ni+dir_shift]) and not isDirectConn(ntCs[ni+dir_shift],dir_head,-dir_3p):
			if p.avoidAligningEnds:
				niCs_add.add(ni+dir_shift)

	### contraction cases
	if dir_head == -dir_shift:

		### check if nucleotide compelementary to current spot has anti-head-direction feature
		if isNucleotide(ntCs[ni]) and not isDirectConn(ntCs[ni],-dir_head,-dir_3p):
			niCs_add.add(ni)

		### check if nucleotide in spot about to be overtaken has anti-head-direction feature
		if not isDirectConn(nts[ni+dir_shift],-dir_head,dir_3p):
			nis_add.add(ni+dir_shift)

		### check if nucleotide compelementary to spot about to be overtaken has head-direction feature
		if isNucleotide(ntCs[ni+dir_shift]) and not isDirectConn(ntCs[ni+dir_shift],dir_head,-dir_3p):
			if p.avoidAligningEnds:
				niCs_add.add(ni+dir_shift)

	### result
	return nis_add, niCs_add


### shift a group of nucleotides in the given direction by one spot
def shiftGroup(nts, colors, nis_shift, dir_shift, vstrand_ref, dir_3p_ref):

	### sort nucleotides to shift
	nis_shift = list(nis_shift)
	if dir_shift == 1:
		nis_shift.sort(reverse=True)
	else:
		nis_shift.sort()

	### loop over nucleotides
	for ni in nis_shift:

		### get bearings
		dir_3p = getDir3pRef(nts[ni][0], vstrand_ref, dir_3p_ref)

		### find head direction
		dir_head = getDirHead(nts[ni],dir_3p)
		col_head, col_tail = getCols(dir_head, dir_3p)

		### determine connection
		if isEnd(nts[ni],dir_head,dir_3p):
			nt_conn = [-1,-1]
			ni_conn = -1
		else:
			nt_conn = nts[ni][col_head:col_head+2]
			ni_conn = findNucleotide(nt_conn, nts)

		### update color identifiers if necessary
		colors = updateColors(nts[ni], colors, dir_shift, dir_3p)

		### extension
		if dir_shift == dir_head:

			### shift head forward
			nts[ni+dir_shift][col_head+0] = nt_conn[0]
			nts[ni+dir_shift][col_head+1] = nt_conn[1]
			nts[ni+dir_shift][col_tail+0] = nts[ni][0]
			nts[ni+dir_shift][col_tail+1] = nts[ni][1]
			nts[ni][col_head+0] = nts[ni+dir_shift][0]
			nts[ni][col_head+1] = nts[ni+dir_shift][1]

		### contraction
		else:

			### shift head backward
			nts[ni+dir_shift][col_head+0] = nt_conn[0]
			nts[ni+dir_shift][col_head+1] = nt_conn[1]
			nts[ni][col_tail+0] = -1
			nts[ni][col_tail+1] = -1
			nts[ni][col_head+0] = -1
			nts[ni][col_head+1] = -1

		### update crossover nucleotide if necessary
		if ni_conn != -1:
			dir_3p_conn = getDir3pRef(nts[ni_conn][0], nts[ni][0], dir_3p)
			dir_head_conn = getDirHead(nts[ni_conn], dir_3p_conn)
			col_head_conn = getCol(dir_head_conn, dir_3p_conn)
			nts[ni_conn][col_head_conn+0] = nts[ni+dir_shift][0]
			nts[ni_conn][col_head_conn+1] = nts[ni+dir_shift][1]

	### results
	return nts, colors


################################################################################
### Utility Functions

### check if feature at given location is compatible with DNAfold
def isFeatureLegal(ni, dir_feature, p):
	return (2*ni+dir_feature+1)%(2*p.nnt_bead) == 0


### check if nucleotide exists
def isNucleotide(nt):
	return (nt[2] != -1 or nt[4] != -1)


### check if nucleotide exists and has direct connection in given direction
def isDirectConn(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	return (nt[col] == nt[0] and nt[col+1] == nt[1]+dir_feature)


### check if nucleotide exists and has bridge connection in given direction
def isBridge(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	return (nt[col] == nt[0] and nt[col+1] != nt[1]+dir_feature)


### check if nucleotide exists and has crossover in given direction
def isCrossover(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	return (nt[col] != nt[0] and nt[col] != -1)


### check if nucleotide exists and has end in given direction
def isEnd(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	return (isNucleotide(nt) and nt[col] == -1 and nt[col+1] == -1)


### find head direction of feature
def getDirHead(nt, dir_3p):
	if not isNucleotide(nt):
		print(f"Error: Cannot determine head direction for non-existent nucleotide @ vstrand {nt[0]}, nucleotide {nt[1]}\n")
		sys.exit()
	elif not isDirectConn(nt,1,dir_3p):
		return 1
	elif not isDirectConn(nt,-1,dir_3p):
		return -1
	else:
		print(f"Error: Cannot determine head direction for nucleotide with no feature @ vstrand {nt[0]}, nucleotide {nt[1]}\n")
		sys.exit()


### get difference in positions between nucleotide and its connection on the given side
def getDntConn(nt, dir_feature, dir_3p):
	col = 4 if dir_feature == dir_3p else 2
	if nt[col] == -1:
		print(f"Error: No connection on given side @ vstrand {nt[0]}, nucleotide {nt[1]}\n")
	return nt[col+1]-nt[1]


### get 3p direction for scaffold nucletide
def getDir3pScaf(vstrand):
	return 1 - 2*(vstrand % 2)


### get 3p direction for staple nucleotide
def getDir3pStap(vstrand):
	return 2*(vstrand % 2) - 1


### get 3p direction for nucleotide with same scaffold/staple identity as reference nucleotide
def getDir3pRef(vstrand, vstrand_ref, dir_3p_ref):
	if dir_3p_ref == getDir3pScaf(vstrand_ref):
		return getDir3pScaf(vstrand)
	else:
		return getDir3pStap(vstrand)


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


### determine if vstrand is single-stranded at nucleotide loction
def isDNAss(scaffold, staples, ni):
	if isNucleotide(scaffold[ni]) and not isNucleotide(staples[ni]):
		nts_ss = scaffold
		nts_hole = staples
		dir_3p_ss = getDir3pScaf(scaffold[ni][0])
		return nts_ss, nts_hole, dir_3p_ss
	elif not isNucleotide(scaffold[ni]) and isNucleotide(staples[ni]):
		nts_ss = staples
		nts_hole = scaffold
		dir_3p_ss = getDir3pStap(staples[ni][0])
		return nts_ss, nts_hole, dir_3p_ss
	else:
		return scaffold, staples, 0


### determine if there is room on vstrand for single spot shift
def isRoomVstrand(ni, dir_shift, nts):
	if dir_shift == -1 and ni == 0:
		return False
	if dir_shift == 1 and ni == len(nts)-1:
		return False
	if nts[ni+dir_shift][0] != nts[ni][0]:
		return False
	return True


################################################################################
### Logic Functions

def isNucleotideBridge(nt, dir_3p):
	return isBridge(nt,1,dir_3p) or isBridge(nt,-1,dir_3p)


def isNucleotideCrossover(nt, dir_3p):
	return isCrossover(nt,1,dir_3p) or isCrossover(nt,-1,dir_3p)


def isNucleotideEnd(nt, dir_3p):
	return isEnd(nt,1,dir_3p) or isEnd(nt,-1,dir_3p)


def isIllegalBridge(nt, dir_3p, p):
	return ( (isBridge(nt,dir_3p,dir_3p) and not isFeatureLegal(nt[1],dir_3p,p)) or 
			 (isBridge(nt,-dir_3p,dir_3p) and not isFeatureLegal(nt[1],-dir_3p,p)) )


def isIllegalCrossover(nt, dir_3p, p):
	return ( (isCrossover(nt,dir_3p,dir_3p) and not isFeatureLegal(nt[1],dir_3p,p)) or 
			 (isCrossover(nt,-dir_3p,dir_3p) and not isFeatureLegal(nt[1],-dir_3p,p)) )


def isIllegalEnd(nt, dir_3p, p):
	return ( (isEnd(nt,dir_3p,dir_3p) and not isFeatureLegal(nt[1],dir_3p,p)) or 
			 (isEnd(nt,-dir_3p,dir_3p) and not isFeatureLegal(nt[1],-dir_3p,p)) )


def isNucleotideLegal(nt, dir_3p, p):
	return not (isIllegalBridge(nt,dir_3p,p) or isIllegalCrossover(nt,dir_3p,p) or isIllegalEnd(nt,dir_3p,p))


### run the script
if __name__ == "__main__":
	main()
	print()

