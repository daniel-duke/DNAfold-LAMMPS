import armament as ars
import utils
import utilsLocal
from ovito import scene
from ovito.io import import_file
from ovito.vis import Viewport, SimulationCellVis
from ovito.modifiers import ComputePropertyModifier
from ovito.modifiers import DeleteSelectedModifier
import argparse
import numpy as np
import sys

## Description
# this script reads a caDNAno file (and optionally an oxDNA configuration
  # file) to get the positions for a DNA origami design, then it writes
  # geometry and ovito session state files for visualization.


################################################################################
### Parameters

def main():

	### where to get files
	useDanielFiles = False

	### special code to make Daniel happy
	if useDanielFiles:

		### chose design
		desID = "2HBx4"			# design identificaiton
		confTag = None			# if using oxdna positions, tag for configuration file (None for caDNAno positions)
		rstapTag = None			# if reserving staples, tag for reserved staples file (None for all staples)
		circularScaf = True		# whether to add backbone bond between scaffold ends
		scaf_shift = 0			# if linear scaffold, bead shift for cut location (3' end chasing 5' end)
		hideStap = False		# whether to hide all staples from view, only showing scaffold
		win_render = 'none'		# what window to render as png (none, front, side_ortho, side_perspec, corner)

		### get input files
		cadFile = utilsLocal.getCadFile(desID)
		rstapFile = utilsLocal.getRstapFile(desID, rstapTag) if rstapTag is not None else None

		### determine position source
		if confTag is not None:
			position_src = 'oxdna'
			topFile, confFile = utilsLocal.getOxFiles(desID, confTag)
		else:
			position_src = 'cadnano'

		### set output folder
		outFold = utilsLocal.getSimHomeFold(desID)

	### regular code for the general populace
	if not useDanielFiles:

		### get arguments
		parser = argparse.ArgumentParser()
		parser.add_argument('--cadFile',		type=str,	required=True,	help='name of caDNAno file')
		parser.add_argument('--topFile',		type=str, 	default=None,	help='if using oxdna positions, name of topology file')
		parser.add_argument('--confFile',		type=str, 	default=None,	help='if using oxdna positions, name of conformation file')
		parser.add_argument('--rstapFile',		type=str, 	default=None,	help='if reserving staples, name of reserved staples file')
		parser.add_argument('--circularScaf',	type=int,	default=True,	help='whether to add backbone bond between scaffold ends')
		parser.add_argument('--scaf_shift',		type=int,	default=0,		help='if linear scaffold, bead shift for cut location (3\' end chasing 5\' end)')
		parser.add_argument('--hideStap',		type=int,	default=False,	help='whether to hide all staples from view, only showing scaffold')
		parser.add_argument('--win_render',		type=str,	default='none',	help='what window to render (none, front, side_ortho, side_perspec, corner)')
		
		### set arguments
		args = parser.parse_args()
		cadFile = args.cadFile
		topFile = args.topFile
		confFile = args.confFile
		rstapFile = args.rstapFile
		circularScaf = args.circularScaf
		scaf_shift = args.scaf_shift
		hideStap = args.hideStap
		win_render = args.win_render

		### determine position source
		if topFile is not None and confFile is not None:
			position_src = 'oxdna'
		else:
			position_src = 'cadnano'
			if topFile is not None:
				print("Flag: oxDNA topology file provided without configuration file, using caDNAno positions.")
			if confFile is not None:
				print("Flag: oxDNA configuration file provided without topology file, using caDNAno positions.")

		### set output folder
		outFold = "./"


################################################################################
### Heart

	### get reserved staples
	reserved_strands = []
	if rstapFile is not None:
		reserved_strands = readRstap(rstapFile)

	### get positions
	if position_src == 'cadnano':
		r, strands = utils.initPositionsCaDNAno(cadFile, scaf_shift)
	if position_src == 'oxdna':
		r, strands = utils.initPositionsOxDNA(cadFile, topFile, confFile, scaf_shift)

	### prepare the data for nice redering
	types, bonds, dbox3 = prepGeoData(r, strands, reserved_strands, circularScaf)

	### write geometry
	ars.createSafeFold(outFold + "analysis")
	outGeoFile = outFold + "analysis/geometry_ideal.in"
	ars.writeGeo(outGeoFile, dbox3, r, strands, types, bonds)

	### write ovito file
	ovitoFile = outFold + "analysis/vis_ideal.ovito"
	figFile = outFold + "analysis/vis_ideal_" + win_render + ".png"
	writeOvito(ovitoFile, outGeoFile, figFile, hideStap, win_render)


################################################################################
### File Handlers

### write session state vito file that visualizes the geometry
def writeOvito(ovitoFile, outGeoFile, figFile, hideStap, win_render):

	### set colors
	scaf_color = ars.getColor("orchid")
	stap_color = ars.getColor("silver")

	### get base geometry
	pipeline = import_file(outGeoFile, atom_style="molecular")
	pipeline.add_to_scene()

	### prepare basic DNAfold scene
	pipeline = utils.setOvitoBasics(pipeline)

	### set scaffold and staple particle radii and bond widths (thin scaffold)
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Radius', expressions=['(ParticleType==1)?0.6:1']))
	pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds', output_property='Width', expressions=['(@1.ParticleType==1)?1.2:2']))

	### set colors (scaffold color 1, staple color 2)
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Color', expressions=[f'(ParticleType==1)?{scaf_color[0]}/255:{stap_color[0]}/255', f'(ParticleType==1)?{scaf_color[1]}/255:{stap_color[1]}/255', f'(ParticleType==1)?{scaf_color[2]}/255:{stap_color[2]}/255']))

	### remove reserved staples
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Selection', expressions=['ParticleType==3']))

	### add option to hide all staples
	pipeline.modifiers.append(ComputePropertyModifier(enabled=hideStap, output_property='Selection', expressions=['ParticleType!=1']))
	pipeline.modifiers.append(DeleteSelectedModifier())

	### render chosen viewport
	if win_render != 'none':
		if win_render == 'front':
			viewport = scene.viewports[0]
		elif win_render == 'side_ortho':
			viewport = scene.viewports[1]
		elif win_render == 'side_perspec':
			viewport = scene.viewports[2]
		elif win_render == 'corner':
			viewport = scene.viewports[3]
		else:
			print("Error: Unrecognized window selection.\n")
			sys.exit()

		### save figure
		viewport.render_image(size=(1600,1200), filename=figFile)

	### write ovito file
	scene.save(ovitoFile)
	pipeline.remove_from_scene()


### read staple file
def readRstap(rstapFile):

	### read reserved staples file
	ars.testFileExist(rstapFile, "reserved staples")
	with open(rstapFile, 'r') as f:
		reserved_strands = [ int(line.strip()) for line in f ]

	### return strand indices of reserved staples
	return reserved_strands


################################################################################
### Utility Functions

### get geometry data ready for visualization
def prepGeoData(r, strands, reserved_strands, circularScaf):
	n_ori = len(strands)
	n_scaf = sum(strands==1)
	n_padBox = 2

	### types
	types = np.ones(n_ori)
	for bi in range(n_scaf,n_ori):
		if strands[bi] in reserved_strands:
			types[bi] = 3
		else:
			types[bi] = 2

	### get bonds
	bonds = np.zeros((0,3),dtype=int)
	for bi in range(n_ori-1):
		if strands[bi] == strands[bi+1]:
			bonds = np.append(bonds, [[strands[bi],bi+1,bi+2]], axis=0)

	### add scaffold ends bond
	if circularScaf:
		bonds = np.append(bonds, [[1,1,n_scaf]], axis=0)

	### box diameter
	dbox3 = [ max(abs(r[:,0]))+n_padBox*2.4, max(abs(r[:,1]))+n_padBox*2.4, max(abs(r[:,2]))+n_padBox*2.72 ]
	dbox3 = [ 2*i for i in dbox3 ]

	### return results
	return types, bonds, dbox3


### run the script
if __name__ == "__main__":
	main()
	print()

