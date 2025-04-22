import armament as ars
import utils
import argparse
from ovito import scene
from ovito.io import import_file
from ovito.vis import Viewport, SimulationCellVis
from ovito.modifiers import LoadTrajectoryModifier
from ovito.modifiers import ComputePropertyModifier
from ovito.modifiers import DeleteSelectedModifier
import os

## Description
# this script loads the geometry and trajectory files from several simulation
  # "copies" (same simulation, different random seeds) and writes an ovito
  # session state file that visualizes all the trajectories at once.
# source folder must contain "copies.txt" file, which contains the names of the
  # directories containing the simulations to analyze.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',	type=str,	default=None,	help='name of copies file, which contains a list of simulation folders')	
	parser.add_argument('--simFold',	type=str,	default=None,	help='name of simulation folder, should exist within current directory')
	parser.add_argument('--rseed',		type=int,	default=1,		help='random seed, used to find simFold if necessary')

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	simFold = args.simFold
	rseed = args.rseed

	### scaffold and staple colors
	scaf_color = ars.getColor("orchid")
	stap_color = ars.getColor("silver")


################################################################################
### Heart

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(copiesFile, simFold, rseed)

	### initialize pipeline
	geoFile = simFolds[0] + "analysis/geometry_vis.in"
	ars.testFileExist(geoFile,"geometry")
	pipeline = import_file(geoFile, atom_style="molecular")
	pipeline.add_to_scene()

	### prepare basic DNAfold scene
	pipeline = utils.setOvitoBasics(pipeline)

	### add trajectories
	visible = False
	for i in range(nsim):

		### select trajectory
		simIndex = nsim-i-1
		if simIndex == 0:
			visible = True

		### add to pipeline
		datFile = simFolds[nsim-i-1] + "analysis/trajectory_centered.dat"
		traj_pipeline = import_file(datFile, multiple_frames=True)
		traj_mod = LoadTrajectoryModifier(enabled=visible)
		traj_mod.source.load(datFile)
		pipeline.modifiers.append(traj_mod)

	### set colors
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Color',expressions=[f'(ParticleType==1)?{scaf_color[0]}/255:{stap_color[0]}/255',f'(ParticleType==1)?{scaf_color[1]}/255:{stap_color[1]}/255',f'(ParticleType==1)?{scaf_color[2]}/255:{stap_color[2]}/255']))
	
	### remove reserved staples, or all staples
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Selection',expressions=['Position.X==0']))
	pipeline.modifiers.append(ComputePropertyModifier(enabled=False,output_property='Selection',expressions=['ParticleType!=1']))
	pipeline.modifiers.append(DeleteSelectedModifier())

	### write ovito file
	ovitoFile = "analysis/vis_folding.ovito"
	scene.save(ovitoFile)
	pipeline.remove_from_scene()


### run the script
if __name__ == "__main__":
	main()
	print()

