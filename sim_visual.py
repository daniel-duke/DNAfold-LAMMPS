import armament as ars
import utils
import argparse
from ovito import scene
from ovito.io import import_file
from ovito.modifiers import LoadTrajectoryModifier
from ovito.modifiers import ComputePropertyModifier
from ovito.modifiers import DeleteSelectedModifier
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os

## Description
# this script reads a DNAfold trajectory (or a batch of trajectories) and
  # writes an ovito session state file that visualizes the simulation (or 
  # all the simulations) at once.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',		type=str,	default=None,	help='name of copies file (first column - simulation folder names)')	
	parser.add_argument('--simFold',		type=str,	default=None,	help='name of simulation folder, used if no copies file, defaults to current folder')
	parser.add_argument('--doUnification',	type=int,	default=False,	help='whether to unwrap the scaffold and place hybridized staples ')
	parser.add_argument('--doAlignment',	type=int,	default=False,	help='whether to align the principal axes of the scaffold with the simulation box when sufficiently folded')
	parser.add_argument('--align_type',		type=str,	default='rmsd',	help='if aligning, what method to use (rmsd, pcs)')
	parser.add_argument('--align_cut',		type=float,	default=0.8,	help='if aligning, fraction of native scaffold hybridizations required to deem the scaffold "sufficiently folded"')
	parser.add_argument('--axis_thetas',	type=arr,	default=0,		help='if aligning, rotation angles (in degrees) of the aligned positions about the three coordinate axes (x,y,z)')
	parser.add_argument('--cadFile',		type=str,	default=None,	help='if aligning by RMSD, name of caDNAno file, for initializing positions')
	parser.add_argument('--topFile',		type=str, 	default=None,	help='if aligning by RMSD and using oxdna positions, name of topology file')
	parser.add_argument('--confFile',		type=str, 	default=None,	help='if aligning by RMSD and using oxdna positions, name of conformation file')
	parser.add_argument('--coarse_time',	type=int,	default=1,		help='if unifying or aligning, coarse factor for time steps')
	parser.add_argument('--hideStap',		type=int,	default=False,	help='whether to hide all staples from view, only showing scaffold')
	parser.add_argument('--win_render',		type=str,	default='none',	help='what window to render (none, front, side_ortho, side_perspec, corner)')
	parser.add_argument('--frame_rate',		type=float,	default=10,		help='if rendering, frame rate of movie (frames per second)')
	parser.add_argument('--frame_start',	type=int,	default=0,		help='if rendering, first frame of the movie')
	parser.add_argument('--frame_stop',		type=int,	default=None,	help='if rendering, last frame of the movie (None for last step)')
	parser.add_argument('--coarse_frames',	type=int,	default=1,		help='if rendering, coarse factor for frames')
	parser.add_argument('--saveOvito',		type=int,	default=True,	help='whether to save ovito session state file')

	### analysis parameters
	r12_cut_hyb = 2.0

	### scaffold and staple colors
	scaf_color = ars.getColor('orchid')
	stap_color = ars.getColor('silver')

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	simFold = args.simFold
	doUnification = args.doUnification
	doAlignment = args.doAlignment
	align_type = args.align_type
	align_cut = args.align_cut
	axis_thetas = args.axis_thetas
	cadFile = args.cadFile
	topFile = args.topFile
	confFile = args.confFile
	coarse_time = args.coarse_time
	hideStap = args.hideStap
	win_render = args.win_render
	frame_rate = args.frame_rate
	frame_start = args.frame_start
	frame_stop = args.frame_stop
	coarse_frames = args.coarse_frames
	saveOvito = args.saveOvito

	### check input
	if doAlignment and doUnification:
		print("Error: Choose either unification (with no alignment) or aligment (which includes unification).\n")
		sys.exit()
	if axis_thetas == 0:
		axis_thetas = np.zeros(3)
	elif len(axis_thetas) != 3:
		print("Error: Rotation of ideal positions must be 3 comma-separated values (one rotation for each axis).\n")
		sys.exit()

	### check for files required for RMSD alignment
	if doAlignment and align_type == 'rmsd':
		if cadFile is None:
			print("Error: caDNAno file required for RMSD alignment.\n")
			sys.exit()
		if topFile is not None and confFile is not None:
			position_src = 'oxdna'
		else:
			position_src = 'cadnano'
			if topFile is not None:
				print("Flag: oxDNA topology file provided without configuration file, using caDNAno positions.")
			if confFile is not None:
				print("Flag: oxDNA configuration file provided without topology file, using caDNAno positions.")
	

################################################################################
### Heart

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(copiesFile, simFold)

	### get pickled data
	connFile = "analysis/connectivity_vars.pkl"
	strands, complements, n_scaf, scaf_shift = readConn(connFile)

	### get ideal positions
	r_ideal = None
	if doAlignment and align_type == 'rmsd':
		if position_src == 'cadnano':
			r_ideal = utils.initPositionsCaDNAno(cadFile, scaf_shift)[0]
		if position_src == 'oxdna':
			r_ideal = utils.initPositionsOxDNA(cadFile, topFile, confFile, scaf_shift)[0]

	### initialize pipeline
	geoFile = simFolds[0] + "analysis/geometry_vis.in"
	ars.checkFileExist(geoFile,"geometry")
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

		### get centered trajectory
		datFile = simFolds[nsim-i-1] + "analysis/trajectory_centered.dat"

		### for nice output
		if doUnification or doAlignment:
			print()

		### unify trajectory
		if doUnification:

			### read data
			points, _, dbox, used_every = ars.readAtomDump(datFile, coarse_time=coarse_time, ignorePBC=True, getUsedEvery=True)

			### calculations
			points_unified = unifyTrajectory(points, dbox, strands, complements, n_scaf, r12_cut_hyb)
			points_final = rotateAxes(points_unified, axis_thetas)

			### write data
			outDatFile = simFolds[nsim-i-1] + "analysis/trajectory_unified.dat";
			writeAtomDump(outDatFile, dbox, points_final, strands, used_every)
			datFile = outDatFile

		### align trajectory
		if doAlignment:

			### read data
			hybFile = simFolds[nsim-i-1] + "analysis/hyb_status.dat"
			hyb_status = utils.readHybStatus(hybFile, coarse_time=coarse_time)
			points, _, dbox, used_every = ars.readAtomDump(datFile, coarse_time=coarse_time, ignorePBC=True, getUsedEvery=True)

			### calculations
			points_aligned = alignTrajectory(points, hyb_status, dbox, strands, complements, n_scaf, align_type, align_cut, r12_cut_hyb, r_ideal)
			points_final = rotateAxes(points_aligned, axis_thetas)

			### write data
			outDatFile = simFolds[nsim-i-1] + "analysis/trajectory_aligned.dat";
			writeAtomDump(outDatFile, dbox, points_final, strands, used_every)
			datFile = outDatFile

		### add trajectory to pipeline
		traj_pipeline = import_file(datFile, multiple_frames=True)
		traj_mod = LoadTrajectoryModifier(enabled=visible)
		traj_mod.source.load(datFile)
		pipeline.modifiers.append(traj_mod)

	### set colors (scaffold color 1, staple color 2)
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Color', expressions=[f'(ParticleType==1)?{scaf_color[0]}/255:{stap_color[0]}/255', f'(ParticleType==1)?{scaf_color[1]}/255:{stap_color[1]}/255', f'(ParticleType==1)?{scaf_color[2]}/255:{stap_color[2]}/255']))
	
	### hide reserved staples
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Selection', expressions=['Position.X==0 && Position.Y==0 && Position.Z==0']))
	
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

		### save movie
		movFile = "analysis/vis_folding_" + win_render + ".mp4"
		if frame_stop is None: frame_stop = pipeline.num_frames-1
		viewport.render_anim(size=(1600,1200), filename=movFile, fps=frame_rate, range=[frame_start,frame_stop], every_nth=coarse_frames)

	### get ovito file
	ovitoFile = "analysis/vis_folding.ovito"
	if nsim == 1: ovitoFile = simFolds[0] + ovitoFile

	### write ovito file
	if saveOvito: scene.save(ovitoFile)
	pipeline.remove_from_scene()

	# plotChords(strands, complements)
	# plt.show()


################################################################################
### Plotters

def plotChords(strands, complements):
	nbead = len(strands)

	# Find scaffold beads
	scaffold_indices = np.where(strands==1)[0]
	n_scaf = len(scaffold_indices)

	# Positions of scaffold around circle
	theta = np.linspace(0, 2*np.pi, n_scaf, endpoint=False)
	x = np.cos(theta)
	y = np.sin(theta)

	# Prepare plot
	ars.magicPlot()
	fig, ax = plt.subplots(figsize=(8,8))
	ax.set_aspect("equal")
	ax.axis("off")

	# Draw scaffold beads
	ax.scatter(x, y, c="black", s=20, zorder=3)

	# Draw backbone connections (staples only)
	for i in range(nbead - 1):
		if strands[i] == strands[i+1] and strands[i] != 1:
			scaf1 = complements[i]
			scaf2 = complements[i+1]

			if scaf1 in scaffold_indices and scaf2 in scaffold_indices:
				ax.plot([x[scaffold_indices == scaf1][0],
						 x[scaffold_indices == scaf2][0]],
						[y[scaffold_indices == scaf1][0],
						 y[scaffold_indices == scaf2][0]],
						color="grey", alpha=0.6, linewidth=1.0)


################################################################################
### File Handlers

### get connectivity variables
def readConn(connFile):
	ars.checkFileExist(connFile, "connectivity")
	with open(connFile, 'rb') as f:
		params = pickle.load(f)
	strands = params['strands']
	complements = params['complements']
	n_scaf = params['n_scaf']
	circularScaf = params['circularScaf']
	scaf_shift = 0 if circularScaf else params['scaf_shift']
	return strands, complements, n_scaf, scaf_shift


### write lammps-style atom dump
def writeAtomDump(outDatFile, dbox, points, col2s, dump_every):
	nstep = points.shape[0]
	npoint = points.shape[1]
	len_npoint = len(str(npoint))
	len_ncol2 = len(str(max(col2s)))
	len_dbox = len(str(int(dbox/2)))
	with open(outDatFile, 'w') as f:
		for i in range(nstep):
			f.write(f"ITEM: TIMESTEP\n{i*dump_every}\n")
			f.write(f"ITEM: NUMBER OF ATOMS\n{npoint}\n")
			f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} xlo xhi\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} ylo yhi\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} zlo zhi\n")
			f.write("ITEM: ATOMS id mol xs ys zs\n")
			for j in range(npoint):
				f.write(f"{j+1:<{len_npoint}} " + \
						f"{col2s[j]:<{len_ncol2}}  " + \
						f"{points[i,j,0]/dbox+1/2:11.8f} " + \
						f"{points[i,j,1]/dbox+1/2:11.8f} " + \
						f"{points[i,j,2]/dbox+1/2:11.8f}\n")


################################################################################
### Calculation Managers

### modify trajectory to keep scaffold whole (and centered) and keep hybridized staples on the scaffold
def unifyTrajectory(points, dbox, strands, complements, n_scaf, r12_cut_hyb):
	print("Unifying trajectory...")

	### get dimensions
	nstep = points.shape[0]
	nbead = points.shape[1]
	nstrand = max(strands)

	### initialize
	points_unified = np.zeros((nstep,nbead,3))

	### loop over steps
	for i in range(nstep):

		### unwrap scaffold, keeping the chain unbroken
		r_scaf_unwrapped = ars.unwrapChain(points[i,:n_scaf], dbox)

		### center around unwrapped scaffold
		com_scaf = np.mean(r_scaf_unwrapped, axis=0)
		points_unified[i,:n_scaf] = r_scaf_unwrapped - com_scaf

		### initialize strand reference points
		refs_centered = np.zeros((nstrand,3))

		### loop over staple beads, setting reference to complementary scaffold bead if hybridized and not dummy
		for j in range(n_scaf, nbead):
			if len(complements[j]) > 0 and not all(points[i,j]==0):
				c = complements[j][0]-1
				if np.linalg.norm(ars.applyPBC( points[i,j]-points[i,c], dbox )) < r12_cut_hyb:
					ref = points_unified[i,c]
					if all(refs_centered[strands[j]-1]==0):
						refs_centered[strands[j]-1] = ref

		### loop over staple strands, setting reference to staple com if not set to scaffold complement
		for j in range(1,nstrand):
			if all(refs_centered[j]==0):
				r_stap_com = ars.calcCOM(points[i,strands-1==j], dbox)
				refs_centered[j] = ars.applyPBC(r_stap_com-com_scaf, dbox)

		### unwrap the staple beads about their reference
		for j in range(n_scaf,nbead):
			ref = refs_centered[strands[j]-1]
			points_unified[i,j] = ref + ars.applyPBC( points[i,j] - com_scaf - ref, dbox )

	### result
	return points_unified


### rotate trajectory to align principal components of scaffold with simulation box, then unify
def alignTrajectory(points, hyb_status, dbox, strands, complements, n_scaf, align_type, align_cut, r12_cut_hyb, r_ideal=None):
	print("Aligning trajectory...")

	### get dimensions
	nstep = points.shape[0]
	nbead = points.shape[1]

	### place hybridized staples on scaffold
	points = unifyTrajectory(points, dbox, strands, complements, n_scaf, r12_cut_hyb)

	### initialization
	points_aligned = np.zeros((nstep,nbead,3))
	R_prev = None
	R_curr = None
	fixed = False

	### align trajectory, working backwards
	for i in range(nstep):
		step = nstep-i-1
		R_prev = R_curr

		### check whether axes have been fixed
		if fixed == False:

			### if insufficient hybridizations, freeze axes
			if sum(hyb_status[step,:n_scaf])/n_scaf < align_cut:
 
				### check if this is the last step
				if i == 0:
					print("Flag: Scaffold not sufficiently folded for alignment.")
					points_aligned = points
					break

				### fix reference frame
				print(f"Alignment axes fixed at step {step+1}")
				fixed = True

			### otherwise, get axes
			else:

				### aligning by principal components
				if align_type == 'pcs':
					R_curr = ars.alignPCs(points[step], n_scaf, [2,1,0], getPCs=True)[1]

					### match directions with previous axes
					if R_prev is not None:
						dots = np.sum(R_prev * R_curr, axis=0)
						R_curr[:,dots<0] *= -1

				### aligning by RMSD
				elif align_type == 'rmsd':

					### check for ideal positions
					if r_ideal is None:
						print("Error: ideal positions necessary for aligning by RMSD.\n")
						sys.exit()

					### do the hard work
					R_curr = utils.kabschAlgorithm(points[step], r_ideal, n_scaf, getR=True)[1]

				### error
				else: 
					print("Error: Unrecognized alignment type.\n")
					sys.exit()

		### perform alignment
		points_aligned[step] = points[step] @ R_curr

	### place hybridized staples on scaffold
	points_unified = unifyTrajectory(points_aligned, dbox, strands, complements, n_scaf, r12_cut_hyb)

	### result
	return points_unified


################################################################################
### Calculation Managers

### define float array argument
def arr(arg):
	return list(map(float, arg.split(',')))


### apply rotations about the coordinate axes
def rotateAxes(points, thetas, extrinsic=False):

	### notes
	# this function maintains the row-vector convention throughout all the
	  # calculations; in practice ,this means the 2D transformation matrix is
	  # the transpose of the standard (column-vector) version, and the order
	  # of extrinsic vs intrinsic matrix multiplications is swapped.
	# by default, this code uses intrinsic rotations, simply because they are
	  # easier to visualize.
	# this code uses the x-y'-z'' order of rotations, which does not match any
	  # standard convention, simply because x-axis rotations are the most likely
	  # to be desired and z-axis rotations are the least likely.

	### initialize
	R_total = np.eye(3)
	axes = np.arange(3)

	### loop over axes
	for axis, theta in enumerate(thetas):
		i,j = np.roll(axes,-axis)[1:]
		theta_rad = np.deg2rad(theta)
		R = np.eye(3)

		### rotation matrix
		R[i,i] = np.cos(theta_rad)
		R[i,j] = np.sin(theta_rad)
		R[j,i] = -np.sin(theta_rad)
		R[j,j] = np.cos(theta_rad)

		### add rotation
		if not extrinsic:
			R_total = R @ R_total
		else:
			R_total = R_total @ R

	### result
	return points @ R_total


### run the script
if __name__ == "__main__":
	main()
	print()

