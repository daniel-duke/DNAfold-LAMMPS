import numpy as np
import sys

class parameters:

	### initialize
	def __init__( self, params):

		### set parameters
		self.rseed = params['rseed']
		self.nstep = params['nstep']
		self.nstep_relax = params['nstep_relax']
		self.dump_every = params['dump_every']
		self.dt = params['dt']
		self.dbox = params['dbox']
		self.debug = params['debug']
		self.dehyb = params['dehyb']
		self.forceBind = params['forceBind']
		self.startBound = params['startBound']
		self.circularScaf = params['circularScaf']
		self.reserveStap = params['reserveStap']
		self.stap_copies = params['stap_copies']
		self.nnt_per_bead = params['nnt_per_bead']
		self.T = params['T']
		self.gamma_t = 6*np.pi*params['visc']*params['r_h_bead']
		self.sigma = params['sigma']
		self.epsilon = 6.96*params['epsilon']
		self.r12_cut_WCA = params['sigma']*2**(1/6)
		self.r12_eq = params['r12_eq']
		self.k_x = 6.96*params['k_x']
		self.r12_cut_hyb = params['r12_cut_hyb']
		self.U_hyb = 6.96*params['U_hyb']
		self.k_theta = params['dsLp']*0.0138*params['T']/params['r12_eq']
		self.n_scaf = 0
		self.n_stap = 0
		self.n_ori = 0
		self.nbead = 0
		self.nstrand = 0

		### check force binding
		if self.stap_copies > 1 and self.forceBind:
			print("Flag: Cannot force bind when using multiple staple copies, removing forced binding.")
			self.forceBind = False

		### check dump frequency
		if self.dump_every%2 != 0:
			print("Error: Dump frequency must be an even number (for write restart purposes).")
			sys.exit()


	### record values
	def record(self, paramsFile):
		with open(paramsFile,'w') as f:
			f.write(f"nstep           {self.nstep:0.0f}\n")
			f.write(f"nstep_relax     {self.nstep_relax:0.0f}\n")
			f.write(f"dump_every      {self.dump_every:0.0f}\n")
			f.write(f"dt [ns]         {self.dt}\n")
			f.write(f"dbox            {self.dbox:0.2f}\n")
			f.write(f"debug           {self.debug}\n")
			f.write(f"dehyb           {self.dehyb}\n")
			f.write(f"forceBind       {self.forceBind}\n")
			f.write(f"startBound      {self.startBound}\n")
			f.write(f"circularScaf    {self.circularScaf}\n")
			f.write(f"reserveStap     {self.reserveStap}\n")
			f.write(f"stap_copies     {self.stap_copies}\n")
			f.write(f"T [K]           {self.T}\n")
			f.write(f"U_hyb [kcal]    {self.U_hyb/6.96:0.2f}\n")

