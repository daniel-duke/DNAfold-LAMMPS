import numpy as np

class parameters:
	def __init__( self, nstep, nstep_relax, dump_every, dt, dbox, forceBind, dehyb, debug, rseed,
				  nnt_per_bead, circularScaf, reserveStap, stap_copies, T, r_h_bead, visc, 
				  sigma, epsilon, r12_eq, k_x, r12_cut_hyb, U_hyb, dsLp):

		### set parameters
		self.nstep = nstep
		self.nstep_relax = nstep_relax
		self.dump_every = dump_every
		self.dt = dt
		self.dbox = dbox
		self.forceBind = forceBind
		self.dehyb = dehyb
		self.debug = debug
		self.rseed = rseed
		self.nnt_per_bead = nnt_per_bead
		self.circularScaf = circularScaf
		self.reserveStap = reserveStap
		self.stap_copies = stap_copies
		self.T = T
		self.gamma_t = 6 * np.pi * visc * r_h_bead
		self.sigma = sigma
		self.epsilon = 6.96*epsilon
		self.r12_cut_WCA = sigma*2**(1/6)
		self.r12_eq = r12_eq
		self.k_x = 6.96*k_x
		self.r12_cut_hyb = r12_cut_hyb
		self.U_hyb = 6.96*U_hyb
		self.k_theta = dsLp*0.0138*T/r12_eq
		self.n_scaf = 0
		self.n_stap = 0
		self.n_ori = 0
		self.nbead = 0
		self.nstrand = 0

		### conditional values
		if stap_copies > 1:
			self.forceBind = False
			if forceBind == True:
				print("Flag: Cannot force bind when using multiple staple copies.")

		### error messages
		if dump_every%2 != 0:
			print("Error: Dump frequency must be an even number (for write restart purposes).")
			sys.exit()


	def record(self, paramsFile):
		with open(paramsFile,'w') as f:
			f.write(f"nstep           {self.nstep:0.0f}\n")
			f.write(f"nstep_relax     {self.nstep_relax:0.0f}\n")
			f.write(f"dump_every      {self.dump_every:0.0f}\n")
			f.write(f"dt              {self.dt}\n")
			f.write(f"dbox            {self.dbox:0.2f}\n")
			f.write(f"forceBind       {self.forceBind}\n")
			f.write(f"dehyb           {self.dehyb}\n")
			f.write(f"debug           {self.debug}\n")
			f.write(f"nnt_per_bead    {self.nnt_per_bead:0.0f}\n")
			f.write(f"circularScaf    {self.circularScaf}\n")
			f.write(f"reserveStap     {self.reserveStap}\n")
			f.write(f"stap_copies     {self.stap_copies}\n")
			f.write(f"T               {self.T}\n")
			f.write(f"sigma           {self.sigma:0.2f}\n")
			f.write(f"epsilon         {self.epsilon:0.2f}\n")
			f.write(f"r12_eq          {self.r12_eq:0.2f}\n")
			f.write(f"k_x             {self.k_x:0.2f}\n")
			f.write(f"r12_cut_hyb     {self.r12_cut_hyb:0.2f}\n")
			f.write(f"U_hyb           {self.U_hyb:0.2f}\n")

