import numpy as np
import matplotlib.pylab as plt

class HB_lattice():
    
    def __init__(self, n, t_nn, t_soc, t_nnn, lattice_spacing, g_factor, Bmax, num_map=[0, 1], B_map=0, mapRes=200, smearing=35):
        
        self.n = n
        self.t_nn = t_nn # nn
        self.t_nnn = t_nnn # nnn
        self.t_soc = t_soc # spin-orbit part
        self.a = lattice_spacing
        self.g = g_factor 
        self.Bmax = Bmax
        self.N = n**2

        self.num_map = num_map
        self.B_map = B_map
        self.mapRes = mapRes
        self.smearing = smearing
  
        self.info()
        self.init_coords()
        self.plot_eigenvalues()


    def info(self):
        print('HB lattice parameters:')
        print('lattice_size =', self.n)
        print('nn hopping =', self.t_nn, 'eV')
        print('spin-orbit component of nn hopping =', self.t_soc, 'eV')
        print('nnn hopping =', self.t_nnn, 'eV')
        print('lattice_spacing =', self.a, 'nm')
        print('g-factor =', self.g)
        print('max value of magnetic field =', self.Bmax, 'T')


    def init_coords(self):
        coord = {}
        for i in range(self.N):
            x_i = i//self.n + 1
            y_i = i%self.n + 1
            coord[i] = np.array([x_i, y_i])

            self.coord = coord
        

    def calc_eigenvalues(self, B):
        b = 0.242e-3 * B * self.a**2 # unitless  b = B * a^2 * e/h
    
        Ham_up = np.zeros((self.N, self.N), dtype=complex)
        Ham_ud = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                phase = np.exp(-2 * np.pi * 1j * b * (self.coord[i][0] + self.coord[j][0]) * (self.coord[i][1] - self.coord[j][1])/2)  
                r = np.linalg.norm(self.coord[i] - self.coord[j])
                if(r == 1):
                    Ham_up[i, j] = self.t_nn * phase  ## nn
                if(np.abs(r - 1.41421) < 1e-4):
                    Ham_up[i, j] = self.t_nnn * phase # nnn
  
        Ham_dn = np.copy(Ham_up)

        #Zeeman splitting
        for i in range(self.N):
            Ham_up[i, i] += -5.588e-5 * self.g * 0.5 * B
            Ham_dn[i, i] +=  5.588e-5 * self.g * 0.5 * B

  
        #Non-diagonal components due to nn spin-orbit term t_soc
        for i in range(self.N):
            for j in range(self.N):
                r = np.linalg.norm(self.coord[i] - self.coord[j])
                if(r == 1):
                    Ham_ud[i, j] = self.t_soc 
    
        Ham = np.block([[Ham_up, Ham_ud], [np.conj(Ham_ud), Ham_dn]])

        evals, evects = np.linalg.eigh(Ham)
        
        return evals, evects
    
    
    def plot_eigenvalues(self):
        num_b = 200
        Bz = np.linspace(0, self.Bmax, num_b)
        eigvals = np.zeros((num_b, 2 * self.N))


        for i in range(num_b):
            eigvals[i], _ = self.calc_eigenvalues(Bz[i])    

        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)

        for i in range(2 * self.N):
            ax.plot(Bz, eigvals[:,i], color='Blue', linewidth=0.8)

        # with open("Energies_B.dat", "w") as fp:
        #     for i in range(num_b):
        #         print('{0.real:.4f}'.format(Bz[i]), '  '.join('{0.real:.4f}'.format(item) for item in eigvals[i,:]), file=fp)
    

        ax.set_xlabel('B (T)')
        ax.set_ylabel('Energy (eV)')
        ax.set_xlim(np.min(Bz), np.max(Bz))
        fig.savefig('Eigenvalues.png', dpi=300, facecolor='w', transparent=False, bbox_inches='tight')



    def plot_map(self):

        print('map plotting settings:')
        print('states are included in the map (mind spin degeneracy) =', self.num_map)
        print('magnetic field value = ', self.B_map, 'T')
        print('map resolution =', self.mapRes)
        print('smearing of gaussian function =', self.smearing)

        def getPsiR(i, x, y, psi):
        
            #basis gaussian functions
            def phi(x,y):
                return np.exp(-(x**2 + y**2)/self.smearing)
        
            psiR = phi(x,y) * complex(0,0)
        
            for num in range(2 * self.N):
                psiR += psi[num, i] * phi(x - self.a * self.coord[num % self.N][0], y - self.a * self.coord[num % self.N][1])
        
            return psiR
        

        evals, evecs = self.calc_eigenvalues(self.B_map)


        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        z = np.zeros((self.mapRes, self.mapRes))

        x = np.linspace(self.a * 0.5, self.a * (self.coord[self.N - 1][0] + 0.5), self.mapRes)
        y = np.copy(x)
        xGrid, yGrid = np.meshgrid(x, y)

        print('Eigenvalues of plotting states (in eV):')
        num_plot = np.array(self.num_map, dtype=int)

        for i in num_plot:
            z += np.abs(getPsiR(i, xGrid, yGrid, evecs))**2
            print(i, evals[i])

        ax.pcolor(x, y, z, cmap='Reds', shading='nearest')
        ax.axis('off')
        ax.set_xlabel('R (nm)')
        ax.set_ylabel('R (nm)')
        ax.set_aspect('equal', 'box')

        fig.savefig('Eigenvectors_map.png', dpi=300, facecolor='w', transparent=False, bbox_inches='tight')


    

