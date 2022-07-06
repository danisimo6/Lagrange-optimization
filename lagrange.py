import numpy as np 
import matplotlib.pyplot as plt


class lagrange_mult:
    def __init__(self, r0, lam, h ):
        self.r = r0
        self.lam = lam
        self.h = h

    def Fct(self,r):
        """
        Function to be minimized
        """
        return np.sum(r**2)

    def Gct(self,r):
        """
        Constraint
        """    
        return r[1]-r[0]

    def grad(self,f):
        """
        Calulate the gradient of the input function
        input: scalar function
        output: 2-vector
        """
        dx = np.array([self.h,0])
        dy = np.array([0,self.h])
        gradx = (f(self.r+dx)-f(self.r))/self.h
        grady = (f(self.r+dy)-f(self.r))/self.h

        return np.array([gradx,grady])

    def optimization(self,nstep):
        histx = np.zeros(nstep)
        histy = np.zeros(nstep)
        for i in range(nstep):
            self.r += -self.grad(self.Fct)*self.h - self.lam*self.grad(self.Gct)*self.h    
            self.lam += self.Gct(self.r)*self.h
            histx[i] = self.r[0]
            histy[i] = self.r[1]
            
        return histx,histy   


prova = lagrange_mult(np.array([5.,2.],dtype='float64'),5,0.01)
histx, histy = prova.optimization(400)

j=0
for i in range(400):
    plt.plot(histx[0:i],histy[0:i], color='blue')
    plt.plot(np.linspace(-10,10,100), -np.linspace(-10,10,100), color='red', linestyle='--')
    plt.grid(zorder=0)
    plt.xlim(-10,10)
    plt.ylim(-10,10)

    if i%10 == 0:
        plt.savefig('/home/dani/Documents/python/imanim/tmp%04d.png' % j, dpi = 200)
        j+=1
        
    #plt.draw()
    #plt.pause(0.005)
      
#plt.show()

