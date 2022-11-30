# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:14:10 2021

@author: Marta Pinto
@author: Koen Michielsen
"""

import os
import numpy as np  
import abreast

class Abreast:

  def __init__(self) -> None:
    self._cc_slices =  30
    self._cc_angles = 256
    self._rn = np.zeros(15)
    self._bshape = np.zeros([self._cc_angles+1,self._cc_slices])
    self._rndgen = np.random.default_rng()

    ''' Read PCA data from csv '''
    path_cc = os.path.join(os.path.dirname(__file__), "../data/pca_cc/")
    self._PCA_nd = np.genfromtxt(path_cc + "PCA_normal_fit.csv", delimiter=',')
    self._PCA_cd = np.genfromtxt(path_cc + "PCA_cdf.csv", delimiter=',')
    self._PCAu00 = np.genfromtxt(path_cc + "PCAmean.csv", delimiter=',')
    self._PCAu01 = np.genfromtxt(path_cc + "PCA1.csv", delimiter=',')
    self._PCAu02 = np.genfromtxt(path_cc + "PCA2.csv", delimiter=',')
    self._PCAu03 = np.genfromtxt(path_cc + "PCA3.csv", delimiter=',')
    self._PCAu04 = np.genfromtxt(path_cc + "PCA4.csv", delimiter=',')
    self._PCAu05 = np.genfromtxt(path_cc + "PCA5.csv", delimiter=',')
    self._PCAu06 = np.genfromtxt(path_cc + "PCA6.csv", delimiter=',')
    self._PCAu07 = np.genfromtxt(path_cc + "PCA7.csv", delimiter=',')
    self._PCAu08 = np.genfromtxt(path_cc + "PCA8.csv", delimiter=',')
    self._PCAu09 = np.genfromtxt(path_cc + "PCA9.csv", delimiter=',')
    self._PCAu10 = np.genfromtxt(path_cc + "PCA10.csv", delimiter=',')
    self._PCAu11 = np.genfromtxt(path_cc + "PCA11.csv", delimiter=',')
    self._PCAu12 = np.genfromtxt(path_cc + "PCA12.csv", delimiter=',')
    self._PCAu13 = np.genfromtxt(path_cc + "PCA13.csv", delimiter=',')
    self._PCAu14 = np.genfromtxt(path_cc + "PCA14.csv", delimiter=',')
    self._PCAu15 = np.genfromtxt(path_cc + "PCA15.csv", delimiter=',')

    self._ndist_av = self._PCA_nd[:,0]
    self._ndist_sd = self._PCA_nd[:,1]


  @property
  def generated_shape(self) -> np.ndarray:
    return self._bshape

  def generate(self, thickness: float=None, flip_right: bool=False, max_sd: float=2.0, custom_rnd: np.ndarray=None, gauss_approx: bool=True) -> None:
    if isinstance(thickness, int): thickness *= 1.0
    if isinstance(thickness, float):
      if thickness < 30.0:
        raise ValueError("Thickness <30mm not supported.")
      if thickness > 90.0:
        raise ValueError("Thickness >90mm not supported.")

    ''' Assemble breast shape '''
    if gauss_approx:
      if isinstance(custom_rnd, np.ndarray):
        self._rn = custom_rnd * self._ndist_sd + self._ndist_av
      else:
        self._rn = self._rndgen.normal(self._ndist_av,self._ndist_sd)
        self._rn = np.clip(self._rn, self._ndist_av-max_sd*self._ndist_sd, self._ndist_av+max_sd*self._ndist_sd)
    else:
      if not isinstance(custom_rnd, np.ndarray):
        custom_rnd = self._rndgen.uniform(0.0,1.0,size=15)
      for pc in range(0,15):
        self._rn[pc] = np.interp(custom_rnd[pc], self._PCA_cd[1:,0], self._PCA_cd[1:,pc+1])
      custom_rnd = None

    if isinstance(thickness, float):
      self._rn[0] = 3.1988 * thickness - 194.0606

    self._bshape  = self._PCAu00.copy()
    self._bshape += self._rn[ 0] * self._PCAu01
    self._bshape += self._rn[ 1] * self._PCAu02
    self._bshape += self._rn[ 2] * self._PCAu03
    self._bshape += self._rn[ 3] * self._PCAu04
    self._bshape += self._rn[ 4] * self._PCAu05
    self._bshape += self._rn[ 5] * self._PCAu06
    self._bshape += self._rn[ 6] * self._PCAu07
    self._bshape += self._rn[ 7] * self._PCAu08
    self._bshape += self._rn[ 8] * self._PCAu09
    self._bshape += self._rn[ 9] * self._PCAu10
    self._bshape += self._rn[10] * self._PCAu11
    self._bshape += self._rn[11] * self._PCAu12
    self._bshape += self._rn[12] * self._PCAu13
    self._bshape += self._rn[13] * self._PCAu14
    self._bshape += self._rn[14] * self._PCAu15

    if flip_right:
      self._bshape[:256,:] = np.flipud(self._bshape[:256,:])

    self._bshape[:256,:] *= 200.0

  def export(self, filePath: str, format: str = "points", **kwargs)-> None:

    if format == "points":
      abreast.export_points(filePath, self._get_vertices(cartesian=True))
    #elif format == "mesh":
    #  pass
    elif format == "vox":
      abreast.export_vox(filePath, self._get_vertices(), **kwargs)
    else:
      raise ValueError("Output format {} not known.".format(format))

  def _get_vertices(self, cartesian: bool = False) -> np.ndarray:
    vert = np.zeros((3,self._cc_angles*self._cc_slices))

    r_values = self._bshape[:256,:]
    a_values = np.tile(np.linspace(-np.pi/2, np.pi/2, self._cc_angles, endpoint=True),(self._cc_slices,1)).transpose()
    z_values = np.tile(self._bshape[256,:],(self._cc_angles,1))

    if cartesian:
     x_values = r_values * np.cos(a_values)
     y_values = r_values * np.sin(a_values)
     r_values = x_values
     a_values = y_values

    vert[0,:] = r_values.flatten()
    vert[1,:] = a_values.flatten()
    vert[2,:] = z_values.flatten()

    return vert

def main():
  mybreast = Abreast()
  mybreast.generate(thickness = 53, flip_right=True, gauss_approx=False)
  print(mybreast.generated_shape[256,:])
  #mybreast.export('test_points','points')
  #mybreast.export('test_voxelized','vox',dx=0.25)

if __name__ == '__main__':
  main()
