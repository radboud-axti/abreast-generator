# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:59:35 2021

@author: Marta Pinto
@author: Koen Michielsen
"""

import os
import tifffile
import numpy as np
from scipy import interpolate

def export_points(filePath: str, vertices: np.ndarray) -> None:

  filePath += '.obj'
  vert_list = vertices.T.tolist()
  with open(filePath, 'w') as f:
    f.write('# abreast point cloud\n')
    f.write('\n')
    for v in vert_list:
      line = 'v ' + str(v)[1:-1] + '\n'
      f.write(line)

def export_vox(filePath: str, vertices: np.ndarray, dx: float=1.0, dy: float=None, dz: float=None, exportSlices: bool=False) -> None:

  if exportSlices:
    os.makedirs(filePath, exist_ok=True)
  else:
    os.makedirs(os.path.dirname(os.path.abspath(filePath)), exist_ok=True)  

  GRID_STEP = 64

  # 1. Get output grid dimensions
  if not dy:
    dy = dx
  if not dz:
    dz = dx

  maxr = vertices[0,:].max()
  maxz = vertices[2,:].max() + 0.5

  nx = (maxr / dx) / GRID_STEP * 2
  ny = (maxr / dy) / GRID_STEP
  nx = GRID_STEP * round(nx + 1)
  ny = GRID_STEP * round(ny + 1)
  nz = round(maxz / dz)

  x0 = dx*(nx-1)/2
  y0 = dy/2

  # 2. Assign grid points for voxelization
  gridc = np.mgrid[-x0:x0+dx:dx,y0:ny*dy:dy]
  grida = np.arctan2(gridc[0,:,:],gridc[1,:,:])
  gridr = np.sqrt(gridc[0,:,:]**2 + gridc[1,:,:]**2)
  gridz = np.mgrid[0.0:nz*dz:dz] + dz/2.0

  grida = grida.reshape(-1)
  gridc = np.argsort(grida)
  grida = grida[gridc]
  gridc = np.argsort(gridc)

  # 3. Fit bspline to allow interpolation on voxel grid
  tx = np.mgrid[-3.0:max(maxz,nz*dz)+3.0:32j]
  ty = np.mgrid[-1.01*np.pi/2:1.01*np.pi/2:256j]

  splinefit = interpolate.bisplrep(vertices[2,:],vertices[1,:],vertices[0,:], task=-1, tx=tx, ty=ty, nxest=50, nyest=300)

  # 4. Voxelize slices separately to allow fine grids
  if not exportSlices:
    gridv = np.zeros([nx,ny,nz])

  nzfill = np.ceil(np.log10(nz)).astype(np.int8)
  for idx, zz in np.ndenumerate(gridz):
    # Obtain in-plane curve at current height
    gridf = interpolate.bisplev(zz, grida, splinefit)
    gridf = gridf[gridc].reshape(nx,ny)
    # Find points inside curve
    if exportSlices:
      gridv = np.greater(gridf, gridr)
      gridv = gridv.astype(np.uint8)
      # Export slice as .tif image
      fn = filePath + '/slice' + str(idx[0]).zfill(nzfill) + '.tif'
      tifffile.imwrite(fn, gridv, resolution=(1/dx,1/dy), imagej=True, metadata={'spacing': dz, 'unit':'mm'})
    else:
      gridv[:,:,idx[0]] = np.greater(gridf, gridr)
      gridv = gridv.astype(np.uint8)

  if not exportSlices:
    fn = filePath + '.tif'
    tifffile.imwrite(fn, gridv, resolution=(1/dx,1/dy), imagej=True, metadata={'spacing': dz, 'unit':'mm', 'axes': 'ZYX'})
