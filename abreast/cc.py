# -*- coding: utf-8 -*-
"""
@author: Marta Pinto
@author: Koen Michielsen
@author: Pim van den Berg
"""

import numpy as np
import tifffile
from scipy import interpolate
from pathlib import Path
import struct

'''
Example usage:

# Generate a single breast
mybreast = Abreast()
mybreast.generate(thickness=60)
mybreast.export_voxelized(Path("voxelized.tiff"))

# Generate many breasts using the vectorized implementation
mybreast = Abreast()
bshapes = mybreast.generate_many_shapes(amount=1000, thickness=80.0, gauss_approx=False)
for i in range(len(bshapes)):
    mybreast.generated_shape = bshapes[i, :, :]
    mybreast.export_voxelized(f"test_{i:0>3}.tiff")
'''
class Abreast:
    def __init__(self, seed=None) -> None:
        self._cc_slices = 30
        self._cc_angles = 256
        self._rn = np.zeros(15)
        self._bshape = np.zeros([self._cc_angles+1, self._cc_slices])
        self._rndgen = np.random.default_rng(seed)

        ''' Read PCA data from csv '''
        path_cc = Path(__file__).parent / "data/pca_cc/"
        self._PCA_nd = np.genfromtxt(path_cc / "PCA_normal_fit.csv", delimiter=',')
        self._PCA_cd = np.genfromtxt(path_cc / "PCA_cdf.csv", delimiter=',')
        self._PCAu00 = np.genfromtxt(path_cc / "PCAmean.csv", delimiter=',')
        for i in range(1, 16):
            setattr(self, f"_PCAu{i:0>2}", np.genfromtxt(path_cc / f"PCA{i}.csv", delimiter=','))

        self._ndist_av = self._PCA_nd[:, 0]
        self._ndist_sd = self._PCA_nd[:, 1]


    @property
    def generated_shape(self) -> np.ndarray:
        return self._bshape
    

    @generated_shape.setter
    def generated_shape(self, val):
        if not isinstance(val, np.ndarray):
            raise ValueError("Generated shape must be a numpy array")
        if not np.array_equal(self._bshape.shape, val.shape):
            raise ValueError(f"Generated shape must have the correct shape ({self._bshape.shape})")
        
        self._bshape = val
    

    ''' Generates a random breast shape. '''
    def generate_many_shapes(
        self,
        amount: int,
        thickness: float = None,
        flip_right: bool = False,
        max_sd: float = 2.0,
        gauss_approx: bool = True
    ) -> np.ndarray:
        if isinstance(thickness, int):
            thickness *= 1.0
        if isinstance(thickness, float):
            if thickness < 30.0:
                raise ValueError("Thickness <30mm not supported.")
            if thickness > 90.0:
                raise ValueError("Thickness >90mm not supported.")
        
        ''' Assemble breast shape '''
        if gauss_approx:
            components = self._rndgen.normal(self._ndist_av, self._ndist_sd, size=(amount, 15))
            # TODO: this creates an incorrect distribution?
            components = np.clip(components, self._ndist_av-max_sd * self._ndist_sd, self._ndist_av+max_sd*self._ndist_sd)
        else:
            uniform_noise = self._rndgen.uniform(0.0, 1.0, size=(15, amount))
            components = np.zeros((amount, 15))
            for pc in range(0, 15):
                components[:, pc] = np.interp(uniform_noise[pc], self._PCA_cd[1:, 0], self._PCA_cd[1:, pc+1])

        if isinstance(thickness, float):
            components[:, 0] = _thickness_to_pc0(thickness)

        n_bshapes = np.zeros((amount, self._PCAu00.shape[0], self._PCAu00.shape[1]))
        n_bshapes[:] = self._PCAu00
        for i in range(15):
            n_bshapes += components[:, i][:, None, None] * getattr(self, f"_PCAu{i+1:0>2}")

        if flip_right:
            n_bshapes[:, :256, :] = n_bshapes[:, :256, :][:, ::-1, :]

        n_bshapes[:, :256, :] *= 200.0
        
        return n_bshapes


    ''' Generates a random breast shape. '''
    def generate(
        self,
        thickness: float = None,
        flip_right: bool = False,
        max_sd: float = 2.0,
        custom_rnd: np.ndarray = None,
        gauss_approx: bool = True
    ):
        if isinstance(thickness, int):
            thickness *= 1.0
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
                self._rn = self._rndgen.normal(self._ndist_av, self._ndist_sd)
                # TODO: this creates an incorrect distribution?
                self._rn = np.clip(self._rn, self._ndist_av-max_sd * self._ndist_sd, self._ndist_av+max_sd*self._ndist_sd)
        else:
            if not isinstance(custom_rnd, np.ndarray):
                custom_rnd = self._rndgen.uniform(0.0, 1.0, size=15)
            for pc in range(0, 15):
                self._rn[pc] = np.interp(custom_rnd[pc], self._PCA_cd[1:, 0], self._PCA_cd[1:, pc+1])
            custom_rnd = None

        if isinstance(thickness, float):
            self._rn[0] = _thickness_to_pc0(thickness)

        self._bshape = self._PCAu00.copy()
        for i in range(15):
            self._bshape += self._rn[i] * getattr(self, f"_PCAu{i+1:0>2}")
        
        if flip_right:
            self._bshape[:256, :] = np.flipud(self._bshape[:256, :])

        self._bshape[:256, :] *= 200.0


    ''' Deprecated: Use export_points or export_voxelized instead. '''
    def export(self, filePath: str, format: str = "points", **kwargs) -> None:
        if format == "points":
            _export_obj(Path(filePath).with_suffix(".obj"), self._get_vertices(cartesian=True))
        elif format == "vox":
            _export_voxelized_tiff(Path(filePath).with_suffix(".tif"), self._get_vertices(cartesian=False), **kwargs)
        else:
            raise ValueError("Output format {} not known.".format(format))
    

    ''' Export the shape as *.obj or *.ply. '''
    def export_points(self, filePath: str | Path) -> None:
        if isinstance(filePath, str):
            filePath = Path(filePath)

        if filePath.suffix == "":
            filePath = filePath.with_suffix(".obj")
        if filePath.suffix != ".obj" and filePath.suffix != "*.ply":
            raise ValueError("Only exporting as *.obj or *.ply is implemented.")
        
        if filePath.suffix == ".obj":
            _export_obj(filePath, self._get_vertices(cartesian=True))
        elif filePath.suffix == ".ply":
            _export_ply(filePath, self._get_vertices(cartesian=True))


    ''' Export the shape as a voxelized binary u8 .tiff stack with values 0 and 255. '''
    def export_voxelized(self, filePath: str | Path) -> None:
        if isinstance(filePath, str):
            filePath = Path(filePath)

        if filePath.suffix == "":
            filePath = filePath.with_suffix(".tiff")
        if filePath.suffix != ".tif" and filePath.suffix != ".tiff":
            raise ValueError("Only exporting as *.tif/*.tiff is implemented.")
        _export_voxelized_tiff(filePath, self._get_vertices(cartesian=False), mult=255)


    def _get_vertices(self, cartesian: bool = False) -> np.ndarray:
        vert = np.zeros((3, self._cc_angles*self._cc_slices))

        r_values = self._bshape[:256, :]
        a_values = np.tile(np.linspace(-np.pi/2, np.pi/2, self._cc_angles, endpoint=True), (self._cc_slices, 1)).transpose()
        z_values = np.tile(self._bshape[256, :], (self._cc_angles, 1))

        if cartesian:
            x_values = r_values * np.cos(a_values)
            y_values = r_values * np.sin(a_values)
            r_values = x_values
            a_values = y_values

        vert[0, :] = r_values.flatten()
        vert[1, :] = a_values.flatten()
        vert[2, :] = z_values.flatten()

        return vert


def _thickness_to_pc0(thickness: float):
    return 3.1988 * thickness - 194.0606


# Writes text-based .obj
def _export_obj(filePath: Path, vertices: np.ndarray) -> None:
    vert_list = vertices.T.tolist()
    with open(filePath, 'w') as f:
        f.write('# abreast point cloud\n')
        f.write('\n')
        for v in vert_list:
            line = 'v ' + str(v)[1:-1] + '\n'
            f.write(line)


# Writes binary f32 .ply
def _export_ply(filePath: Path, vertices: np.ndarray):
    with open(filePath, "wb") as f:
        # Write the PLY header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(vertices)}\n".encode("ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"end_header\n")

        # Write vertices
        for i in range(len(vertices)):
            f.write(struct.pack("<3f", *vertices[i]))


# Writes binary .tiff
def _export_voxelized_tiff(filePath: Path, vertices: np.ndarray, dx: float = 1.0, dy: float = None, dz: float = None, exportSlices: bool = False, mult=1) -> None:
    if exportSlices:
        filePath = filePath.with_suffix("")
        filePath.mkdir(parents=True, exist_ok=True)
    else:
        filePath.parent.mkdir(parents=True, exist_ok=True)

    GRID_STEP = 64

    # 1. Get output grid dimensions
    if not dy:
        dy = dx
    if not dz:
        dz = dx

    maxr = vertices[0, :].max()
    maxz = vertices[2, :].max() + 0.5

    nx = (maxr / dx) / GRID_STEP * 2
    ny = (maxr / dy) / GRID_STEP
    nx = int(GRID_STEP * round(nx + 1))
    ny = int(GRID_STEP * round(ny + 1))
    nz = int(round(maxz / dz))

    x0 = dx*(nx-1)/2
    y0 = dy/2

    # 2. Assign grid points for voxelization
    gridc = np.mgrid[-x0:x0+dx:dx, y0:ny*dy:dy]
    grida = np.arctan2(gridc[0, :, :], gridc[1, :, :])
    gridr = np.sqrt(gridc[0, :, :]**2 + gridc[1, :, :]**2)
    gridz = np.mgrid[0.0:nz*dz:dz] + dz/2.0

    grida = grida.reshape(-1)
    gridc = np.argsort(grida)
    grida = grida[gridc]
    gridc = np.argsort(gridc)

    # 3. Fit bspline to allow interpolation on voxel grid
    tx = np.mgrid[-3.0:max(maxz, nz*dz)+3.0:32j]
    ty = np.mgrid[-1.01*np.pi/2:1.01*np.pi/2:256j]

    splinefit = interpolate.bisplrep(
        vertices[2, :],
        vertices[1, :],
        vertices[0, :],
        task=-1,
        tx=tx, ty=ty,
        nxest=50, nyest=300
    )

    # 4. Voxelize slices separately to allow fine grids
    if not exportSlices:
        gridv = np.zeros([nx, ny, nz])

    nzfill = np.ceil(np.log10(nz)).astype(np.int8)
    for idx, zz in np.ndenumerate(gridz):
        # Obtain in-plane curve at current height
        gridf = interpolate.bisplev(zz, grida, splinefit)
        gridf = gridf[gridc].reshape(nx, ny)
        # Find points inside curve
        if exportSlices:
            gridv = np.greater(gridf, gridr)
            gridv = gridv.astype(np.uint8)
            # Export slice as .tif image
            fn = filePath / f"slice{str(idx[0]).zfill(nzfill)}.tif"
            tifffile.imwrite(fn, gridv, resolution=(1/dx, 1/dy), imagej=True, metadata={'spacing': dz, 'unit': 'mm'}, compression="zlib")
        else:
            gridv[:, :, idx[0]] = np.greater(gridf, gridr)
            gridv = gridv.astype(np.uint8)

    if not exportSlices:
        # Fix export ordering to index along Z in the first axis
        # These two swaps make the ordering the same as when viewing a stack of slices exported using exportSlices=False.
        gridv = np.swapaxes(gridv, 0, 2)
        gridv = np.swapaxes(gridv, 1, 2)
        tifffile.imwrite(filePath, gridv*mult, resolution=(1/dx,1/dy), imagej=True, metadata={'spacing': dz, 'unit':'mm'}, compression="zlib")
