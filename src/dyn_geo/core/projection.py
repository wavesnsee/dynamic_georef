import cv2
from georef.operators import Georef, ExtrinsicMatrix, ProjectionGrid, Projector
from dyn_geo.core import img

def project_ls_im(ls, georef_params, undistort_map, pgrid, z_proj=0):

    # initialization of output list
    imgs_proj = []

    # projection grid
    projection_grid = ProjectionGrid(pgrid.xmin, pgrid.xmax, pgrid.res, pgrid.ymin, pgrid.ymax, pgrid.res, pgrid.z)

    # project images
    for i, f in enumerate(ls):

        # read im
        im = img.read_jpeg(f)

        # projector
        gp = georef_params[i]
        gp._undistort_map_x = undistort_map[0]
        gp._undistort_map_y = undistort_map[1]
        projector = Projector(gp, projection_grid)

        # project image
        im_proj = projector.project_image(im)
        imgs_proj.append(im_proj.img_proj)

    return imgs_proj