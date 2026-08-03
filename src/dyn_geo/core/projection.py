from georef.operators import Georef, ExtrinsicMatrix, ProjectionGrid, Projector
from dyn_geo.core import img

def project_ls_im(ls, georef_params, z_proj=0):

    # initialization of output list
    imgs_proj = []

    # resolution of projection
    res = 0.25

    # projection grid
    projection_grid = ProjectionGrid(-20, 30, res, 15, 50, res, z_proj)

    # project images
    for i, f in enumerate(ls):

        # read im
        im = img.read_jpeg(f)

        # projector
        projector = Projector(georef_params[i], projection_grid)

        # project image
        im_proj = projector.project_image(im)

        imgs_proj.append(im_proj.img_proj)

    return imgs_proj