from georef.operators import Georef, ExtrinsicMatrix, ProjectionGrid, Projector
from dyn_geo.core import img

def project_ls_im(ls, georef_params, pgrid, z_proj=0):

    # initialization of output list
    imgs_proj = []

    # projection grid
    projection_grid = ProjectionGrid(pgrid.xmin, pgrid.xmax, pgrid.res, pgrid.ymin, pgrid.ymax, pgrid.res, pgrid.z)

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