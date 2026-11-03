import turbojpeg as tjpeg
jpeg = tjpeg.TurboJPEG()

from georef.operators import ProjectionGrid, Projector
from dyn_geo.core.camera_extrinsics import read_cam_params
from dyn_geo.core import img


def project_ls_im(ls, georef_params, undistort_map, pgrid, odir,
                  return_imgs_proj=True, save_to_files=False, save_format='.png'):
    '''
    project images
    :param ls: list of jpeg files to be projected
    :param georef_params: georef parameters, same size as ls
    :param undistort_map: undistort x and y map
    :param pgrid: projection grid info
    :param odir: output directory. Used only if 'save_to_files' is set to True
    :param return_imgs_proj: boolean. If true, function returns arrays of projected images
    :param save_to_files: boolean. If true, function saves projected images in the 'save_format'
    :param save_format: format file of projected images
    '''

    # initialization of output list
    imgs_proj = []

    # projection grid
    projection_grid = ProjectionGrid(pgrid.xmin, pgrid.xmax, pgrid.res, pgrid.ymin, pgrid.ymax, pgrid.res, pgrid.z)

    # project images
    for i, f in enumerate(ls):

        # read im
        im = img.read_jpeg(f, pixel_format=tjpeg.TJPF_BGR)

        # projector
        gp = georef_params[i]
        gp._undistort_map_x = undistort_map[0]
        gp._undistort_map_y = undistort_map[1]
        projector = Projector(gp, projection_grid)

        # project image
        im_proj = projector.project_image(im)
        if return_imgs_proj:
            imgs_proj.append(im_proj.img_proj)

        # save to file
        if save_to_files:
            f_png = odir / ('P_' + f.stem + save_format)
            projector.to_png(f_png)

    if return_imgs_proj:
        return imgs_proj



def run(dir_imgs, start, end, pgrid, dir_cparams, odir):
    '''
    project images

    :param dir_imgs: input directory containing average images to be projected
    :param start: start date for images to be projected
    :param end: end date for images to be projected
    :param pgrid: projection grid infos
    :param dir_cparams: directory containing computed camera parameters
    :param odir: output directory in which projected images are to be stored
    '''

    # load georef parameters for each target image
    t_cparams, georef_params, undistort_map = read_cam_params(dir_cparams, start, end)

    # list of target images
    ls = sorted(dir_imgs.glob('*.jp*g'))
    t_im = [img.get_date(f) for f in ls]

    # keep only images at dates of camera params
    indices = [i for i, val in enumerate(t_im) if val in t_cparams]
    ls = [ls[i] for i in indices]

    # project images
    project_ls_im(ls, georef_params, undistort_map, pgrid, odir, return_imgs_proj=False, save_to_files=True)

    return