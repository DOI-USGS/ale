
import tempfile

import json
import knoten
import pvl
import pyproj
import csmapi

from numbers import Number

import numpy as np
import pandas as pd

from pysis import isis
from pysis.exceptions import ProcessError

import holoviews as hv
from holoviews import opts, dim
from bokeh.models import HoverTool


def reproject(record, semi_major, semi_minor, source_proj, dest_proj, **kwargs):
    """
    Thin wrapper around PyProj's Transform() function to transform 1 or more three-dimensional
    point from one coordinate system to another. If converting between Cartesian
    body-centered body-fixed (BCBF) coordinates and Longitude/Latitude/Altitude coordinates,
    the values input for semi-major and semi-minor axes determine whether latitudes are
    planetographic or planetocentric and determine the shape of the datum for altitudes.
    If semi_major == semi_minor, then latitudes are interpreted/created as planetocentric
    and altitudes are interpreted/created as referenced to a spherical datum.
    If semi_major != semi_minor, then latitudes are interpreted/created as planetographic
    and altitudes are interpreted/created as referenced to an ellipsoidal datum.

    Parameters
    ----------
    record : object
          Pandas series object

    semi_major : float
              Radius from the center of the body to the equater

    semi_minor : float
              Radius from the pole to the center of mass

    source_proj : str
                      Pyproj string that defines a projection space ie. 'geocent'

    dest_proj : str
                   Pyproj string that defines a project space ie. 'latlon'

    Returns
    -------
    : list
    Transformed coordinates as y, x, z

    """
    source_pyproj = pyproj.Proj(proj = source_proj, a = semi_major, b = semi_minor)
    dest_pyproj = pyproj.Proj(proj = dest_proj, a = semi_major, b = semi_minor)

    y, x, z = pyproj.transform(source_pyproj, dest_pyproj, record[0], record[1], record[2], **kwargs)

    return y, x, z


def point_info(cube_path, x, y, point_type, allow_outside=False):
    """
    Use Isis's campt to get image/ground point info from an image

    Parameters
    ----------
    cube_path : str
                path to the input cube

    x : float
        point in the x direction. Either a sample or a longitude value
        depending on the point_type flag

    y : float
        point in the y direction. Either a line or a latitude value
        depending on the point_type flag

    point_type : str
                 Options: {"image", "ground"}
                 Pass "image" if  x,y are in image space (sample, line) or
                 "ground" if in ground space (longitude, lattiude)

    Returns
    -------
    : PvlObject
      Pvl object containing campt returns
    """
    point_type = point_type.lower()

    if point_type not in {"image", "ground"}:
        raise Exception(f'{point_type} is not a valid point type, valid types are ["image", "ground"]')


    if isinstance(x, Number) and isinstance(y, Number):
        x, y = [x], [y]

    with tempfile.NamedTemporaryFile("w+") as f:
        # ISIS wants points in a file, so write to a temp file
        if point_type == "ground":
            # campt uses lat, lon for ground but sample, line for image.
            # So swap x,y for ground-to-image calls
            x,y = y,x
        elif point_type == "image":
            # convert to ISIS pixels
            x = np.add(x, .5)
            y = np.add(y, .5)

        f.write("\n".join(["{}, {}".format(xval,yval) for xval,yval in zip(x, y)]))
        f.flush()
        try:
            pvlres = isis.campt(from_=cube_path, coordlist=f.name, allowoutside=allow_outside, usecoordlist=True, coordtype=point_type)
        except ProcessError as e:
            warn(f"CAMPT call failed, image: {cube_path}\n{e.stderr}")
            return

        pvlres = pvl.loads(pvlres)
        if len(x) > 1 and len(y) > 1:
            for r in pvlres:
                # convert all pixels to PLIO pixels from ISIS
                r[1]["Sample"] -= .5
                r[1]["Line"] -= .5
        else:
            pvlres["GroundPoint"]["Sample"] -= .5
            pvlres["GroundPoint"]["Line"] -= .5

    return pvlres


def reprojection_diff(isd, cube, nx=4, ny=8):
    """
    """
    hv.extension('bokeh')

    isdjson = json.load(open(isd))

    nlines = isdjson['image_lines']
    nsamples = isdjson['image_samples']

    # generate meshgrid
    xs, ys = np.mgrid[0:nsamples:nsamples/nx, 0:nlines:nlines/ny]
    xs, ys = xs.flatten(), ys.flatten()

    csmcam = knoten.csm.create_csm(isd)

    # CS101 C++ programming class style dividers
    ##############################
    isis_pts = point_info(cube, xs, ys, 'image')
    isisgnds = np.asarray([g[1]['BodyFixedCoordinate'].value for g in isis_pts])
    csm_pts = np.asarray([[p.samp, p.line] for p in [csmcam.groundToImage(csmapi.EcefCoord(*(np.asarray(bf)*1000))) for bf in isisgnds]])
    isis2csm_diff = csm_pts - np.asarray([xs,ys]).T
    isis2csm_diffmag = np.linalg.norm(isis2csm_diff, axis=1)
    isis2csm_angles = np.arctan2(*isis2csm_diff.T[::-1])

    data = np.asarray([csm_pts.T[0], csm_pts.T[1], xs, ys,  isis2csm_diff.T[0], isis2csm_diff.T[1], isis2csm_diffmag, isis2csm_angles]).T
    data = pd.DataFrame(data, columns=['x', 'y', 'isisx','isisy', 'diffx', 'diffy', 'magnitude', 'angles'])

    isis2ground2csm_plot = hv.VectorField((data['x'], data['y'], data['angles'], data['magnitude']), group='isis2ground2csmimage' ).opts(opts.VectorField(pivot='tail', colorbar=True, cmap='coolwarm', title='ISIS2Ground->CSM2Image Pixel Diff', arrow_heads=True, magnitude='Magnitude', color=dim('Magnitude')))
    isis2ground2csm_plot = isis2ground2csm_plot.redim(x='sample', y='line')
    isis2ground2csm_plot = isis2ground2csm_plot.opts(plot=dict(width=500, height=1000))
    isis2ground2csm_plot = isis2ground2csm_plot*hv.Points(data, group='isis2ground2csmimage').opts(size=5, tools=['hover'], invert_yaxis=True)
    isis2ground2csm_plot = isis2ground2csm_plot.redim.range(a=(data['magnitude'].min(), data['magnitude'].max()))
    ##############################

    ##############################
    csmgnds = np.asarray([[p.x, p.y, p.z] for p in [csmcam.imageToGround(csmapi.ImageCoord(y,x), 0) for x,y in zip(xs,ys)]])
    csmlon, csmlat, _ = reproject(csmgnds.T, isdjson['radii']['semimajor'], isdjson['radii']['semiminor'], 'geocent', 'latlong')
    isis_imgpts = point_info(cube, csmlon, csmlat, 'ground')
    isis_imgpts = np.asarray([(p[1]['Sample'], p[1]['Line']) for p in isis_imgpts])

    csm2isis_diff = isis_imgpts - np.asarray([xs,ys]).T
    csm2isis_diffmag = np.linalg.norm(csm2isis_diff, axis=1)
    csm2isis_angles = np.arctan2(*csm2isis_diff.T[::-1])

    csm2isis_data = np.asarray([xs, ys, isis_imgpts.T[0], isis_imgpts.T[1], csm2isis_diff.T[0], csm2isis_diff.T[1], csm2isis_diffmag, csm2isis_angles]).T
    csm2isis_data = pd.DataFrame(csm2isis_data, columns=['x', 'y', 'csmx','csmy', 'diffx', 'diffy', 'magnitude', 'angles'])

    csm2ground2isis_plot = hv.VectorField((csm2isis_data['x'], csm2isis_data['y'], csm2isis_data['angles'], csm2isis_data['magnitude']),  group='csmground2image2isis').opts(opts.VectorField(pivot='tail', colorbar=True, cmap='coolwarm', title='CSM2Ground->ISIS2Image Pixel Diff', arrow_heads=True, magnitude='Magnitude', color=dim('Magnitude')))
    csm2ground2isis_plot = csm2ground2isis_plot.redim(x='sample', y='line')
    csm2ground2isis_plot = csm2ground2isis_plot.opts(plot=dict(width=500, height=1000))
    csm2ground2isis_plot = csm2ground2isis_plot*hv.Points(csm2isis_data, group='csmground2image2isis').opts(size=5, tools=['hover'], invert_yaxis=True)
    ###############################

    ###############################
    isis_lonlat = np.asarray([[p[1]['PositiveEast360Longitude'].value, p[1]['PlanetocentricLatitude'].value] for p in isis_pts])
    csm_lonlat = np.asarray([csmlon+360, csmlat]).T

    isiscsm_difflatlon = isis_lonlat - csm_lonlat
    isiscsm_difflatlonmag = np.linalg.norm(isiscsm_difflatlon, axis=1)
    isiscsm_angleslatlon = np.arctan2(*isiscsm_difflatlon.T[::-1])

    isiscsm_latlondata = np.asarray([isis_lonlat.T[0], isis_lonlat.T[1], csm_lonlat.T[0], csm_lonlat.T[1], isiscsm_difflatlon.T[0], isiscsm_difflatlon.T[1], isiscsm_difflatlonmag, isiscsm_angleslatlon]).T
    isiscsm_latlondata = pd.DataFrame(isiscsm_latlondata, columns=['isislon', 'isislat', 'csmlon','csmlat', 'difflon', 'difflat', 'magnitude', 'angles'])

    isiscsm_plotlatlon = hv.VectorField((isiscsm_latlondata['isislon'], isiscsm_latlondata['isislat'], isiscsm_latlondata['angles'], isiscsm_latlondata['magnitude']), group='isisvscsmlatlon').opts(opts.VectorField(pivot='tail', colorbar=True, cmap='coolwarm', title='Image2Ground latlon Diff', arrow_heads=True, magnitude='Magnitude', color=dim('Magnitude')))
    isiscsm_plotlatlon = isiscsm_plotlatlon.redim(x='longitude', y='latitude')
    isiscsm_plotlatlon = isiscsm_plotlatlon.opts(plot=dict(width=500, height=1000))
    isiscsm_plotlatlon = isiscsm_plotlatlon*hv.Points(isiscsm_latlondata, ['isislon', 'isislat'], group='isisvscsmlatlon').opts(size=5, tools=['hover'], invert_yaxis=True)
    ###############################

    return isis2ground2csm_plot, csm2ground2isis_plot, isiscsm_plotlatlon,  data, csm2isis_data, isiscsm_latlondata

